"""Plain neural network (no ODE solver) for the sled on moguls.

The network learns the one-step map ``state(t) -> state(t + h)`` with
``h = STEP * dt``. Its inputs are 5 features of (y, v, t) and its
outputs are the changes (dy, dv) over h. Long trajectories are obtained
by feeding the network its own output again and again.

Edit the settings below and run::

    python python/train_mlp.py
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from prepare_data import prepare_data

plt.switch_backend("Agg")  # figures are only saved, no window is needed

# ------------------------------------------------------ settings ---
# ROOT = Path(__file__).resolve().parent.parent
DATA = Path("data/sleds_b0.5_u3.5.zarr")
OUT_DIR = Path("checkpoints")

HIDDEN = 64                  # neurons per hidden layer
STEP = 5                     # h = STEP * dt = 0.05 s: how far it jumps
BATCH = 4096
LR = 1e-3
EPOCHS = 20
SEED = 0
ROLLOUT_SECONDS = (1, 3, 5, 10, 25)  # report rollout error at these t
PLOT_STEPS = [0, 200, 500, 1000]     # data steps in the plot
# ---------------------------------------------------------------------


def make_features(y, v, t, meta, v_scale):
    """Build the 5 network input features from the state and the time.

    The slope is periodic in both directions, so the cross-slope
    position enters only through ``sin(q*y)`` and ``cos(q*y)``, and the
    down-slope position ``x = u*t`` only through ``sin(p*x)`` and
    ``cos(p*x)``, with ``p = 2*pi/Lx`` and ``q = 2*pi/Ly``.

    Parameters
    ----------
    y : torch.Tensor
        Cross-slope positions in meters, dtype float32, any shape
        ``(...)``. In practice ``(N,)`` in `rollout` and ``(N, n)`` in
        `make_pairs`. Wrapped or unwrapped gives the same features.
    v : torch.Tensor
        Cross-slope velocities in m/s, dtype float32, same shape as
        ``y``.
    t : torch.Tensor
        Time in seconds, dtype float32. A scalar (0-d tensor) or any
        shape that can be expanded to the shape of ``y``, e.g. ``(n,)``
        when ``y`` is ``(N, n)``. The down-slope position is
        ``x = meta["u"] * t`` (all sleds start at x0 = 0).
    meta : dict
        Dataset attributes. Keys used: ``"Lx"`` and ``"Ly"`` (mogul
        wavelengths in meters, float) and ``"u"`` (down-slope speed in
        m/s, float).
    v_scale : float
        Constant that ``v`` is divided by (standard deviation of ``v``
        in the training sleds).

    Returns
    -------
    torch.Tensor
        dtype float32, shape ``(*y.shape, 5)``. The last axis holds
        ``[sin(q*y), cos(q*y), v / v_scale, sin(p*x), cos(p*x)]``.
    """
    p, q = 2 * math.pi / meta["Lx"], 2 * math.pi / meta["Ly"]
    x = (meta["u"] * t).expand_as(y)  # x0 = 0 in the data
    return torch.stack([torch.sin(q * y), torch.cos(q * y), v / v_scale,
                        torch.sin(p * x), torch.cos(p * x)], dim=-1)


def make_pairs(data, t, meta, v_scale):
    """Cut trajectories into (input, target) rows for regression.

    Every sled is sampled every ``STEP`` data steps (non-overlapping
    jumps). Each row pairs the features of the state at one time with
    the change of (y, v) over the following ``STEP`` data steps. Uses
    the module constant ``STEP``.

    Parameters
    ----------
    data : torch.Tensor
        Trajectories, dtype float32, shape ``(N, T, 2)``: N sleds, T
        time steps, last axis ``(y, v)`` with ``y`` in meters
        (unwrapped, continuous in time) and ``v`` in m/s.
    t : torch.Tensor
        Time grid in seconds, dtype float32, shape ``(T,)``.
    meta : dict
        Dataset attributes. Keys used: ``"Lx"``, ``"Ly"`` and ``"u"``
        (all float, see `make_features`).
    v_scale : float
        Constant that ``v`` is divided by in the features.

    Returns
    -------
    X : torch.Tensor
        dtype float32, shape ``(M, 5)``. Features of the current state
        (see `make_features`).
    Y : torch.Tensor
        dtype float32, shape ``(M, 2)``. Columns ``(dy, dv)``: the
        change of position (meters) and of velocity (m/s) over
        ``STEP`` data steps. Not standardized.

    Notes
    -----
    ``M = N * n`` with ``n = len(range(0, T - STEP, STEP))``, i.e.
    n = 500 for T = 2501 and STEP = 5. Rows are ordered sled by sled.
    """
    idx = torch.arange(0, data.shape[1] - STEP, STEP)
    now, nxt = data[:, idx], data[:, idx + STEP]  # (N, n, 2)
    X = make_features(now[..., 0], now[..., 1], t[idx], meta,
                      v_scale).reshape(-1, 5)
    dy = nxt[..., 0] - now[..., 0]
    dv = nxt[..., 1] - now[..., 1]
    return X, torch.stack([dy, dv], dim=-1).reshape(-1, 2)


@torch.no_grad()
def rollout(net, z0, t, meta, v_scale, Y_mean, Y_std):
    """Feed the network its own output to build long trajectories.

    Starting from ``z0`` at ``t[0]``, repeatedly predict the change of
    (y, v) over ``STEP`` data steps, undo the standardization of the
    targets, update the state and use it as the next input. All sleds
    advance together. Runs without gradient tracking
    (``torch.no_grad``), so the result does not require grad.

    Parameters
    ----------
    net : torch.nn.Module
        Trained network mapping ``(M, 5)`` features to ``(M, 2)``
        standardized changes ``(dy, dv)``, dtype float32.
    z0 : torch.Tensor
        Initial states, dtype float32, shape ``(N, 2)``, columns
        ``(y, v)`` at the first time of ``t`` (``y`` in meters, ``v``
        in m/s).
    t : torch.Tensor
        Full time grid of the data in seconds, dtype float32, shape
        ``(T,)``. Only the entries ``t[n * STEP]`` are used.
    meta : dict
        Dataset attributes. Keys used: ``"Lx"``, ``"Ly"`` and ``"u"``
        (all float, see `make_features`).
    v_scale : float
        Constant that ``v`` is divided by in the features.
    Y_mean, Y_std : torch.Tensor
        Mean and standard deviation of the training targets
        ``(dy, dv)``, dtype float32, shape ``(2,)``. They convert the
        network output back to meters and m/s.

    Returns
    -------
    torch.Tensor
        dtype float32, shape ``(N, n_steps + 1, 2)`` with
        ``n_steps = (T - 1) // STEP``. Entry ``[:, k]`` is the
        predicted state ``(y, v)`` at time ``t[k * STEP]``; ``y`` is
        not wrapped and ``[:, 0]`` equals ``z0``.
    """
    n_steps = (len(t) - 1) // STEP
    z, out = z0, [z0]
    for n in range(n_steps):
        inc = net(make_features(z[:, 0], z[:, 1], t[n * STEP], meta,
                                v_scale)) * Y_std + Y_mean
        z = z + inc  # columns of inc are (dy, dv)
        out.append(z)
    return torch.stack(out, dim=1)


def plot_phase_space(true, pred, meta, path):
    """Save a figure comparing the true and the predicted phase space.

    One column per entry of ``PLOT_STEPS``, titled with the distance
    travelled down-slope (``step * dt * u``, in meters). Top row: true
    ``(y, v)`` of all sleds. Bottom row: network rollout. The
    unwrapped ``y`` is wrapped into [-Ly/2, Ly/2) for display only.

    Parameters
    ----------
    true : numpy.ndarray
        True states, dtype float32, shape ``(N, n, 2)``, sampled every
        ``STEP`` data steps (``n = (T - 1) // STEP + 1``). Last axis
        ``(y, v)`` with ``y`` unwrapped.
    pred : numpy.ndarray
        Predicted states from `rollout`, converted to numpy. Same
        shape and dtype as ``true``.
    meta : dict
        Dataset attributes. Keys used: ``"dt"`` (data time step in
        seconds), ``"u"`` (down-slope speed in m/s) and ``"Ly"``
        (cross-slope period in meters), all float.
    path : pathlib.Path or str
        Output PNG file.

    Returns
    -------
    None
        The figure is written to ``path`` and closed.
    """
    half = meta["Ly"] / 2
    fig, axes = plt.subplots(2, len(PLOT_STEPS), sharex=True, sharey=True,
                             figsize=(3 * len(PLOT_STEPS), 6))
    for j, step in enumerate(PLOT_STEPS):
        n = step // STEP
        for row, data in enumerate((true, pred)):
            y = (data[:, n, 0] + half) % meta["Ly"] - half  # display only
            axes[row, j].scatter(y, data[:, n, 1], s=2, color="k")
        axes[0, j].set_title(f"{round(step * meta['dt'] * meta['u'])} m")
        axes[1, j].set_xlabel("y (m)")
    axes[0, 0].set_ylabel("true: v (m/s)")
    axes[1, 0].set_ylabel("network: v (m/s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    """Train the one-step network, test its rollout and save results.

    Steps: load and split the data (`prepare_data`); build the (input,
    target) pairs and standardize the targets with training statistics;
    train the network with Adam, printing the one-step validation error
    per epoch; roll out the validation sleds from t = 0 and print the
    rms error at the times in ``ROLLOUT_SECONDS``; save the phase-space
    figure and the model.

    All settings are module constants: ``DATA``, ``OUT_DIR``,
    ``HIDDEN``, ``STEP``, ``BATCH``, ``LR``, ``EPOCHS``, ``SEED``,
    ``ROLLOUT_SECONDS`` and ``PLOT_STEPS``.

    Returns
    -------
    None
        Prints progress to stdout and writes two files to ``OUT_DIR``:
        ``mlp_phase_space.png`` and ``mlp.pt`` (a dict with the
        ``state_dict``, ``Y_mean``, ``Y_std``, ``v_scale``, ``hidden``,
        ``step`` and ``meta``).
    """
    torch.manual_seed(SEED)
    D = prepare_data(DATA)
    meta, t, v_scale = D["meta"], D["t"], D["v_scale"]

    X_tr, Y_tr = make_pairs(D["train"], t, meta, v_scale)
    X_va, Y_va = make_pairs(D["val"], t, meta, v_scale)
    # StandardScaler on the TARGETS, fitted on the training pairs only.
    # Here (unlike the ODE state) it is fine: dy and dv are ordinary
    # regression targets.
    Y_mean, Y_std = Y_tr.mean(0), Y_tr.std(0)
    Y_tr, Y_va = (Y_tr - Y_mean) / Y_std, (Y_va - Y_mean) / Y_std
    print(f"training pairs {tuple(X_tr.shape)}, "
          f"validation pairs {tuple(X_va.shape)}")

    net = nn.Sequential(nn.Linear(5, HIDDEN), nn.Tanh(),
                        nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
                        nn.Linear(HIDDEN, 2))
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            loss = ((net(X_tr[b]) - Y_tr[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            val = ((net(X_va) - Y_va) ** 2).mean().item()
        print(f"epoch {epoch:3d} | val one-step MSE "
              f"(1.0 = predicting the mean): {val:.5f}")

    # The real test: feed the network its own output, starting at t = 0
    val_traj = D["val"]
    pred = rollout(net, val_traj[:, 0], t, meta, v_scale, Y_mean, Y_std)
    true = val_traj[:, ::STEP]
    print("\nrollout error: rms over the validation sleds, next to the "
          "spread (std) of the true values at that time")
    for s in ROLLOUT_SECONDS:
        n = round(s / (STEP * meta["dt"]))
        ey = (pred[:, n, 0] - true[:, n, 0]).pow(2).mean().sqrt().item()
        ev = (pred[:, n, 1] - true[:, n, 1]).pow(2).mean().sqrt().item()
        print(f"  t = {s:2d} s: y {ey:.3f} m (spread "
              f"{true[:, n, 0].std().item():.2f}), v {ev:.3f} m/s (spread "
              f"{true[:, n, 1].std().item():.2f})")

    OUT_DIR.mkdir(exist_ok=True)
    plot_phase_space(true.numpy(), pred.numpy(), meta,
                     OUT_DIR / "mlp_phase_space.png")
    torch.save({"state_dict": net.state_dict(), "Y_mean": Y_mean,
                "Y_std": Y_std, "v_scale": v_scale, "hidden": HIDDEN,
                "step": STEP, "meta": meta}, OUT_DIR / "mlp.pt")
    print(f"saved {OUT_DIR / 'mlp_phase_space.png'} and mlp.pt")


if __name__ == "__main__":
    main()