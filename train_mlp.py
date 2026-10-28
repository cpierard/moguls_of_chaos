#%%
"""Plain neural network (no ODE solver) for the sled on moguls.
"""
import math
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from prepare_data import prepare_data
import matplotlib.animation as animation
import numpy as np


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
                      v_scale).reshape(-1, 5) # (M, 5)
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
    n_steps = (len(t) - 1) // STEP # number of steps to predict
    z, out = z0, [z0] # z is the current state, out will hold all predicted states
    for n in range(n_steps):
        increase = net(make_features(z[:, 0], z[:, 1], t[n * STEP], meta,
                                v_scale)) * Y_std + Y_mean
        z = z + increase  # columns of increase are (dy, dv)
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
            if row == 0:
                color = "C0"
            else:
                color = "C1"    

            y = (data[:, n, 0] + half) % meta["Ly"] - half  # display only
            axes[row, j].scatter(y, data[:, n, 1], s=2, color=color)
        axes[0, j].set_title(f"{round(step * meta['dt'] * meta['u'])} m")
        axes[1, j].set_xlabel("y (m)")
    axes[0, 0].set_ylabel("true: v (m/s)")
    axes[1, 0].set_ylabel("network: v (m/s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


#%% ------------------------------------------------------ settings ---

DATA = Path("data/sleds_b0.5_u3.5.zarr")
OUT_DIR = Path("checkpoints")

HIDDEN = 64                  # neurons per hidden layer
STEP = 1                     # h = STEP * dt = 0.05 s: how far it jumps
BATCH = 4096
LR = 1e-3
EPOCHS = 20
SEED = 0
ROLLOUT_SECONDS = (1, 3, 5, 10, 25)  # report rollout error at these t
PLOT_STEPS = [0, 200, 500, 1000]     # data steps in the plot

#%% ------------------------ TRAIN -------------------------------
torch.manual_seed(SEED)
D = prepare_data(DATA)
meta, t, v_scale = D["meta"], D["t"], D["v_scale"]

X_tr, Y_tr = make_pairs(D["train"], t, meta, v_scale) # X is input features, Y is target (dy, dv)
X_va, Y_va = make_pairs(D["val"], t, meta, v_scale) # validation pairs
# StandardScaler on the TARGETS, fitted on the training pairs only.
# Here (unlike the ODE state) it is fine: dy and dv are ordinary
# regression targets.
Y_mean, Y_std = Y_tr.mean(0), Y_tr.std(0)
Y_tr, Y_va = (Y_tr - Y_mean) / Y_std, (Y_va - Y_mean) / Y_std
print(f"training pairs {tuple(X_tr.shape)}, "
        f"validation pairs {tuple(X_va.shape)}")

#building the network and optimizer
net = nn.Sequential(nn.Linear(5, HIDDEN), nn.Tanh(),
                    nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
                    nn.Linear(HIDDEN, 2))
opt = torch.optim.Adam(net.parameters(), lr=LR)

#
print("One-step validation MSE is in standardized units: "
        "1.0 = no better than predicting the mean, 0.0 = perfect.")


#%%
for epoch in range(1, EPOCHS + 1):
    # training the network in batches
    for i in range(0, len(X_tr), BATCH):
        b = slice(i, i + BATCH) # selecting a batch of indices
        loss = ((net(X_tr[b]) - Y_tr[b]) ** 2).mean() # mean squared error loss
        opt.zero_grad()
        loss.backward() # computing gradients
        opt.step() # updating weights
    with torch.no_grad():
        val = ((net(X_va) - Y_va) ** 2).mean().item()
    print(f"epoch {epoch:3d}/{EPOCHS} | val one-step MSE = {val:.5f}")

#%% ----------------------- Validation -------------------------
val_traj = D["val"]
pred = rollout(net, val_traj[:, 0], t, meta, v_scale, Y_mean, Y_std)
true = val_traj[:, ::STEP]
print("\nRollout from t = 0 on the validation sleds, where the "
        "network feeds on its own output.\n"
        "error = rms over sleds of (prediction - truth); "
        "spread = std of the true values at that time.\n"
        "An error close to the spread means no skill is left.")

for s in ROLLOUT_SECONDS:
    n = round(s / (STEP * meta["dt"]))
    ey = (pred[:, n, 0] - true[:, n, 0]).pow(2).mean().sqrt().item()
    ev = (pred[:, n, 1] - true[:, n, 1]).pow(2).mean().sqrt().item()
    sy = true[:, n, 0].std().item()
    sv = true[:, n, 1].std().item()
    print(f"  t = {s:2d} s | y: error = {ey:6.3f} m "
            f"(spread = {sy:5.2f} m) | v: error = {ev:5.3f} m/s "
            f"(spread = {sv:4.2f} m/s)")


#%% Save the network and a figure of the phase space
OUT_DIR.mkdir(exist_ok=True)
# plot_phase_space(true.numpy(), pred.numpy(), meta,
#                     OUT_DIR / "mlp_phase_space.png")
torch.save({"state_dict": net.state_dict(), "Y_mean": Y_mean,
            "Y_std": Y_std, "v_scale": v_scale, "hidden": HIDDEN,
            "step": STEP, "meta": meta}, OUT_DIR / f"mlp_b{meta['b']:.2f}_u{meta['u']:.2f}.pt")
print(f"saved {OUT_DIR / f"mlp_b{meta['b']:.2f}_u{meta['u']:.2f}.pt"}")

#%% ----------------------- Test -------------------------

test_traj = D["test"]
pred = rollout(net, test_traj[:, 0], t, meta, v_scale, Y_mean, Y_std)
# make x. It's the same for all sleds.
x_pred = meta["u"] * t[::STEP]
x = meta["u"] * t

sleds = torch.randperm(test_traj.shape[0])[:7]

#%%
fig, axes = plt.subplots(1, 7, sharex=True, sharey=True, figsize=(8.8, 3))
for ax, sled in zip(axes, sleds):
    sled = sled.item()
    ax.plot(test_traj[sled, :, 0], x, label="True")
    ax.plot(pred[sled, :, 0], x_pred, label="Predicted")

    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    ax.grid()
    ax.set_xlabel('')
    ax.set_title(f'sled {sled}')

fig.supxlabel('y - Crossslope position (m)', )
fig.supylabel('x - Downslope position (m)', )
axes[0].legend(loc='upper center', ncol=2, bbox_to_anchor=(0.9, -0.13))
axes[-1].text(-70, 115, f'b = {meta["b"]:.2f} m, u = {meta["u"]:.2f} m/s', va='center', fontsize=10)
fig.savefig(f"fig/traj_test_b{meta['b']:.2f}_u{meta['u']:.2f}.png", dpi=150)

#%% plot the average cross-slope position and velocity over time for the test sleds
rmse_pred = ((pred - test_traj[:, ::STEP]) ** 2).mean(0).sqrt()

plt.plot(t[::STEP], rmse_pred[:, 0], label='y - Crossslope position (m)')
# plt.plot(t, rmse_true[:, 0], label='y - Crossslope position (m) spread')

plt.plot(t[::STEP], rmse_pred[:, 1], label='v - Crossslope velocity (m/s)')
plt.xlabel('Time (s)')
plt.ylabel('RMS Error')
plt.title(f'b = {meta["b"]:.2f} m, u = {meta["u"]:.2f} m/s')
plt.legend()
plt.savefig(f"fig/mlp_rms_error_b{meta['b']:.2f}_u{meta['u']:.2f}.png", dpi=150)

#%% poincare maps
plot_phase_space(test_traj[:, ::STEP].numpy(), pred.numpy(), meta,
                    f"fig/mlp_phase_space_b{meta['b']:.2f}_u{meta['u']:.2f}.png")

#%% make a gif animation of the test sleds and pred of the phase space

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(6, 3.5))
half = meta["Ly"] / 2
axes[0].set_xlabel("y (m)")
axes[1].set_xlabel("y (m)")
axes[0].set_ylabel("v (m/s)")
axes[0].set_title(f"True at x={x[0]:.1f} m")
axes[1].set_title(f"Predicted at x={x_pred[0]:.1f} m")
axes[0].set_xlim(-half, half)
axes[1].set_xlim(-half, half)
axes[0].set_ylim(test_traj[:, :, 1].min(), test_traj[:, :, 1].max())
axes[1].set_ylim(pred[:, :, 1].min(), pred[:, :, 1].max())
axes[0].text(.5, -6, f"b = {meta['b']:.2f} m", fontsize=10, va='center')
axes[1].text(.5, -6, f"b = {meta['b']:.2f} m", fontsize=10, va='center')
fig.tight_layout()

def update(frame):  
    axes[0].clear()
    axes[1].clear()
    y_true = (test_traj[:, frame, 0] + half) % meta["Ly"] - half
    y_pred = (pred[:, frame, 0] + half) % meta["Ly"] - half
    axes[0].scatter(y_true, test_traj[:, frame, 1], s=2, color="C0")
    axes[1].scatter(y_pred, pred[:, frame, 1], s=2, color="C1")
    axes[0].set_title(f"True at x={x[frame]:.1f} m")
    axes[1].set_title(f"Predicted at x={x_pred[frame]:.1f} m")
    axes[0].set_xlim(-half, half)
    axes[1].set_xlim(-half, half)
    axes[0].set_ylim(test_traj[:, :, 1].min(), test_traj[:, :, 1].max())
    axes[1].set_ylim(pred[:, :, 1].min(), pred[:, :, 1].max())
    axes[0].text(.5, -6, f"b = {meta['b']:.2f} m", fontsize=10, va='center')
    axes[1].text(.5, -6, f"b = {meta['b']:.2f} m", fontsize=10, va='center')    
    
ani = animation.FuncAnimation(fig, update, frames=range(0, 500, STEP*2), repeat=False)
ani.save(f"animations/mlp_phase_space_b{meta['b']:.2f}_u{meta['u']:.2f}.gif", writer="imagemagick", fps=60, dpi=100)
plt.close(fig)


# %%
