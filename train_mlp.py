"""Plain neural network (no ODE solver): learn the one-step map  state(t) -> state(t + h).

Inputs : 5 features of (y, v, t)       Outputs: the change (dy, dv) over h = STEP * dt
Long trajectories come from feeding the network its own output again and again.

Edit the settings below and run:   python python/train_mlp.py
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from prepare_data import prepare_data

# ------------------------------------------------------------- settings ----
DATA = "data/sleds_b0.5_u3.5.zarr"
OUT_DIR = Path("checkpoints")

HIDDEN = 64
STEP = 5                         # h = STEP * dt = 0.05 s: how far the network jumps
BATCH = 4096
LR = 1e-3
EPOCHS = 20
SEED = 0
ROLLOUT_SECONDS = (1, 3, 5, 10, 25)   # where to report the rollout error
PLOT_STEPS = [0, 200, 500, 1000]      # data steps shown in the phase-space figure
# ------------------------------------------------------------------------


def wrap(d, Ly):
    """Map into [-Ly/2, Ly/2): y = 2 and y = -2 are the same point."""
    return (d + Ly / 2) % Ly - Ly / 2


def make_features(y, v, t, meta, v_scale):
    """(y, v, t) -> 5 numbers per sled. y, v: (...,) tensors; t: scalar or broadcastable."""
    p, q = 2 * math.pi / meta["Lx"], 2 * math.pi / meta["Ly"]
    x = (meta["u"] * t).expand_as(y)                 # x0 = 0 in the data
    return torch.stack([torch.sin(q * y), torch.cos(q * y), v / v_scale,
                        torch.sin(p * x), torch.cos(p * x)], dim=-1)


def make_pairs(data, t, meta, v_scale):
    """data: (N, T, 2) -> X (M, 5) and Y (M, 2) = change of (y, v) over STEP data steps."""
    idx = torch.arange(0, data.shape[1] - STEP, STEP)
    now, nxt = data[:, idx], data[:, idx + STEP]                       # (N, n, 2)
    X = make_features(now[..., 0], now[..., 1], t[idx], meta, v_scale).reshape(-1, 5)
    dy = wrap(nxt[..., 0] - now[..., 0], meta["Ly"])                   # y is wrapped in the data
    dv = nxt[..., 1] - now[..., 1]
    return X, torch.stack([dy, dv], dim=-1).reshape(-1, 2)


@torch.no_grad()
def rollout(net, z0, t, meta, v_scale, Y_mean, Y_std):
    """Apply the network to its own output. z0: (N, 2). Returns (N, n_steps + 1, 2)."""
    n_steps = (len(t) - 1) // STEP
    z, out = z0, [z0]
    for n in range(n_steps):
        inc = net(make_features(z[:, 0], z[:, 1], t[n * STEP], meta, v_scale)) * Y_std + Y_mean
        z = torch.stack([wrap(z[:, 0] + inc[:, 0], meta["Ly"]), z[:, 1] + inc[:, 1]], dim=-1)
        out.append(z)
    return torch.stack(out, dim=1)


def plot_phase_space(true, pred, meta, path):
    """Top row: true (y, v) of the validation sleds; bottom row: network rollout."""
    fig, axes = plt.subplots(2, len(PLOT_STEPS), sharex=True, sharey=True,
                             figsize=(3 * len(PLOT_STEPS), 6))
    for j, step in enumerate(PLOT_STEPS):
        n = step // STEP
        for row, data in enumerate((true, pred)):
            axes[row, j].scatter(data[:, n, 0], data[:, n, 1], s=2, color="k")
        axes[0, j].set_title(f"{round(step * meta['dt'] * meta['u'])} m")
        axes[1, j].set_xlabel("y (m)")
    axes[0, 0].set_ylabel("true: v (m/s)")
    axes[1, 0].set_ylabel("network: v (m/s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    torch.manual_seed(SEED)
    D = prepare_data(DATA)
    meta, t, v_scale = D["meta"], D["t"], D["v_scale"]

    X_tr, Y_tr = make_pairs(D["train"], t, meta, v_scale)
    X_va, Y_va = make_pairs(D["val"], t, meta, v_scale)
    # StandardScaler on the TARGETS, fitted on the training pairs only. Here (unlike the
    # ODE state) it is fine: dy and dv are ordinary regression targets.
    Y_mean, Y_std = Y_tr.mean(0), Y_tr.std(0)
    Y_tr, Y_va = (Y_tr - Y_mean) / Y_std, (Y_va - Y_mean) / Y_std
    print(f"training pairs {tuple(X_tr.shape)}, validation pairs {tuple(X_va.shape)}")

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
        print(f"epoch {epoch:3d} | val one-step MSE (1.0 = predicting the mean): {val:.5f}")

    # ---- the real test: feed the network its own output, starting from t = 0
    val_traj = D["val"]
    pred = rollout(net, val_traj[:, 0], t, meta, v_scale, Y_mean, Y_std)
    true = val_traj[:, ::STEP]
    print("\nrollout error (rms over validation sleds); the data's own spread is "
          f"y ~ {meta['Ly'] / math.sqrt(12):.2f} m, v ~ {v_scale:.2f} m/s")
    for s in ROLLOUT_SECONDS:
        n = round(s / (STEP * meta["dt"]))
        ey = wrap(pred[:, n, 0] - true[:, n, 0], meta["Ly"]).pow(2).mean().sqrt().item()
        ev = (pred[:, n, 1] - true[:, n, 1]).pow(2).mean().sqrt().item()
        print(f"  t = {s:2d} s: y {ey:.3f} m, v {ev:.3f} m/s")

    OUT_DIR.mkdir(exist_ok=True)
    plot_phase_space(true.numpy(), pred.numpy(), meta, OUT_DIR / "mlp_phase_space.png")
    torch.save({"state_dict": net.state_dict(), "Y_mean": Y_mean, "Y_std": Y_std,
                "v_scale": v_scale, "hidden": HIDDEN, "step": STEP, "meta": meta},
               OUT_DIR / "mlp.pt")
    print(f"saved {OUT_DIR / 'mlp_phase_space.png'} and mlp.pt")


if __name__ == "__main__":
    main()
