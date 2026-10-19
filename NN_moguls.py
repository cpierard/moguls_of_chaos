#%%
import numpy as np
import zarr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


def load_trajectories(path, stride=1, wrap_y=False):
    """Return a dict with:
        state : (N, T, 2) float64, last axis = (y, v)
        x     : (N, T)    down-slope position x0 + u*t (unwrapped)
        t     : (T,)      time grid
        meta  : dict of attributes (b, u, g, c, a, Lx, Ly, dt, seed, ...)

    y is stored unwrapped, so it is continuous in time. With wrap_y=True it is
    mapped into [-Ly/2, Ly/2), which introduces jumps. For learning, prefer
    feeding (sin(q*y), cos(q*y)) with q = 2*pi/Ly to the network instead.
    """
    root = zarr.open_group(path, mode="r")
    state = root["states"][:, ::stride, :]
    t = root["t"][::stride]
    x0 = root["x0"][...]
    meta = root.attrs.asdict()

    if wrap_y:
        half = meta["Ly"] / 2
        state[..., 0] = (state[..., 0] + half) % meta["Ly"] - half

    x = x0[:, None] + meta["u"] * t[None, :]
    return {"state": state, "x": x, "t": t, "meta": meta}


def plot_trajectory_snapshots(d):
    """Plot cross-slope position and velocity at four points in time."""
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(4, 4))
    axes = axes.reshape(4)
    steps = d['state'].shape[1]
    time_steps = [0, steps // 8, steps // 4, steps - 1]
    axes[1].set_title(f"b = {d['meta']['b']:.2f}, u = {d['meta']['u']:.2f} m/s", fontsize=10)

    for i, j in enumerate(time_steps):
        axes[i].scatter(d['state'][:, j, 0], d['state'][:, j, 1], s=0.1, color='k')
        dist = round(j*(d['meta']['u']*d['meta']['dt']))    
        axes[i].text(1.5,-4.9, '{} m'.format(dist) , ha='center', fontsize=8)

    plt.tight_layout()
    # plt.suptitle(f"b = {d['meta']['b']:.2f}, u = {d['meta']['u']:.2f} m/s")
    fig.text(0.5, -0.01, 'y - Cross slope position (m)', ha='center')
    fig.text(-0.01, 0.5, 'v - Cross slope velocity (m/s)', va='center', rotation='vertical')
    return fig, axes


def to_torch_dataset(d, dtype=None):
    """TensorDataset whose items are (state (T,2), x (T,)). Shared grid: d['t']."""
    import torch
    from torch.utils.data import TensorDataset

    dtype = dtype or torch.float32
    return TensorDataset(
        torch.as_tensor(d["state"], dtype=dtype),
        torch.as_tensor(d["x"], dtype=dtype),
    )


#%% Load the data and plot some snapshots (y wrapped)
d = load_trajectories("data/sleds_b0.8_u3.5.zarr", wrap_y=True)

fig, axes = plot_trajectory_snapshots(d)


 
def prepare_data(path, val_size=0.1, test_size=0.1, random_state=42, dtype=torch.float32):
    d = load_trajectories(path, wrap_y=True)
    state = d["state"]                    # (N, T, 2) -> axis 0 = sleds
    dt = d["meta"]["dt"]
 
    # Split BY SLED (axis 0): each sled is a whole trajectory. Splitting individual
    # time points would put neighbouring points of the same trajectory in train and test.
    train_val, test = train_test_split(state, test_size=test_size, random_state=random_state)
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size),
                                  random_state=random_state)
 
    # "Scaler.fit" on the training sleds only. These are plain constants that the
    # model divides by internally; the data itself stays in physical units.
    v_scale = float(train[..., 1].std())
    dv_scale = float((np.diff(train[..., 1], axis=1) / dt).std())   # std of dv/dt
 
    as_t = lambda a: torch.as_tensor(a, dtype=dtype)
    return {
        "train": as_t(train), "val": as_t(val), "test": as_t(test),
        "t": as_t(d["t"]), "v_scale": v_scale, "dv_scale": dv_scale, "meta": d["meta"],
    }
 