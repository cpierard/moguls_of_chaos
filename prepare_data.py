"""Split the sled trajectories into train / validation / test sets and fit the scales.

    from prepare_data import prepare_data
    D = prepare_data("data/sleds_b0.5_u3.5_seed1.zarr")
    D["train"]            # (N_train, T, 2) tensor, columns = (y, v), y wrapped to [-Ly/2, Ly/2)
    D["t"], D["v_scale"], D["dv_scale"], D["meta"]
"""
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import zarr

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
