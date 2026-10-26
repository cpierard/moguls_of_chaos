"""Functions designed to read and split the sled trajectories into train / validation / test sets and fit the scales.
    D = prepare_data("data/sleds_b0.5_u3.5_seed1.zarr")
    D["train"] # (N_train, T, 2) tensor, columns = (y, v), y wrapped to [-Ly/2, Ly/2)
    D["t"], D["v_scale"], D["dv_scale"], D["meta"]
"""
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import zarr

def load_trajectories(path, stride=1, wrap_y=False):
    """Load trajectories from a Zarr store.

    Args:
        path: Path to the Zarr store.
        stride: Step between sampled time points.
        wrap_y: Whether to wrap y values into [-Ly/2, Ly/2).

    Returns:
        A dictionary containing ``state`` (shape (N, T, 2), columns y and v),
        ``x`` (shape (N, T), unwrapped down-slope position x0 + u*t), ``t``
        (the time grid), and ``meta`` (the store attributes).

    The stored y values are unwrapped and continuous in time. Setting
    ``wrap_y`` introduces jumps. For learning, prefer feeding
    (sin(q*y), cos(q*y)) to the network, where q = 2*pi/Ly.
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
    """Split trajectories by sled and compute velocity scales from training data.

    Args:
        path: Path to the Zarr store.
        val_size: Fraction of the full dataset reserved for validation.
        test_size: Fraction of the full dataset reserved for testing.
        random_state: Seed for reproducible data splits.
        dtype: PyTorch dtype for the returned tensors.

    Returns:
        A dictionary containing the split tensors, time grid, velocity scales,
        and metadata.
    """
    d = load_trajectories(path, wrap_y=False) # we need unwrapped y to train the NN
    state = d["state"] # (N, T, 2) -> axis 0 = sleds, axis 1 = time, axis 2 = (y, v)
    dt = d["meta"]["dt"]

    # Split BY SLED (axis 0): each sled is a whole trajectory. Splitting individual
    # time points would put neighbouring points of the same trajectory in train and test.
    train_val, test = train_test_split(state, test_size=test_size, random_state=random_state)
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size),
                                  random_state=random_state)

    # We need to scale up the tensors to avoid biases. To do this, we compute the std of the 
    # training data and use it to scale the inputs and outputs.
    v_scale = float(train[..., 1].std())
    dv_scale = float((np.diff(train[..., 1], axis=1) / dt).std())   # std of dv/dt

    as_t = lambda a: torch.as_tensor(a, dtype=dtype)
    return {
        "train": as_t(train), "val": as_t(val), "test": as_t(test),
        "t": as_t(d["t"]), "v_scale": v_scale, "dv_scale": dv_scale, "meta": d["meta"],
    }
