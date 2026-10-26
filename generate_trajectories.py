import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import zarr
import matplotlib.pyplot as plt

#%%
# Data class contains the constant and variables assumed constant
@dataclass(frozen=True)
class SledParams:
    g: float = 9.81    # gravity (m/s^2)
    c: float = 0.5     # linear damping of the cross-slope velocity (1/s)
    a: float = 0.25    # mean slope
    b: float = 0.5     # mogul height (m)
    Lx: float = 10.0   # mogul wavelength down-slope (m)
    Ly: float = 4.0    # mogul wavelength cross-slope (m)
    u: float = 3.5     # constant down-slope speed (m/s)


def rhs(y, v, x, P):
    """Right hand side equation for the sled. Assuming u is constant.
    The state variables are y and v.
    """
    p, q = 2 * np.pi / P.Lx, 2 * np.pi / P.Ly
    sx, cx = np.sin(p * x), np.cos(p * x)
    sy, cy = np.sin(q * y), np.cos(q * y)

    H_x = -P.a + P.b * p * sx * cy
    H_y = P.b * q * cx * sy
    H_xx = P.b * p**2 * cx * cy
    H_yy = P.b * q**2 * cx * cy
    H_xy = -P.b * p * q * sx * sy

    F = (P.g + H_xx * P.u**2 + 2 * H_xy * P.u * v + H_yy * v**2) / (1 + H_x**2 + H_y**2)
    return v, -F * H_y - P.c * v


def rk4_step(y, v, x, dt, P):
    h = dt / 2
    k1y, k1v = rhs(y, v, x, P)
    k2y, k2v = rhs(y + h * k1y, v + h * k1v, x + P.u * h, P)
    k3y, k3v = rhs(y + h * k2y, v + h * k2v, x + P.u * h, P)
    k4y, k4v = rhs(y + dt * k3y, v + dt * k3v, x + P.u * dt, P)
    return (y + dt / 6 * (k1y + 2 * k2y + 2 * k3y + k4y),
            v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v))


def simulate(P, x0, y0, v0, dt, n_steps):
    """Vectorized loop, in which N sleds trajectories are integrated at the same dt.
    P is the dataclass, x0, y0, v0 are the initial condition vectores, dt and n_steps ar time variables."""
    states = np.empty((len(y0), n_steps + 1, 2))
    y, v = y0, v0
    states[:, 0, 0], states[:, 0, 1] = y, v
    for i in range(n_steps):
        y, v = rk4_step(y, v, x0 + P.u * i * dt, dt, P)
        states[:, i + 1, 0], states[:, i + 1, 1] = y, v
    return states


def save(path, P, states, t, x0, dt, seed):
    """Save the trajectories into a zarr. These are going to be used for training the NN."""
    root = zarr.open_group(path, mode="w")
    # chunk along sleds so reading a batch of whole trajectories touches few chunks
    root.create_array("states", data=states, chunks=(min(256, len(x0)), states.shape[1], 2))
    root.create_array("t", data=t)
    root.create_array("x0", data=x0)
    root.attrs.update({**asdict(P), "dt": dt, "seed": seed, "state_names": "y,v"})


# %% Setting up parameters
seed = 1234
rng = np.random.default_rng(seed)
N = 8000 # number of sleds
n_steps = 2500
dt = 0.01
x0 = np.zeros(N)
y0 = SledParams().Ly * (rng.random(N) - 0.5) # uniform in [-Ly/2, Ly/2) 
v0 = 10.0 * (rng.random(N) - 0.5) # uniform in [-5, 5)
t = np.arange(n_steps + 1) * dt
b_range = np.linspace(0.1, 0.8, 8) # Defining a range of height of bumps b to simulate
out = "data" 

for b in b_range:
    P = SledParams(b=b)
    tic = time.perf_counter()
    states = simulate(P, x0, y0, v0, dt, n_steps)

    path = os.path.join(out, f"sleds_b{b:g}_u{SledParams.u:g}.zarr")

    save(path, P, states, t, x0, dt, seed)

    print(f"b={b:g}: {N} sleds x {n_steps} steps in "
            f"{time.perf_counter() - tic:.1f} s -> {path}")
