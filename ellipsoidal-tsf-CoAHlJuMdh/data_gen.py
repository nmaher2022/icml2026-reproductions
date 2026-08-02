# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Synthetic chaotic-system generators for Ellipsoidal TSF (arXiv 2505.17370v6), Appendix A.4.1 /
Table 7. RK4 integration of Lorenz-63, Roessler, and Chua's circuit, with an optional mid-trajectory
"param" shock (constants change partway through) matching the paper's nonstationary-shock protocol.
Exact shock timing (train/test boundary) is not given in the paper text -- documented assumption,
see PAPER_BRIEFING.md / REPRO_LOG.md.
"""
import numpy as np


def lorenz63_deriv(state, sigma, rho, beta):
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.stack([dx, dy, dz], axis=-1)


def rossler_deriv(state, a, b, c):
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return np.stack([dx, dy, dz], axis=-1)


def chua_h(x, m0, m1):
    return m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1) - np.abs(x - 1))


def chua_deriv(state, alpha, beta, m0, m1):
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    dx = alpha * (y - x - chua_h(x, m0, m1))
    dy = x - y + z
    dz = -beta * y
    return np.stack([dx, dy, dz], axis=-1)


SYSTEMS = {
    "lorenz63": dict(deriv=lorenz63_deriv, dt=0.01,
                      base_params=dict(sigma=10.0, rho=28.0, beta=8.0 / 3.0),
                      param_shock=dict(sigma=10.1, rho=28.1, beta=8.1 / 3.0),
                      init=np.array([1.0, 0.98, 1.1])),
    "rossler": dict(deriv=rossler_deriv, dt=0.01,
                     base_params=dict(a=0.2, b=0.2, c=5.7),
                     param_shock=dict(a=0.25, b=0.25, c=5.75),
                     init=np.array([0.1, 0.1, 0.1])),
    "chua": dict(deriv=chua_deriv, dt=0.005,
                  base_params=dict(alpha=15.6, beta=28.0, m0=-8.0 / 7.0, m1=-5.0 / 7.0),
                  param_shock=dict(alpha=15.9, beta=28.5, m0=-8.1 / 7.0, m1=-5.2 / 7.0),
                  init=np.array([0.7, 0.0, 0.0])),
}


def rk4_step(deriv, state, dt, params):
    k1 = deriv(state, **params)
    k2 = deriv(state + 0.5 * dt * k1, **params)
    k3 = deriv(state + 0.5 * dt * k2, **params)
    k4 = deriv(state + dt * k3, **params)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(system_name, n_steps, shock="base", shock_frac=0.7, burn_in=500, seed=0):
    """Integrate `system_name` for n_steps (post burn-in), in float64, cast to float32 at the end
    (paper: float64 RK4, then converted to float32 -- pilot runs showed float64-throughout hurts
    all models). If shock == "param", base_params are used for steps < shock_frac*n_steps and
    param_shock params after (mid-trajectory nonstationary shock, applied at the train/test
    boundary -- exact timing not specified in the paper text, documented assumption)."""
    cfg = SYSTEMS[system_name]
    rng = np.random.default_rng(seed)
    state = cfg["init"].astype(np.float64) + rng.normal(scale=1e-3, size=3)
    for _ in range(burn_in):
        state = rk4_step(cfg["deriv"], state, cfg["dt"], cfg["base_params"])

    traj = np.zeros((n_steps, 3), dtype=np.float64)
    shock_step = int(n_steps * shock_frac)
    for t in range(n_steps):
        params = cfg["base_params"]
        if shock == "param" and t >= shock_step:
            params = cfg["param_shock"]
        traj[t] = state
        state = rk4_step(cfg["deriv"], state, cfg["dt"], params)
    return traj.astype(np.float32), shock_step


def make_windows(series_1d, context_len, horizon_len, stride=1):
    """series_1d: (T,) float32 array (single channel, per paper's channel-independent design).
    Returns X: (N, context_len), Y: (N, horizon_len)."""
    T = len(series_1d)
    n = (T - context_len - horizon_len) // stride + 1
    X = np.zeros((n, context_len), dtype=np.float32)
    Y = np.zeros((n, horizon_len), dtype=np.float32)
    for i in range(n):
        s = i * stride
        X[i] = series_1d[s:s + context_len]
        Y[i] = series_1d[s + context_len:s + context_len + horizon_len]
    return X, Y


def train_test_split_windows(X, Y, test_frac=0.2, val_frac=0.1):
    """Time-ordered split (no shuffling) -- standard LTSF convention, avoids leakage across the
    shock boundary."""
    n = len(X)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_test - n_val
    return (X[:n_train], Y[:n_train],
            X[n_train:n_train + n_val], Y[n_train:n_train + n_val],
            X[n_train + n_val:], Y[n_train + n_val:])


def get_dataset(system_name, n_steps, context_len, horizon_len, channel=0, shock="base",
                 shock_frac=0.7, stride=1, seed=0):
    traj, shock_step = simulate(system_name, n_steps, shock=shock, shock_frac=shock_frac, seed=seed)
    series = traj[:, channel]
    X, Y = make_windows(series, context_len, horizon_len, stride=stride)
    Xtr, Ytr, Xval, Yval, Xte, Yte = train_test_split_windows(X, Y)
    train_std = float(Ytr.std())
    return dict(Xtr=Xtr, Ytr=Ytr, Xval=Xval, Yval=Yval, Xte=Xte, Yte=Yte,
                train_std=train_std, traj=traj, shock_step=shock_step)


if __name__ == "__main__":
    for name in SYSTEMS:
        for shock in ("base", "param"):
            d = get_dataset(name, n_steps=6000, context_len=96, horizon_len=96, shock=shock)
            print(name, shock, "traj std", d["traj"].std(axis=0), "Xtr", d["Xtr"].shape,
                  "Xte", d["Xte"].shape, "train_std", d["train_std"],
                  "nan?", np.isnan(d["Xtr"]).any())
