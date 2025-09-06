import numpy as np
from scipy.optimize import minimize
from numpy.linalg import eigvals
import pandas as pd
from datetime import datetime
import time

# ---- negative log-likelihood helper ----
def _neg_loglik_theta(x, t, c, T):
    # Map unconstrained -> positive
    mu = np.exp(x[0:2])                       # [mu_W, mu_N]
    alpha = np.exp(x[2:6]).reshape(2, 2)      # 2x2 alpha
    beta  = np.exp(x[6:10]).reshape(2, 2)     # 2x2 beta

    n = len(t)
    if n == 0:
        return np.inf

    # h[i,k,a] recursion
    h = np.zeros((n, 2, 2))
    for i in range(1, n):
        dt = t[i] - t[i-1]
        a_prev = c[i-1]
        decay = np.exp(-beta * dt)
        h[i] = decay * (h[i-1] + np.eye(2)[:, a_prev])

    lam = np.zeros((n, 2))
    for i in range(n):
        lam[i] = mu + (alpha * h[i]).sum(axis=1)

    if np.any(lam <= 0) or np.any(~np.isfinite(lam)):
        return np.inf

    # point log-likelihood
    ll_point = np.sum(np.log(lam[np.arange(n), c]))

    # integral term
    ll_int = 0.0
    for k in range(2):
        term = mu[k] * T
        for a in range(2):
            mask = (c == a)
            if mask.any():
                term += (alpha[k,a] / beta[k,a]) * np.sum(1 - np.exp(-beta[k,a] * (T - t[mask])))
        ll_int += term

    # stability penalty
    N = alpha / beta
    rho = max(eigvals(N).real)
    pen = 0.0
    if rho >= 0.99:
        pen = 1e6 * (rho - 0.99)**2

    return -(ll_point - ll_int) + pen

# ---- main fitting function ----
def fit_hawkes_2type_exp(t, c, T=None, x0=None):
    t = np.asarray(t, dtype=float)
    c = np.asarray(c, dtype=int)
    order = np.argsort(t)
    t, c = t[order], c[order]
    if T is None:
        T = float(t[-1])
    if x0 is None:
        x0 = np.log(np.array([
            0.05, 0.05,      # mu_W, mu_N
            0.1, 0.05, 0.05, 0.1,  # alpha[W,W], alpha[W,N], alpha[N,W], alpha[N,N]
            1.0, 1.0, 1.0, 1.0     # betas
        ]))
    obj = lambda v: _neg_loglik_theta(v, t, c, T)
    res = minimize(obj, x0, method="L-BFGS-B")

    # recover params
    mu = np.exp(res.x[:2])
    alpha = np.exp(res.x[2:6]).reshape(2,2)
    beta  = np.exp(res.x[6:10]).reshape(2,2)
    N = alpha / beta
    rho = max(eigvals(N).real)

    return {
        "mu": mu, "alpha": alpha, "beta": beta,
        "branching_matrix": N, "spectral_radius": float(rho),
        "success": bool(res.success), "message": res.message
    }

def get_date(date_str: str) -> float:
    # dates are stored in yyyyMMdd hh:mm:ss
    year = int(date_str[0:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    hour = int(date_str[9:11])
    minute = int(date_str[12:14])
    second = float(date_str[15:])
    microsecond = int((second % 1) * 1_000_000)
    time = datetime(year, month, day, hour, minute, int(second), microsecond)
    return time.timestamp()

# ---- Example data ----
if __name__ == "__main__":
    # Load your CSV
    df = pd.read_csv("data/GBP-USD_june25.csv", header=None, names=["instrument", "timestamp", "bid", "ask"],     dtype={"instrument": str, "timestamp": str, "bid": float, "ask": float})
    df["timestamp"] = df["timestamp"].apply(get_date)
    df = df.head(1000000)

    start_time = time.perf_counter()
    # Ensure sorted by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Convert to relative time (seconds from first event)
    t0 = df['timestamp'].iloc[0]
    df['t_sec'] = df['timestamp'] - t0

    # Compute bid-ask spread
    df['spread'] = df['ask'] - df['bid']

    # Compute spread changes
    df['diff'] = df['spread'].diff()

    # Threshold to filter noise (set eps > 0 to ignore micro-ticks)
    eps = 0.0
    events = df.loc[df['diff'].abs() > eps, ['t_sec','diff']].copy()

    # Map to event type: 0 = widen, 1 = narrow
    events['c'] = (events['diff'] < 0).astype(int)

    # Prepare arrays for Hawkes fit
    t = events['t_sec'].to_numpy(dtype=float)
    c = events['c'].to_numpy(dtype=int)
    T = float(df['t_sec'].iloc[-1])   # end of observation window
    
    fit = fit_hawkes_2type_exp(t, c, T)
    print("Fitting completed in {:.2f} seconds".format(time.perf_counter() - start_time))
    print(fit)
