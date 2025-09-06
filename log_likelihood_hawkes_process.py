import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# -----------------------------
# 1. Event data (replace with your own)
# -----------------------------
# Example: synthetic event times (in seconds, minutes, etc.)
event_times = np.array([0.1, 0.4, 0.6, 0.728, 1.2, 2.0, 2.5, 2.78, 3.16, 3.8, 4.1, 5.0, 6.0, 7.5, 8.0, 9.0])
T = 10.0  # Observation window end time

# -----------------------------
# 2. Log-likelihood function
# -----------------------------
def log_likelihood(beta, event_times, T):
    beta = float(beta)
    if beta <= 0:
        return -np.inf  # log-likelihood undefined for non-positive beta

    N = len(event_times)
    log_intensity_sum = 0.0

    # Sum of log intensities at event times
    for i in range(1, N):
        time_diffs = event_times[i] - event_times[:i]
        lambda_i = np.sum(np.exp(-beta * time_diffs))
        if lambda_i <= 0:
            return -np.inf
        log_intensity_sum += np.log(lambda_i)

    # Integral term
    integral_sum = np.sum((1 - np.exp(-beta * (T - event_times))) / beta)

    return log_intensity_sum - integral_sum

# -----------------------------
# 3. Negative log-likelihood for optimizer
# -----------------------------
def negative_log_likelihood(beta, event_times, T):
    return -log_likelihood(beta, event_times, T)

# -----------------------------
# 4. Optimize beta
# -----------------------------
result = minimize(
    negative_log_likelihood,
    x0=[1.0],                      # initial guess
    args=(event_times, T),
    bounds=[(1e-5, None)],         # beta must be > 0
    method='L-BFGS-B'
)

estimated_beta = result.x[0]
print(f"✅ Estimated β: {estimated_beta:.4f}")
print("Optimization successful:", result.success)
print("Log-likelihood at optimum:", -result.fun)

# -----------------------------
# 5. (Optional) Plot likelihood curve
# -----------------------------
betas = np.linspace(0.01, 5.0, 200)
likelihoods = [log_likelihood(b, event_times, T) for b in betas]

plt.figure(figsize=(8, 4))
plt.plot(betas, likelihoods, label="Log-Likelihood")
plt.axvline(x=estimated_beta, color='r', linestyle='--', label=f"Estimated β = {estimated_beta:.4f}")
plt.xlabel("β")
plt.ylabel("Log-Likelihood")
plt.title("Hawkes Process Log-Likelihood vs β")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
