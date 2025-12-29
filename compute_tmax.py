import numpy as np
import pandas as pd

# ---------------------------------------------
# Monte Carlo tmax calculator
#
# Model:
#   dS/dt = u(t) - kg*S
#   dI/dt = kg*S - ka*I
#   dC/dt = (ka*I)/Vd - ke*C
#
# Drinking input (fixed total dose, no dose column used):
#   u(t) = 14 / tau_min   for 0 <= t <= tau_min
#        = 0              otherwise
#
# Your CSV headers (as in your screenshot):
#   k_a_hr, k_e_hr, k_g_min, Vd, tau_min
#
# Output columns appended:
#   tmax_min, tmax_hr, tmax_status
# ---------------------------------------------


def compute_tmax_from_df(
    df: pd.DataFrame,
    t_end_min: int = 480,   # 8 hours
    dt_min: float = 1.0,    # 1-minute step
) -> pd.DataFrame:
    # Read parameters (already renamed to ka_hr, ke_hr, kg_min)
    ka_hr  = df["ka_hr"].to_numpy(float)     # 1/hr
    ke_hr  = df["ke_hr"].to_numpy(float)     # 1/hr
    kg     = df["kg_min"].to_numpy(float)    # 1/min
    Vd     = df["Vd"].to_numpy(float)
    tau    = df["tau_min"].to_numpy(float)   # min

    n = len(df)
    if n == 0:
        raise ValueError("Input dataframe is empty.")

    # Convert to per-minute
    ka = ka_hr / 60.0
    ke = ke_hr / 60.0

    # Sanity checks
    if np.any(tau <= 0):
        bad = np.where(tau <= 0)[0][:10]
        raise ValueError(f"tau_min must be > 0. Bad indices (first 10): {bad.tolist()}")
    if np.any(Vd <= 0):
        bad = np.where(Vd <= 0)[0][:10]
        raise ValueError(f"Vd must be > 0. Bad indices (first 10): {bad.tolist()}")
    if np.any(ka <= 0) or np.any(ke <= 0) or np.any(kg <= 0):
        bad = np.where((ka <= 0) | (ke <= 0) | (kg <= 0))[0][:10]
        raise ValueError(f"All rate constants must be > 0. Bad indices (first 10): {bad.tolist()}")

    # Time grid
    steps = int(np.floor(t_end_min / dt_min))
    times = np.arange(steps + 1, dtype=float) * dt_min

    # States (vectorized across patients)
    S = np.zeros(n, dtype=float)
    I = np.zeros(n, dtype=float)
    C = np.zeros(n, dtype=float)

    # Output containers
    tmax = np.full(n, np.nan, dtype=float)
    status = np.full(n, "no_peak_by_end", dtype=object)
    found = np.zeros(n, dtype=bool)

    # Fixed drinking rate: 14 g total ethanol over tau minutes
    u_rate = 14.0 / tau  # g/min during drinking

    # dC/dt from model (RHS, not finite differences)
    def dCdt(Iv: np.ndarray, Cv: np.ndarray) -> np.ndarray:
        return (ka * Iv) / Vd - ke * Cv

    dC_prev = dCdt(I, C)

    # Forward Euler integration + first zero-crossing of dC/dt
    for j in range(steps):
        t = times[j]

        # Input u(t): active during drinking interval
        u = np.where(t <= tau, u_rate, 0.0)

        # Euler updates
        S += dt_min * (u - kg * S)
        I += dt_min * (kg * S - ka * I)
        C += dt_min * dCdt(I, C)

        # Derivative at next timepoint
        dC_next = dCdt(I, C)

        # Peak occurs when dC/dt crosses from >=0 to <=0
        cross = (~found) & (dC_prev >= 0) & (dC_next <= 0)
        if np.any(cross):
            denom = (dC_prev[cross] - dC_next[cross])
            frac = np.where(np.abs(denom) > 1e-12, dC_prev[cross] / denom, 0.5)
            frac = np.clip(frac, 0.0, 1.0)
            tmax[cross] = t + frac * dt_min
            status[cross] = "ok"
            found[cross] = True

        dC_prev = dC_next
        if found.all():
            break

    # Keep original columns + append results
    out = df.copy()
    out["tmax_min"] = tmax
    out["tmax_hr"] = tmax / 60.0
    out["tmax_status"] = status

    # IMPORTANT: remove Dose_g from the OUTPUT if it exists in your input CSV
    out = out.drop(columns=["Dose_g"], errors="ignore")

    return out


if __name__ == "__main__":
    filepath = "/Users/paulschnabl/Desktop/monte_carlo_parameters_10000.csv"
    df = pd.read_csv(filepath)

    # Rename your actual CSV headers -> internal names used by the model
    df = df.rename(columns={
        "k_a_hr": "ka_hr",
        "k_e_hr": "ke_hr",
        "k_g_min": "kg_min",
    })

    required = ["ka_hr", "ke_hr", "kg_min", "Vd", "tau_min"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Columns found: {list(df.columns)}"
        )

    out = compute_tmax_from_df(df, t_end_min=480, dt_min=1.0)

    out_path = filepath.replace(".csv", "_with_tmax.csv")
    out.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print(out["tmax_status"].value_counts(dropna=False))
    ok = out["tmax_status"] == "ok"
    print("tmax_min summary (ok only):")
    print(out.loc[ok, "tmax_min"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]))
