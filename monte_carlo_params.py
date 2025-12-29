import numpy as np
import pandas as pd

# -----------------------------
# Settings
# -----------------------------
N = 10_000
rng = np.random.default_rng(42)

def sample_parabola(a, b, n, rng, alpha=2, beta=2):
    """
    Parabola-shaped distribution on [a,b] using Beta(2,2) by default.
    PDF on [0,1] is proportional to x(1-x), peaking at the midpoint.
    """
    x = rng.beta(alpha, beta, size=n)
    return a + (b - a) * x

# -----------------------------
# Simulate variables (10,000 "patients")
# -----------------------------

# k_a (1/hr): parabola-shaped in [1.0, 1.7]
ka = sample_parabola(1.0, 1.7, N, rng)

# k_g (1/min): parabola-shaped in [0.0249, 0.0307]
kg = sample_parabola(0.0249, 0.0307, N, rng)

# V_d: sex-specific parabola-shaped within ranges
sex = rng.choice(["M", "F"], size=N, p=[0.5, 0.5])
Vd = np.empty(N)

m = (sex == "M")
f = (sex == "F")
Vd[m] = sample_parabola(0.68, 0.72, m.sum(), rng)
Vd[f] = sample_parabola(0.55, 0.64, f.sum(), rng)

# k_e (1/hr): SINGLE numeric distribution (no categories)
# Choose one:
#  - parabola-shaped over full plausible range [0.08, 0.35]
ke = sample_parabola(0.08, 0.35, N, rng)

# Dose D (grams ethanol): parabola-shaped 1–8 drinks, rounded to int
num_drinks_cont = sample_parabola(1, 8, N, rng)
num_drinks = np.clip(np.rint(num_drinks_cont), 1, 8).astype(int)
D = 14 * num_drinks  # grams ethanol (US standard drink = 14 g)

# Drinking duration tau (minutes): parabola-shaped 15–25 min
tau = sample_parabola(15, 25, N, rng)

# -----------------------------
# Build table
# -----------------------------
df = pd.DataFrame({
    "k_a_hr": ka,
    "k_e_hr": ke,
    "k_g_min": kg,
    "Vd": Vd,
    "sex": sex,
    "num_drinks": num_drinks,
    "Dose_g": D,
    "tau_min": tau
})

# -----------------------------
# Output
# -----------------------------
print(df.head())

# Save to CSV (in the directory you run this from)
df.to_csv("monte_carlo_parameters_10000.csv", index=False)
print("Saved: monte_carlo_parameters_10000.csv")
