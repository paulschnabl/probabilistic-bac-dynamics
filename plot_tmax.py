import matplotlib.pyplot as plt

# t_max midpoints (minutes)
x = [
    126.3, 131.2, 136.1, 140.9, 145.8, 150.7, 155.6, 160.4, 165.3,
    170.2, 175.1, 179.9, 184.8, 189.7, 194.6, 199.4, 204.3, 209.2,
    214.0, 218.9, 223.8, 228.7, 233.5, 238.4, 243.3
]

# probabilities
y = [
    0.0015, 0.0065, 0.0198, 0.0375, 0.0675, 0.0906, 0.1035, 0.1190,
    0.1078, 0.1073, 0.0933, 0.0738, 0.0561, 0.0416, 0.0305, 0.0208,
    0.0108, 0.0059, 0.0035, 0.0013, 0.0011, 0.0, 0.0002, 0.0, 0.0001
]

plt.plot(x, y, marker='o')
plt.xlabel("t_max (minutes)")
plt.ylabel("Probability")
plt.title("Distribution of Time to Maximum BAC (t_max)")
plt.grid(True)

plt.show()

