import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 500)

# --- 5. Поліноміальні функції ---
s_mf = fuzz.smf(x, 2, 6)          # S-функція
z_mf = fuzz.zmf(x, 2, 6)          # Z-функція
pi_mf = fuzz.pimf(x, 2, 4, 6, 8)  # PI-функція

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].plot(x, s_mf)
axes[0].set_title("S-функція")

axes[1].plot(x, z_mf)
axes[1].set_title("Z-функція")

axes[2].plot(x, pi_mf)
axes[2].set_title("PI-функція")

for ax in axes:
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)

plt.tight_layout()
plt.show()

# --- 6. Мінімаксна інтерпретація логічних операторів ---
# Створимо два довільних fuzzy-набори
A = fuzz.gaussmf(x, mean=4, sigma=1.5)
B = fuzz.gaussmf(x, mean=6, sigma=1.5)

# Кон’юнкція (AND = min)
fuzzy_and = np.fmin(A, B)

# Диз’юнкція (OR = max)
fuzzy_or = np.fmax(A, B)

# Нова фігура для мінімаксних операторів
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x, A, label="A")
plt.plot(x, B, label="B")
plt.title("Множини A і B")
plt.legend()
plt.ylim(-0.05, 1.05)
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x, fuzzy_and, "g")
plt.title("Кон’юнкція (min)")
plt.ylim(-0.05, 1.05)
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x, fuzzy_or, "r")
plt.title("Диз’юнкція (max)")
plt.ylim(-0.05, 1.05)
plt.grid(True)

plt.tight_layout()
plt.show()
