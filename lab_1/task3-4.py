import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Діапазон
x = np.linspace(0, 10, 500)

# 3. Узагальнений дзвін
gbell1 = fuzz.gbellmf(x, a=2, b=2, c=5)
gbell2 = fuzz.gbellmf(x, a=1.5, b=4, c=6)

# 4. Сігмоїдні функції
# Основна одностороння (відкрита зліва і справа)
sig_left = fuzz.sigmf(x, 5, -2)   # відкрита зліва
sig_right = fuzz.sigmf(x, 5, 2)   # відкрита справа

# Додаткова двостороння
sig_two = fuzz.dsigmf(x, b1=3, c1=2, b2=7, c2=2)

# Додаткова несиметрична
sig_asym = fuzz.psigmf(x, b1=3, c1=1, b2=8, c2=5)

# Побудова графіків
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# узагальнений дзвін
axes[0, 0].plot(x, gbell1, label="gbell (a=2, b=2, c=5)")
axes[0, 0].plot(x, gbell2, label="gbell (a=1.5, b=4, c=6)")
axes[0, 0].set_title("Generalized Bell MF")
axes[0, 0].legend()

# односторонні сігмоїди
axes[0, 1].plot(x, sig_left, label="Left-open")
axes[0, 1].plot(x, sig_right, label="Right-open")
axes[0, 1].set_title("One-sided Sigmoid MFs")
axes[0, 1].legend()

# двостороння сігмоїда
axes[1, 0].plot(x, sig_two, label="Two-sided")
axes[1, 0].set_title("Two-sided Sigmoid MF")
axes[1, 0].legend()

# несиметрична двостороння сігмоїда
axes[1, 1].plot(x, sig_asym, label="Asymmetric Two-sided")
axes[1, 1].set_title("Asymmetric Sigmoid MF")
axes[1, 1].legend()

# налаштування
for ax in axes.ravel():
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)

plt.tight_layout()
plt.show()
