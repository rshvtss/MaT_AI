import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Всесвіт
x = np.linspace(0, 10, 200)

# Дві нечіткі множини (наприклад, "Низький" і "Високий")
A = fuzz.trimf(x, [0, 3, 6])   # Трикутна
B = fuzz.gaussmf(x, mean=7, sigma=1.5)  # Гаусова

# 7. Кон’юнкція (AND) та диз’юнкція (OR)
A_and_B = np.fmin(A, B)  # мінімум
A_or_B = np.fmax(A, B)   # максимум

# 8. Доповнення множини A
A_not = 1 - A

# Візуалізація
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Початкові множини
axes[0, 0].plot(x, A, 'b', linewidth=2, label='A (низький)')
axes[0, 0].plot(x, B, 'r', linewidth=2, label='B (високий)')
axes[0, 0].set_title("Початкові множини")
axes[0, 0].legend()
axes[0, 0].grid(True)

# Кон’юнкція
axes[0, 1].plot(x, A_and_B, 'm', linewidth=2)
axes[0, 1].set_title("Кон’юнкція A ∧ B (min)")
axes[0, 1].grid(True)

# Диз’юнкція
axes[1, 0].plot(x, A_or_B, 'g', linewidth=2)
axes[1, 0].set_title("Диз’юнкція A ∨ B (max)")
axes[1, 0].grid(True)

# Доповнення
axes[1, 1].plot(x, A, 'b--', linewidth=2, label='A')
axes[1, 1].plot(x, A_not, 'k', linewidth=2, label='NOT A')
axes[1, 1].set_title("Доповнення ¬A")
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()