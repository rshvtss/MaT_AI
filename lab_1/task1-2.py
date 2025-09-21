import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


# Діапазон x
x = np.linspace(-15, 15, 100)

# 1. Трикутна і трапецієподібна функції приналежності
triMF = fuzz.trimf(x, [2, 4, 9])
trapMF = fuzz.trapmf(x, [1, 3, 5, 7])

# 2. Гаусова проста і двостороння (дві різні σ для лівої та правої частини)
gaussMF = fuzz.gaussmf(x, mean=5, sigma=1.5)

# Для двосторонньої гаусової з різними σ використаємо gauss2mf:
gauss2MF_1 = fuzz.gauss2mf(x, mean1=4, sigma1=1, mean2=6, sigma2=2)
gauss2MF_2 = fuzz.gauss2mf(x, mean1=3, sigma1=0.8, mean2=7, sigma2=1.5)
gauss2MF_3 = fuzz.gauss2mf(x, mean1=2, sigma1=1.2, mean2=8, sigma2=2.5)

# Побудова графіків
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(x, triMF)
axes[0, 0].set_title('Triangular MF')

axes[0, 1].plot(x, trapMF)
axes[0, 1].set_title('Trapezoid MF')

axes[1, 0].plot(x, gaussMF, label="Simple Gaussian")
axes[1, 0].set_title('Gaussian MF')
axes[1, 0].legend()

# усі три двосторонні гаусові на одній площині
axes[1, 1].plot(x, gauss2MF_1, label='Two-sided Gaussian #1')
axes[1, 1].plot(x, gauss2MF_2, label='Two-sided Gaussian #2')
axes[1, 1].plot(x, gauss2MF_3, label='Two-sided Gaussian #3')
axes[1, 1].set_title('Two-sided Gaussians')
axes[1, 1].legend()

for ax in axes.ravel():
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)

plt.tight_layout()
plt.show()
