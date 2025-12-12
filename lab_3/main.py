import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import skfuzzy as fuzz

centers = [
    [85, 55],  # технічні (технічні ≈85, гуманітарні ≈55)
    [65, 65],  # збалансовані (обоє ≈65)
    [50, 85]   # гуманітарні (технічні ≈50, гуманітарні ≈85)
]

n_samples = 400
cluster_std = [6.5, 7.0, 6.0]  # різний розкид у групах

# Згенеруємо дані
X, y_true = make_blobs(n_samples=n_samples, centers=centers,
                       cluster_std=cluster_std, random_state=42)

# Важливо: обмежимо значення в діапазоні [0, 100]
X[:, 0] = np.clip(X[:, 0], 0, 100)  # технічні оцінки
X[:, 1] = np.clip(X[:, 1], 0, 100)  # гуманітарні оцінки

# Візуалізація сирих даних
plt.figure(figsize=(6,5))
plt.scatter(X[:,0], X[:,1], alpha=0.6)
plt.xlabel('Середній бал з технічних дисциплін')
plt.ylabel('Середній бал з гуманітарних дисциплін')
plt.title('Синтетичні дані студентів (сирі)')
plt.grid(True)
plt.show()

# Нормалізація ознак (рекомендовано для FCM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # приводимо до нульового середнього та одиничної дисперсії

# Fuzzy C-Means (нечіткі c-середні)
# Параметри:
n_clusters = 3
m = 2.0          # нечіткість (звичайно 2.0)
error = 0.005
maxiter = 200

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X_scaled.T, c=n_clusters, m=m, error=error, maxiter=maxiter, init=None)

# cntr - центри в нормалізованому просторі; перетворимо їх назад в оригінальний масштаб
cntr_orig = scaler.inverse_transform(cntr)

# Отримання «жорстких» міток (за максимальною належністю)
labels = np.argmax(u, axis=0)

# Візуалізація результатів: кластери в оригінальному масштабі
colors = ['tab:blue', 'tab:orange', 'tab:green']
plt.figure(figsize=(7,6))
for j in range(n_clusters):
    plt.scatter(X[labels == j, 0], X[labels == j, 1],
                color=colors[j], label=f'Кластер {j+1}', alpha=0.6)
# центри
plt.scatter(cntr_orig[:,0], cntr_orig[:,1], marker='*', s=220, color='k', label='Центри')
plt.xlabel('Середній бал з технічних дисциплін')
plt.ylabel('Середній бал з гуманітарних дисциплін')
plt.title('Результат нечіткої кластеризації студентів (FCM)')
plt.legend()
plt.grid(True)
plt.show()

# Графік зміни цільової функції Jm
plt.figure(figsize=(6,4))
plt.plot(jm, marker='o')
plt.xlabel('Ітерація')
plt.ylabel('Значення цільової функції Jm')
plt.title('Зміна значення цільової функції під час навчання FCM')
plt.grid(True)
plt.show()

# Виведення центрів кластерів (в оригінальних одиницях) і FPC
print("Координати центрів кластерів (в оригінальному масштабі):")
for i, c in enumerate(cntr_orig):
    print(f"  Кластер {i+1}: технічні = {c[0]:.2f}, гуманітарні = {c[1]:.2f}")
print(f"\nFuzzy Partition Coefficient (FPC): {fpc:.4f}")
