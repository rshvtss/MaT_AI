import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 2. Формування вибірки для навчання
# ==========================================
print("--- Генерація даних ---")

# Функція, що імітує реальну поведінку сервера (Ground Truth)
# z = f(x, y) де x - RPS, y - розмір транзакції
def server_load_function(x, y):
    # Нелінійна функція: база + вплив RPS + вплив розміру + синергія (х*y)
    base_load = 10
    load = base_load + (0.5 * x) + (0.2 * y) + (0.001 * x * y**1.5)
    # Додаємо випадковий шум
    noise = np.random.normal(0, 2, size=x.shape)
    return np.clip(load + noise, 0, 100) # Обмеження 0-100%

# Генерація 500 зразків
N_SAMPLES = 500
X_rps = np.random.uniform(0, 100, N_SAMPLES)  # 0-100 запитів
X_size = np.random.uniform(1, 50, N_SAMPLES)  # 1-50 MB

y_data = server_load_function(X_rps, X_size)
X_data = np.column_stack((X_rps, X_size))

# Розбиття на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# ==========================================
# 3 & 4. Структура та Навчання мережі
# ==========================================
# Використовуємо MLPRegressor для емуляції гібридного навчання (настройка параметрів)
# Це відповідає налаштуванню параметрів консеквентів у ANFIS
print("--- Навчання мережі ---")

# Параметри навчання:
# hidden_layer_sizes=(10,): Еквівалент кількості правил нечіткого виводу
# max_iter=1000: Кількість епох
# learning_rate_init=0.01: Швидкість навчання
anfis_emulator = MLPRegressor(
    hidden_layer_sizes=(15,),
    activation='tanh',  # Sigmoid-подібна функція, типова для нейронечітких систем
    solver='adam',
    max_iter=2000,
    learning_rate_init=0.01,
    random_state=42
)

anfis_emulator.fit(X_train, y_train)

# ==========================================
# 6. Перевірка адекватності (Валідація)
# ==========================================
print("--- Перевірка адекватності ---")
y_pred = anfis_emulator.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Середньоквадратична помилка (MSE): {mse:.4f}")
print(f"Коефіцієнт детермінації (R^2): {r2:.4f}")

# ==========================================
# 5. Побудова поверхні нечіткого виводу
# ==========================================
print("--- Візуалізація ---")

# Створення сітки для графіку
x_range = np.linspace(0, 100, 50)
y_range = np.linspace(1, 50, 50)
X_grid, Y_grid = np.meshgrid(x_range, y_range)

# Прогонимо сітку через навчену модель
Z_grid = anfis_emulator.predict(np.column_stack((X_grid.ravel(), Y_grid.ravel())))
Z_grid = Z_grid.reshape(X_grid.shape)

# Побудова 3D графіку
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Поверхня
surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', alpha=0.8, edgecolor='none')
# Точки реальних даних (тестові)
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='r', s=10, label='Тестові дані')

ax.set_title('Поверхня "Нечіткого" Виводу (RPS vs Size -> CPU Load)')
ax.set_xlabel('RPS (Запити)')
ax.set_ylabel('Size (Розмір)')
ax.set_zlabel('CPU Load (%)')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.legend()

print("Відображення графіку...")
plt.show()