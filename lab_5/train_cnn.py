import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# --- 1. Завантаження та підготовка даних MNIST ---

# Завантажуємо набір даних
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Перевірка розмірності
print(f"Форма тренувальних даних: {x_train.shape}")  # (60000, 28, 28)
print(f"Форма тестових даних: {x_test.shape}")  # (10000, 28, 28)

# Нормалізація зображень:
# Пікселі мають значення від 0 до 255.
# Ми нормалізуємо їх до діапазону [0, 1] для кращої роботи мережі.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Додавання каналу кольору (CNN очікує (batch, height, width, channels))
# Наші зображення сірі, тому 1 канал
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(f"Нова форма тренувальних даних: {x_train.shape}")  # (60000, 28, 28, 1)

# Перетворення міток (y) у категоріальний формат (one-hot encoding)
# Наприклад, цифра '5' стане вектором [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# --- 2. Розробка (побудова) моделі CNN ---

model = keras.Sequential(
    [
        # Вхідний шар з формою (28, 28, 1)
        keras.Input(shape=(28, 28, 1)),

        # Шар 1: Згортка + Активація
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),

        # Шар 2: Підвибірка (Пулінг)
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Шар 3: Згортка + Активація
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),

        # Шар 4: Підвибірка (Пулінг)
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Шар 5: Випрямлення
        layers.Flatten(),

        # Шар 6: Dropout для боротьби з перенавчанням
        layers.Dropout(0.5),

        # Шар 7: Повнозв'язний
        layers.Dense(128, activation="relu"),  # Використовував 128 замість 100

        # Шар 8: Вихідний (10 класів, softmax)
        layers.Dense(num_classes, activation="softmax"),
    ]
)

# Відображення структури моделі
model.summary()

# --- 3. Компіляція та навчання моделі ---

# Компіляція: визначаємо оптимізатор, функцію втрат та метрики
model.compile(
    loss="categorical_crossentropy",  # Для класифікації з >2 класами
    optimizer="adam",  # Популярний оптимізатор
    metrics=["accuracy"]  # Нас цікавить точність
)

# Параметри навчання
batch_size = 128
epochs = 15  # 15 епох зазвичай дають >99% точності

print("\n--- Початок навчання мережі ---")

# Навчання моделі
# history збереже дані про втрати/точність на кожній епосі
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test)  # Використовуємо тестові дані для валідації
)

print("--- Навчання завершено ---")

# Збереження навченої моделі
model.save("mnist_cnn_model.h5")
print("Модель збережено у файл 'mnist_cnn_model.h5'")

# --- 4. Оцінка моделі (частина демонстрації) ---
score = model.evaluate(x_test, y_test, verbose=0)
print("\n--- Оцінка на тестових даних ---")
print(f"Втрати на тесті (Test loss): {score[0]:.4f}")
print(f"Точність на тесті (Test accuracy): {score[1] * 100:.2f}%")

# --- Графіки навчання (дуже корисні для звіту) ---
plt.figure(figsize=(12, 4))

# Графік точності
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Точність на навчанні')
plt.plot(history.history['val_accuracy'], label='Точність на валідації')
plt.title('Графік точності (Accuracy)')
plt.xlabel('Епоха')
plt.ylabel('Точність')
plt.legend()

# Графік втрат
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Втрати на навчанні')
plt.plot(history.history['val_loss'], label='Втрати на валідації')
plt.title('Графік втрат (Loss)')
plt.xlabel('Епоха')
plt.ylabel('Втрати')
plt.legend()

plt.show()
