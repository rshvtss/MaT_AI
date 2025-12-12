import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os


def preprocess_image(image_path):
    """Завантажує та готує власне зображення для моделі."""

    # Відкриваємо картинку і перетворюємо у відтінки сірого ('L')
    img = Image.open(image_path).convert('L')

    # Інвертуємо кольори (якщо малювали чорним по білому).

    img = ImageOps.invert(img)

    # Змінюємо розмір до 28x28 (стандарт MNIST)
    img = img.resize((28, 28))

    # Перетворюємо на масив та нормалізуємо
    img_array = np.array(img)
    img_array = img_array.astype("float32") / 255.0

    # Додаємо виміри для Keras: (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)

    return img_array


# 1. Перевірка наявності моделі
model_filename = "mnist_cnn_model.h5"
if not os.path.exists(model_filename):
    print(f"ПОМИЛКА: Файл моделі '{model_filename}' не знайдено!")
    print("Спершу запустіть файл навчання (train_cnn.py).")
    exit()

model = keras.models.load_model(model_filename)


image_filename = "my_digit.png"

if not os.path.exists(image_filename):
    print(f"ПОМИЛКА: Файл зображення '{image_filename}' не знайдено!")
    print("Створіть картинку і покладіть її поруч зі скриптом.")
    exit()

# 2. Обробка та прогноз
try:
    processed_img = preprocess_image(image_filename)

    # Прогноз
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0]) * 100

    print(f"\n=== РЕЗУЛЬТАТ ===")
    print(f"Мережа розпізнала цифру: {predicted_class}")
    print(f"Впевненість: {confidence:.2f}%")

    # Показати, що саме "бачить" мережа після обробки
    plt.imshow(processed_img.reshape(28, 28), cmap="gray")
    plt.title(f"Мережа бачить: {predicted_class} ({confidence:.2f}%)")
    plt.axis("off")
    plt.show()

except Exception as e:
    print(f"Виникла помилка при обробці: {e}")