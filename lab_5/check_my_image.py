import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from tensorflow.keras import models


def preprocess_image(image_path):
    # Завантажуємо
    img = Image.open(image_path).convert('L')

    # Інвертуємо (якщо малювали чорним по білому - розкоментуйте рядок нижче)
    img = ImageOps.invert(img)

    # Ресайз та нормалізація
    img = img.resize((28, 28))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # batch dim
    img_array = np.expand_dims(img_array, axis=-1)  # channel dim
    return img_array


# Завантаження моделі
model = models.load_model("mnist_cnn_model.h5")

# Обробка вашого файлу
try:
    my_img = preprocess_image("my_digit.png")
    prediction = model.predict(my_img)
    result = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    print(f"Це цифра: {result} (Впевненість: {confidence:.2f}%)")

    plt.imshow(my_img.reshape(28, 28), cmap="gray")
    plt.title(f"AI бачить: {result}")
    plt.show()
except FileNotFoundError:
    print("Не знайдено файл 'my_digit.png'. Створіть його!")