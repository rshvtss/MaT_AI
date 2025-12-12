import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Concatenate, SimpleRNN, Reshape

# --- DATA ---
x = np.linspace(-10, 10, 2000)
y = np.cos(np.sin(x)) * np.sin(x/2)

def modelTesting(model, label, axs, idx):
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, y, epochs=40, batch_size=100)

    y_pred = model.predict(x)

    axs[idx].plot(x, y, label='Actual')
    axs[idx].plot(x, y_pred, label=label)
    axs[idx].set_title(label)
    axs[idx].legend()

# --- MODELS ---
def feedforwardCreation(layers, neurons):
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_shape=(1,)))
    for _ in range(layers - 1):
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1))
    return model

def cascadeforwardCreation(layers, neurons):
    inputLayer = Input(shape=(1,))
    current = Dense(neurons, activation='relu')(inputLayer)
    for _ in range(layers - 1):
        current = Concatenate()([inputLayer, current])
        current = Dense(neurons, activation='relu')(current)
    output = Dense(1)(current)
    return Model(inputs=inputLayer, outputs=output)

def elmanCreation(layers, neurons):
    model = Sequential()
    model.add(Reshape((1, 1), input_shape=(1,)))
    model.add(SimpleRNN(neurons, return_sequences=True, activation='relu'))
    for _ in range(layers - 1):
        model.add(SimpleRNN(neurons, return_sequences=True, activation='relu'))
    model.add(Dense(1))
    model.add(Reshape((1,)))
    return model

# --- RUN ---
fig, axs = plt.subplots(3, 2, figsize=(12, 10))
axs = axs.ravel()

models = [
    (feedforwardCreation(1, 10), "FF 1x10"),
    (feedforwardCreation(1, 20), "FF 1x20"),
    (cascadeforwardCreation(1, 20), "Cascade 1x20"),
    (cascadeforwardCreation(2, 10), "Cascade 2x10"),
    (elmanCreation(1, 15), "Elman 1x15"),
    (elmanCreation(3, 5), "Elman 3x5"),
]

for i, (model, name) in enumerate(models):
    modelTesting(model, name, axs, i)

plt.tight_layout()
plt.show()
