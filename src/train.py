# =========================================
# src/train.py - MNIST Handwritten Digit Recognition Training Script
# =========================================

import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from cnn_model import build_cnn
from utils import plot_history

# Paths
MODEL_DIR = '../models/'
RESULTS_DIR = '../results/'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------
# Load and preprocess MNIST dataset
# -------------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# -------------------------------
# Build CNN model
# -------------------------------
model = build_cnn()
model.summary()

# -------------------------------
# Train model
# -------------------------------
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.1,
    verbose=1
)

# -------------------------------
# Save model
# -------------------------------
model.save(os.path.join(MODEL_DIR, 'mnist_cnn.h5'))
print(f'Model saved at {MODEL_DIR}mnist_cnn.h5')

# -------------------------------
# Plot training history
# -------------------------------
plot_history(history, save_path=os.path.join(RESULTS_DIR, 'loss_accuracy.png'))
print(f'Training curves saved at {RESULTS_DIR}loss_accuracy.png')