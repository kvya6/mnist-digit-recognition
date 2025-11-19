from tensorflow.keras.models import load_model
import numpy as np
from train import x_test, y_test
from utils import display_samples


model = load_model('../models/mnist_cnn.h5')
loss, acc = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {acc*100:.2f}%')


# Predict on some samples
y_pred = model.predict(x_test[:10])
display_samples(x_test[:10], y_test[:10], y_pred)