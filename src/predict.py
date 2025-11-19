from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model('../models/mnist_cnn.h5')

def predict_image(img_path):
    """
    Predicts the digit in a 28x28 grayscale image.
    """
    img = image.load_img(img_path, color_mode='grayscale', target_size=(28,28))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)
    return np.argmax(pred)

# Example usage:
# print(predict_image('digit.png'))
