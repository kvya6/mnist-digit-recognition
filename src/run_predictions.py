# ========== run_predictions.py ==========
import numpy as np
import matplotlib.pyplot as plt
from train import model, x_test, y_test
from predict import predict_image
from tensorflow.keras.preprocessing import image
import os

# -------------------------
# Part 1: Show MNIST test predictions
# -------------------------
print("Displaying predictions on the first 10 test images...")
y_pred = model.predict(x_test[:10])
y_pred_labels = np.argmax(y_pred, axis=1)

for i in range(10):
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.title(f"True: {y_test[i]}, Predicted: {y_pred_labels[i]}")
    plt.axis('off')
    plt.show()

# -------------------------
# Part 2: Predict your own image
# -------------------------
img_path = input("Enter path to your 28x28 grayscale image (or leave empty to skip): ").strip()

if img_path and os.path.isfile(img_path):
    predicted_digit = predict_image(img_path)
    print(f"The model predicts: {predicted_digit}")

    img = image.load_img(img_path, color_mode='grayscale', target_size=(28,28))
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {predicted_digit}")
    plt.axis('off')
    plt.show()
else:
    print("No custom image provided or file not found. Skipping custom prediction.")
