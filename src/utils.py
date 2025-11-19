# =========================================
# src/utils.py - helper functions
# =========================================

import matplotlib.pyplot as plt
import numpy as np

def plot_history(history, save_path=None):
    plt.figure(figsize=(10,4))

    # Accuracy plot
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    # Loss plot
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    if save_path:
        plt.savefig(save_path)  # <- must be indented under the if
    plt.show()


def display_samples(x, y_true, y_pred, num_samples=5):
    plt.figure(figsize=(10,2))
    for i in range(num_samples):
        plt.subplot(1,num_samples,i+1)
        plt.imshow(x[i].reshape(28,28), cmap='gray')
        plt.title(f'T:{np.argmax(y_true[i])}\nP:{np.argmax(y_pred[i])}')
        plt.axis('off')
    plt.show()
