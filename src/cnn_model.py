# =========================================
# src/cnn_model.py - CNN architecture for MNIST
# =========================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

def build_cnn(input_shape=(28,28,1), num_classes=10):
    """
    Builds and compiles a simple CNN for MNIST digit classification.
    """
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
