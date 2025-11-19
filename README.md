# MNIST Handwritten Digit Recognition

This project is a **Deep Learning mini-project** to classify handwritten digits (0–9) using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.  

The project includes training, evaluation, and prediction functionalities.

---

## Folder Structure

DL/
├── models/ # Saved trained models
├── results/ # Training plots and results
├── src/ # Source code
│ ├── cnn_model.py # CNN architecture
│ ├── train.py # Training script
│ ├── utils.py # Helper functions (plotting, display)
│ ├── evaluate.py # Evaluation utilities
│ ├── predict.py # Predict new images
│ ├── main.py # Entry point for evaluation/prediction
│ └── run_predictions.py # Combined test and custom image predictions
├── requirements.txt # Python dependencies
└── README.md


---

## Installation

1. **Clone or copy the project folder**  
2. **Open VS Code** (or terminal) in the project folder  
3. **Create a virtual environment**:

```powershell
python -m venv venv
& venv\Scripts\Activate.ps1


Install dependencies:

pip install -r requirements.txt


If installation fails, you can install missing packages manually:
pip install tensorflow matplotlib numpy
Training the Model

Navigate to src folder:

cd src


Run training:

python train.py


Trains a CNN on the MNIST dataset

Saves the model in ../models/mnist_cnn.h5

Saves training curves in ../results/loss_accuracy.png

Evaluating the Model

Option 1: Using main.py

python main.py


Displays sample test images with true and predicted labels

Option 2: Using run_predictions.py (recommended)

python run_predictions.py


Displays the first 10 MNIST test images with predictions

Optionally predicts your own 28×28 grayscale image by entering its path when prompted

Predicting a New Image

Prepare a 28×28 grayscale image of a digit (e.g., digit.png).

Run run_predictions.py or predict.py:

from predict import predict_image

print(predict_image('digit.png'))


Prints the predicted digit (0–9)

You can also visualize it using matplotlib

Input and Output
Script	Input	Output
train.py	MNIST dataset	Trained model file + training graphs
main.py	First 10 test images	Images displayed with true & predicted labels
run_predictions.py	MNIST test images + optional custom image	Images displayed + predicted digits
predict.py	Your own 28×28 image	Predicted digit (0–9)
Use of the Project

Digit recognition: Automatically reads handwritten digits from images

Learning: Understand CNNs, image preprocessing, model training, and evaluation

Practical demo: Predicts digits from both MNIST and custom images
