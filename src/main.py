# ========== main.py ==========
# Entry point

from train import model, x_test, y_test
from evaluate import display_samples

if __name__ == '__main__':
    # Example: evaluate model on test set
    print('Evaluating model on test set...')
    y_pred = model.predict(x_test[:10])
    display_samples(x_test[:10], y_test[:10], y_pred)

    # Optional: predict from image file
    # from predict import predict_image
    # print(predict_image('digit.png'))
