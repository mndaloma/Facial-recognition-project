import numpy as np
import tensorflow as tf
from src.dataset import load_dataset

def test():
    # Load the dataset
    X, y, target_names, _, h, w = load_dataset()
    
    # Normalize the pixel values to the range [0, 1]
    X = X / 255.0

    # Load the trained model from a file
    model = tf.keras.models.load_model('face_recognition_model.h5')
    # Predict the labels for the entire dataset
    y_pred = np.argmax(model.predict(X), axis=1)

    # Print the first 10 predictions
    for i in range(10):
        print(f'Predicted: {target_names[y_pred[i]]}, Actual: {target_names[y[i]]}')

if __name__ == '__main__':
    test()
