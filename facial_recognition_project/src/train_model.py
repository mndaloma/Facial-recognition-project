from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.dataset import load_dataset
from src.face_recognition import build_model

def train():
    # Load the dataset
    X, y, target_names, n_samples, h, w = load_dataset()
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Normalize the pixel values to the range [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Build the CNN model
    model = build_model((h, w, 3), len(target_names))
    # Train the model with training data
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Save the trained model to a file
    model.save('face_recognition_model.h5')

if __name__ == '__main__':
    train()
