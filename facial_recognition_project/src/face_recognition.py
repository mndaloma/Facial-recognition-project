import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
print("TensorFlow is installed correctly, and Sequential is available.")

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
print("TensorFlow is installed correctly, and the layers are available.")


def build_model(input_shape, num_classes):
    model = Sequential([
        # First convolutional layer with 32 filters and ReLU activation
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),  # Max pooling to reduce spatial dimensions
        # Second convolutional layer with 64 filters and ReLU activation
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),  # Max pooling
        # Third convolutional layer with 128 filters and ReLU activation
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),  # Max pooling
        Flatten(),  # Flatten the 3D output to 1D
        Dense(512, activation='relu'),  # Fully connected layer with 512 units
        Dropout(0.5),  # Dropout for regularization
        Dense(num_classes, activation='softmax')  # Output layer with softmax activation
    ])
    # Compile the model with Adam optimizer and sparse categorical crossentropy loss
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
