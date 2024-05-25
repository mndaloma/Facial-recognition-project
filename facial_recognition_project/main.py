import os
from src import train_model, test_model

# Train the model if it hasn't been trained yet
if not os.path.exists('face_recognition_model.h5'):
    train_model.train()

# Test the model to see initial results
test_model.test()
