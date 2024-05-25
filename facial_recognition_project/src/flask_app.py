from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import cv2
from dataset import load_dataset

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('face_recognition_model.h5')
_, _, target_names, _, h, w = load_dataset()

def preprocess_image(image):
    # Resize the image to the required input size
    image = cv2.resize(image, (w, h)).astype(np.float32) / 255.0
    # Expand dimensions to match the input shape expected by the model
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def index():
    return "Facial Recognition API is running!"

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico')

@app.route('/predict', methods=['POST'])
def predict():
    # Read the image file from the request
    file = request.files['image']
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # Preprocess the image
    image = preprocess_image(image)
    
    # Make a prediction using the model
    prediction = np.argmax(model.predict(image), axis=1)
    predicted_name = target_names[prediction[0]]
    
    return jsonify({'predicted_name': predicted_name})

if __name__ == '__main__':
    app.run(debug=True)
