from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("tongue_image_classifier_v2.h5")

# Class index to label mapping
class_labels = {0: "Healthy", 1: "Moderate", 2: "Unhealthy"}

# Preprocess the uploaded image
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.route("/predict", methods=["POST"])
def predict():
    if "tongueImage" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["tongueImage"]
    img_bytes = file.read()

    processed = preprocess_image(img_bytes)
    prediction = model.predict(processed)
    label_index = int(np.argmax(prediction))
    result = class_labels[label_index]

    return jsonify({"prediction": result})

@app.route("/")
def home():
    return send_from_directory("static", "page1.html")

@app.route("/upload")
def upload_page():
    return send_from_directory("static", "upload.html")

@app.route("/result")
def result_page():
    return send_from_directory("static", "result.html")

if __name__ == "__main__":
    app.run(debug=True)
