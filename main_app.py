from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)

# Load the saved model
model_path = 'classifier-resnet-weights_fine_tuning_16b_345.hdf5'
model = tf.keras.models.load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the request contains an uploaded file
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})

        # Read the uploaded image file
        file = request.files['file']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Preprocess the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (256, 256))  # Resize the image
        img = img / 255.0  # Normalize the image
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Perform prediction using the loaded model
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)

        # Define result based on predicted class
        if predicted_class == 1:
            result = "There's a Tumor"
        else:
            result = "There is no Tumor"

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
