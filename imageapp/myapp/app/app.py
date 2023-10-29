from flask import Flask, render_template, request, jsonify
from tensorflow import keras
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name)

# Loading my pre-trained model
model = keras.models.load_model('model/model.h5')

# Define a function to preprocess the uploaded image
def preprocess_image(file):
    # Load and preprocess the image
    img = Image.open(file)
    img = img.resize((256, 256))  # Resize to match your model's input size
    img = np.array(img) / 255.0  # Normalize the image data
    return img

# Define your class labels
class_labels = ["parkinglot", "tenniscourt", "storagetanks","sparseresidential","runway","river","overpass","mobilehomepark","mediumresidential","intersection","harbor","golfcourse","freeway","forest","denseresidential","chaparral","buildings","baseballdiamond","beach","airplane","agricultural"]  # Adjust to match your actual class labels

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Preprocess the image
        img = preprocess_image(file)
        # Make a prediction using your model
        prediction = model.predict(np.expand_dims(img, axis=0))

        # Get the predicted class label
        predicted_class = class_labels[np.argmax(prediction)]

        return jsonify({'prediction': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

