from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

app = Flask(__name__)

# Load trained model
model = load_model('C:\\projects\\flower-recognition-project-using-cnn\\Model.keras')

# list of flower class names
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        # Convert the FileStorage object to a file-like object
        file_stream = io.BytesIO(file.read())
        # Preprocess the image
        test_image = image.load_img(file_stream, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image /= 255.0
        test_image = np.expand_dims(test_image, axis=0)

        # Predict the class
        result = model.predict(test_image)
        predicted_class = class_names[np.argmax(result)]

        return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
