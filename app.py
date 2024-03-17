from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the model outside of the Flask application instance
model = None

def load_model():
    global model
    model = tf.keras.models.load_model('trained_model/food_recognition_model.hdf5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            return redirect(url_for('predict', filename=filename))
    return render_template('index.html')

@app.route('/predict/<filename>')
def predict(filename):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = Image.open(img_path)
    img = img.resize((224, 224))  # Resize image if needed
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Ensure the model is loaded before making predictions
    if model is None:
        load_model()

    # Perform prediction using the loaded model
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Assuming you have a list of food classes
    food_classes = ['apple', 'banana', 'pizza', 'burger', 'salad', 'sushi', 'cake']

    return render_template('result.html', filename=filename, predicted_class=food_classes[predicted_class])

if __name__ == '__main__':
    # Load the model when the script is run directly
    load_model()
    app.run(debug=True)
