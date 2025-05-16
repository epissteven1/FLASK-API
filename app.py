from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import librosa
import json
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load label mapping
with open('model/labels.json', 'r') as f:
    labels = json.load(f)

# Load TFLite model
MODEL_PATH = 'model/Filipino_speech_recognition.tflite'
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    print(f"Received file: {filepath}")

    try:
        # Load audio using librosa (convert to mono, 8kHz)
        y, sr = librosa.load(filepath, sr=8000)

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # Ensure fixed number of frames (pad or trim to 100)
        num_frames = 100
        if mfcc.shape[1] < num_frames:
            pad_width = num_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :num_frames]

        # Prepare for model input
        mfcc_array = mfcc.astype(np.float32)
        mfcc_array = np.expand_dims(mfcc_array, axis=0)  # (1, 40, 100)
        mfcc_array = np.expand_dims(mfcc_array, axis=-1) # (1, 40, 100, 1)

        interpreter.set_tensor(input_details[0]['index'], mfcc_array)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])  # (1, num_labels)
        predicted_index = int(np.argmax(output_data))
        predicted_label = labels[predicted_index]

        return jsonify({'prediction': predicted_label})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.remove(filepath)  # Clean up uploaded file


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)