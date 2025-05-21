from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import librosa
import json
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from pydub.effects import normalize
from pydub.silence import detect_nonsilent

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Parameters
TARGET_SR = 8000
N_MFCC = 40
MAX_FRAMES = 100

# Load label mapping
with open('model/labels.json', 'r') as f:
    labels = json.load(f)

# Load TFLite model
MODEL_PATH = 'model/Filipino_speech_recognition.tflite'
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def trim_silence_pydub(audio, silence_thresh=-50, min_silence_len=100):
    nonsilent = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    if nonsilent:
        start, end = nonsilent[0][0], nonsilent[-1][1]
        return audio[start:end]
    return audio

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    print(f"Received file: {filepath}")

    wav_path = filepath.rsplit('.', 1)[0] + '.wav'
    mfcc_json_path = os.path.join(UPLOAD_FOLDER, 'last_mfcc.json')

    try:
        # Auto-detect audio format from the file content/extension
        audio = AudioSegment.from_file(filepath)  
        audio = audio.set_frame_rate(TARGET_SR).set_channels(1)
        audio = normalize(audio)

        # Trim silence
        audio = trim_silence_pydub(audio)

        # Export to WAV for librosa
        audio.export(wav_path, format="wav")

        # Load trimmed audio with librosa
        y, sr = librosa.load(wav_path, sr=TARGET_SR)

        # Further trim silence with librosa
        y, _ = librosa.effects.trim(y, top_db=30)

        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

        # Optional debugging visualization - comment out if running headless
        # plt.figure(figsize=(10, 4))
        # plt.imshow(mfcc, cmap='viridis', aspect='auto')
        # plt.title("MFCC")
        # plt.colorbar()
        # plt.tight_layout()
        # plt.show()

        # Pad or truncate MFCC to fixed length
        if mfcc.shape[1] < MAX_FRAMES:
            pad_width = MAX_FRAMES - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :MAX_FRAMES]

        mfcc_array = mfcc.astype(np.float32)
        mfcc_array = np.expand_dims(mfcc_array, axis=0)  # (1, 40, 100)
        mfcc_array = np.expand_dims(mfcc_array, axis=-1) # (1, 40, 100, 1)

        # Save MFCC JSON for debugging (optional)
        with open(mfcc_json_path, 'w') as mfcc_file:
            json.dump(mfcc_array.squeeze(-1).tolist(), mfcc_file)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], mfcc_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_index = int(np.argmax(output_data))
        predicted_label = labels[predicted_index]
        confidence = float(np.max(output_data))

        return jsonify({
            'prediction': predicted_label,
            'confidence': confidence,
            'saved_file': filepath,
            'mfcc_json': mfcc_json_path
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/')
def index():
    return "✅ Flask Speech Recognition API is running! Use POST /predict to send audio."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
