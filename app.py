from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from pydub.effects import normalize
import os
import speech_recognition as sr

app = Flask(__name__)

# Directory to save uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Audio processing parameters
TARGET_SR = 8000  # Target sample rate for audio

from pydub.silence import split_on_silence

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    print(f"[INFO] Received file: {filepath}")

    try:
        # Load and preprocess audio
        audio = AudioSegment.from_file(filepath)
        audio = audio.set_frame_rate(TARGET_SR).set_channels(1)
        audio = normalize(audio)

        # === Optional: Silence trimming ===
        chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)
        if chunks:
            trimmed_audio = AudioSegment.silent(duration=500)
            for chunk in chunks:
                trimmed_audio += chunk + AudioSegment.silent(duration=200)
            audio = trimmed_audio

        # Export to WAV
        wav_path = filepath.rsplit('.', 1)[0] + '.wav'
        audio.export(wav_path, format="wav")
        print(f"[INFO] Converted and saved WAV file: {wav_path}")

        # Speech recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio_data = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio_data, language='tl-PH')
            print(f"[INFO] Transcribed text: {text}")
            return jsonify({
                'prediction': text,
                'engine': 'Google Speech Recognition',
                'saved_file': wav_path
            })
        except sr.UnknownValueError:
            return jsonify({'error': 'Could not understand the audio'})
        except sr.RequestError as e:
            return jsonify({'error': f'Google API request failed: {e}'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return "✅ Flask Speech Recognition API is running! Use POST /predict to send audio."



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
