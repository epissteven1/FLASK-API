import librosa
import json
import numpy as np

audio_path = 'sample data/Ikagagalak kitang makilala_01.wav'  # your sample audio file

# Load audio (e.g., 16kHz sampling rate)
y, sr = librosa.load(audio_path, sr=8000)

# Extract MFCC (n_mfcc=40)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

# mfcc shape will be (40, frames). You may want to fix length:
# Example: take first 100 frames or pad if less
num_frames = 100
if mfcc.shape[1] < num_frames:
    pad_width = num_frames - mfcc.shape[1]
    mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
else:
    mfcc = mfcc[:, :num_frames]

# Convert to list for JSON serialization
mfcc_list = mfcc.tolist()

data = {"mfcc": mfcc_list}
with open('mfcc_sample.json', 'w') as f:
    json.dump(data, f)
