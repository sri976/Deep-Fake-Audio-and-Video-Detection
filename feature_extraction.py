import librosa
import numpy as np
def extract_features(file_path):
    print("Reading:", file_path)
    audio, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )
    return np.mean(mfccs.T, axis=0)