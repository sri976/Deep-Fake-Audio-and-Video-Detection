import joblib
from feature_extraction import extract_features

model = joblib.load("deepfake_model.pkl")

file_path = "dataset/real/real_hindi.mp3"  # change file

features = extract_features(file_path)
prediction = model.predict([features])

if prediction[0] == 0:
    print("REAL AUDIO")
else:
    print("FAKE AUDIO")