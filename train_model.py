import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from feature_extraction import extract_features
X = []
y = []
real_folder = "dataset/real"
fake_folder = "dataset/fake"
# Check folders
print("Checking dataset folders...")
if not os.path.exists(real_folder):
    print(f"ERROR: Folder not found -> {real_folder}")
    exit()
if not os.path.exists(fake_folder):
    print(f"ERROR: Folder not found -> {fake_folder}")
    exit()
print("Real folder files:", os.listdir(real_folder))
print("Fake folder files:", os.listdir(fake_folder))
# Load Real Audio
for file in os.listdir(real_folder):
    if file.lower().endswith((".wav", ".mp3")):
        try:
            path = os.path.join(real_folder, file)
            features = extract_features(path)
            if features is not None:
                X.append(features)
                y.append(0)
        except Exception as e:
            print(f"Error processing {file}: {e}")
# Load Fake Audio
for file in os.listdir(fake_folder):
    if file.lower().endswith((".wav", ".mp3")):
        try:
            path = os.path.join(fake_folder, file)
            features = extract_features(path)
            if features is not None:
                X.append(features)
                y.append(1)
        except Exception as e:
            print(f"Error processing {file}: {e}")
print("Total samples loaded:", len(X))
print("Total labels loaded:", len(y))
# Check if data exists
if len(X) == 0:
    print("ERROR: No audio files were loaded.")
    print("Make sure dataset/real and dataset/fake contain .wav or .mp3 files.")
    exit()
X = np.array(X)
y = np.array(y)
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Test model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", round(accuracy * 100, 2), "%")
# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model Saved Successfully!")
print("Saved as: model.pkl")