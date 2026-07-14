import os
from feature_extraction import extract_features

for folder in ["real", "fake"]:
    folder_path = os.path.join("dataset", folder)

    for file in os.listdir(folder_path):
        if file.endswith(".mp3"):
            print(f"\nTesting {file}")

            try:
                features = extract_features(
                    os.path.join(folder_path, file)
                )

                print("OK", features.shape)

            except Exception as e:
                print("ERROR:", e)