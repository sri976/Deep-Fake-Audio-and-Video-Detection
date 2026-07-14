from flask import Flask, render_template, request
import pickle
import os
from feature_extraction import extract_features
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
model = pickle.load(open("model.pkl", "rb"))
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return "No file uploaded"
    file = request.files["audio"]
    if file.filename == "":
        return "No file selected"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)
    features = extract_features(filepath)
    prediction = model.predict([features])
    if prediction[0] == 0:
        result = "Real Audio"
    else:
        result = "Fake Audio"
    return render_template("index.html", prediction=result)
if __name__ == "__main__":
    app.run(debug=True)
