from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model_svc.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    cgpa = float(request.form["cgpa"])
    iq = float(request.form["iq"])
    
    # Reshape input for model
    features = np.array([[cgpa, iq]])
    prediction = model.predict(features)[0]
    
    result = "Selected 🎉" if prediction == 1 else "Not Selected ❌"
    return render_template("index.html", prediction_text=f"Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True)
