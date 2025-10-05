from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model_scaled_svc.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    cgpa = float(request.form["cgpa"])
    iq = float(request.form["iq"])
    
    # Validation
    if not (0 <= cgpa <= 10):
        return render_template("index.html", prediction_text="❌ CGPA must be between 0 and 10.")
    if not (0 <= iq <= 500):  # example IQ range
        return render_template("index.html", prediction_text="❌ IQ must be between 0 and 500.")
    
    features = np.array([[cgpa, iq]])
    prediction = model.predict(features)[0]
    
    result = "Selected 🎉" if prediction == 1 else "Not Selected ❌"
    return render_template("index.html", prediction_text=f"Prediction: {result}")


if __name__ == "__main__":
    app.run(debug=True)
