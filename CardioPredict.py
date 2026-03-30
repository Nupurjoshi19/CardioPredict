from flask import Flask, request, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Try loading model and scaler safely
try:
    model_data = joblib.load("heart_disease_model.pkl")
    if isinstance(model_data, dict):
        model = model_data["model"]
        features = model_data["feature_names"]
    else:
        model = model_data
        # If features aren’t stored, define them manually
        features = ['age','sex','cp','trestbps','chol','fbs','restecg',
                    'thalach','exang','oldpeak','slope','ca','thal']
except Exception as e:
    print("Error loading model:", e)
    model = None
    features = []

try:
    scaler = joblib.load("heart_scaler.pkl")
except Exception as e:
    print("Error loading scaler:", e)
    scaler = None

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST" and model is not None and scaler is not None:
        try:
            data = [float(request.form[f]) for f in features]
            df = pd.DataFrame([data], columns=features)
            scaled = scaler.transform(df)
            prediction = model.predict(scaled)[0]

            result = "Disease Risk" if prediction == 1 else "Healthy"
            return render_template("result.html", prediction=result)
        except Exception as e:
            return f"Error during prediction: {e}"
    else:
        return render_template("index.html")

@app.route("/ping")
def ping():
    return "App is alive!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)