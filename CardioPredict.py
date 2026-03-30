from flask import Flask, request, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model and scaler
model_data = joblib.load("heart_disease_model.pkl")
model = model_data["model"]
features = model_data["feature_names"]
scaler = joblib.load("heart_scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Collect form inputs
        data = [float(request.form[f]) for f in features]
        df = pd.DataFrame([data], columns=features)
        scaled = scaler.transform(df)
        prediction = model.predict(scaled)[0]

        result = "Disease Risk" if prediction == 1 else "Healthy"
        return render_template("result.html", prediction=result)
    else:
        # Show the input form when visiting the site
        return render_template("index.html")

# Optional health check route
@app.route("/ping")
def ping():
    return "App is alive!"

if __name__ == "__main__":
    # Bind to 0.0.0.0 and port 5000 for Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)