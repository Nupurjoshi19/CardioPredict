from flask import Flask, request, render_template
import joblib
import pandas as pd

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

    return render_template("index.html")  # input form

if __name__ == "__main__":
    app.run(debug=True)