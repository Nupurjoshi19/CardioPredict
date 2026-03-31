#  CardioPredict

**Interactive Streamlit app powered by Machine Learning to predict heart disease risk.**

🔗 **Live Demo:** [CardioPredict on Streamlit Cloud](https://cardiopredict-19.streamlit.app/)

---
Run the app:
streamlit run CardioPredict.py


Open http://localhost:8501 in your browser.

## 📖 Overview
CardioPredict is a web application that leverages a Logistic Regression model trained on the UCI Heart Disease dataset.  
It provides an intuitive interface with sliders and dropdowns for patient attributes, and outputs:

- ✅ Prediction (Healthy / Disease Risk)  
- 📊 Confidence score & risk level  
- 📈 Probability visualization with Plotly charts  

This project is for **educational purposes only** and is **not a substitute for professional medical advice**.

---

## 🖥️ Features
- Modern, interactive UI built with **Streamlit**  
- Real‑time predictions using a trained ML model  
- Probability visualization with **Plotly**  
- Sidebar with model info and key risk factors  
- Input summary for transparency  

---

## ⚙️ Tech Stack
- **Python 3.10+**
- **Streamlit** for UI
- **Scikit‑learn** for ML
- **Pandas / NumPy** for data handling
- **Plotly** for charts
- **Joblib** for model persistence

---
📊 Dataset
- **Source**: UCI Heart Disease dataset
- **Samples**: 303 patients
- **Features**: 13 attributes (age, sex, chest pain type, cholesterol, blood pressure, etc.)
- **Model**: Logistic Regression (~85% accuracy)

---


## 🚀 Run Locally
Clone the repository and install dependencies:

```bash
git clone https://github.com/Nupurjoshi19/CardioPredict.git
cd CardioPredict
pip install -r requirements.txt

