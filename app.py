import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.write("Hello, Streamlit is working! 🎉")

# Load model, scaler, and encoder
with open("stroke_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler_Stroke.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("ordinal_encoding.pkl", "rb") as f:   # 👈 save your OrdinalEncoder during training
    encoder = pickle.load(f)

st.title("🧠 Stroke Prediction App")

# Collect user input
age = st.number_input("Age", min_value=0, max_value=120, value=30)
hypertension = st.selectbox("Hypertension (0=No, 1=Yes)", [0, 1])
heart_disease = st.selectbox("Heart Disease (0=No, 1=Yes)", [0, 1])
avg_glucose = st.number_input("Average Glucose Level", min_value=40.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)

gender = st.selectbox("Gender", ["Male", "Female"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])

# Create DataFrame for categorical features (in same order as training)
categorical_data = pd.DataFrame([[gender, ever_married, work_type, residence_type, smoking_status]],
                                columns=['gender','ever_married','work_type','Residence_type','smoking_status'])

# Encode categorical data
encoded_cats = encoder.transform(categorical_data)

# Combine numerical + encoded categorical
X_input = np.array([[age, hypertension, heart_disease, avg_glucose, bmi]])
X_input = np.hstack([X_input, encoded_cats])

# Scale input
X_input = scaler.transform(X_input)

# Predict
if st.button("Predict Stroke Risk"):
    prediction = model.predict(X_input)
    st.success(f"Prediction: {'Stroke' if prediction[0] == 1 else 'No Stroke'}")
