import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('logistic_model.pkl')

st.title("🚢 Titanic Survival Prediction App")

# Input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 30)
fare = st.slider("Fare Paid", 0.0, 500.0, 32.0)

# Convert inputs
sex_male = 1 if sex == "male" else 0
input_data = np.array([[pclass, age, fare, sex_male]])  # FIXED ORDER

# Predict
if st.button("Predict"):
    prob = model.predict_proba(input_data)[0][1]
    result = model.predict(input_data)[0]
    
    st.write(f"**Survival Probability:** {prob:.2f}")
    if result == 1:
        st.success("🎉 The passenger is likely to **survive**.")
    else:
        st.error("💀 The passenger is likely to **not survive**.")
