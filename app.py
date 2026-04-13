# app.py
import streamlit as st
import pickle
import numpy as np

# Load model
@st.cache_resource
def load_model():
    with open("Model (3).pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title("ML Model Deployment with Streamlit")

st.write("Enter input features below:")

# Example: 3 input features (change as per your model)
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")

# Prediction
if st.button("Predict"):
    input_data = np.array([[feature1, feature2, feature3]])
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction[0]}")


# requirements.txt
# ----------------
# streamlit
# numpy
# scikit-learn
