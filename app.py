import streamlit as st
import pickle
import numpy as np

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    with open("Model (3).pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="ML Model App",
    page_icon="🤖",
    layout="centered"
)

# ----------------------------
# Custom Styling (Frontend Boost)
# ----------------------------
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            border-radius: 10px;
            padding: 10px 24px;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Title
# ----------------------------
st.title("🤖 Machine Learning App")
st.write("Enter the input values below to get predictions.")

# ----------------------------
# Inputs (EDIT THESE BASED ON YOUR MODEL)
# ----------------------------
st.subheader("Input Features")

feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

# Add more if needed

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict"):
    try:
        input_data = np.array([[feature1, feature2, feature3]])
        prediction = model.predict(input_data)

        st.success(f"Prediction: {prediction[0]}")

    except Exception as e:
        st.error(f"Error: {e}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
