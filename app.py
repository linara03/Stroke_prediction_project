import pickle
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import numpy as np
import random
import sklearn

# =========================
# Load model & dataset
# =========================
def get_data():
    model, scalar = pickle.load(open('stroke_mdl.pkl','rb'))
    df = pd.read_csv('stroke_dataset.csv')
    return model, scalar, df

def YesNo(n):
    return 1 if n == 'Yes' else 0

model, scalar, df = get_data()

# =========================
# Page Config
# =========================
st.set_page_config(page_title="🧠 Stroke Predictor", layout="centered")

# =========================
# Custom CSS for Styling & Animations
# =========================
st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background: linear-gradient(135deg, #f0f9ff, #e0f7fa);
        font-family: 'Segoe UI', sans-serif;
        animation: fadeIn 1.2s ease-in;
    }

    /* Fade-in animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Title styling */
    h1 {
        font-size: 40px !important;
        color: #2E7D32 !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
        animation: pulse 2s infinite;
    }

    /* Pulse animation for title */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.03); }
        100% { transform: scale(1); }
    }

    /* Sliders */
    .stSlider label {
        font-size: 18px;
        font-weight: bold;
        color: #00695C;
    }
    .stSlider .css-14xtw13 {
        background: linear-gradient(to right, #4CAF50, #81C784);
        border-radius: 10px;
    }

    /* Radio buttons */
    .stRadio label {
        font-size: 17px;
        font-weight: 500;
        color: #1565C0;
    }

    /* Number input */
    .stNumberInput label {
        font-size: 17px;
        font-weight: bold;
        color: #AD1457;
    }

    /* Selectbox */
    .stSelectbox label {
        font-size: 17px;
        font-weight: bold;
        color: #4527A0;
    }

    /* Prediction box */
    .prediction-box {
        animation: slideIn 1s ease;
        border-radius: 12px;
        padding: 20px;
        background: #ffffff;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    }

    @keyframes slideIn {
        from { transform: translateY(30px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Title
# =========================
st.markdown(
    """
    <div style="text-align: center; padding: 20px;">
        <h1>🧠 Stroke Predictor</h1>
        <p style="font-size:18px; color:gray;">
            Fill in the information below to <b>predict your risk of stroke</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# User Inputs
# =========================
age = st.slider('Age', 0, 100, 50)
sex = st.radio("Gender", ["Male", "Female"])
hypertension = st.radio("Hypertension", ["Yes","No"])
heart_disease = st.radio("Heart Disease", ["Yes","No"])
avg_gl_level = st.slider('Average Glucose Level', 0, 300, value=int(df['Average_Glucose_Level'].mean()))
bmi = st.slider('Body Mass Index (BMI)', 0, 70, value=int(df['BMI'].mean()))
smoking_status = st.selectbox('Smoking Status', df['Smoking_Status'].unique())
alcohol_intake = st.number_input("Alcohol Intake (units per week)", min_value=0, max_value=50, value=0)
blood_pressure = st.slider('Blood Pressure (mmHg)', 50, 200, value=int(df['Blood_Pressure'].mean()))
cholesterol = st.slider('Cholesterol Level (mg/dL)', 100, 400, value=int(df['Cholesterol'].mean()))
family_history = st.radio("Family History of Stroke", ["Yes", "No"])
mri_result = st.selectbox('MRI Result', df['MRI_Result'].unique())

# Encode gender
sex = 1 if sex == "Male" else 0

# Map smoking status
value_list = [1, 2, 3, 4]
smoking_dict = {key:value for key, value in zip(df['Smoking_Status'].unique(), value_list)}

user_values = [[
    age, sex, YesNo(hypertension), YesNo(heart_disease), avg_gl_level, bmi,
    smoking_dict.get(smoking_status), alcohol_intake, blood_pressure,
    cholesterol, YesNo(family_history), mri_result
]]

st.markdown("---")

# =========================
# Prediction
# =========================
if st.button("🔍 Predict My Risk"):
    prob = model.predict_proba(scalar.transform(user_values))[0][1]
    percentage = round(prob * 100, 2)

    st.markdown(
        f"""
        <div class="prediction-box" style="text-align:center;">
            <h2 style="color:#4CAF50;"> Predicted Stroke Risk: {percentage}% </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    if percentage >= 60:
        st.error("⚠️ High Risk of Stroke")
        st.markdown(
            """
            **Recommended Actions:**
            - 🏥 Consult a doctor immediately  
            - 💉 Get a full medical check-up (blood pressure, glucose, cholesterol)  
            - 💊 Follow prescribed medication if any  
            - 🚭 Avoid smoking and alcohol completely  
            - 🥗 Maintain a healthy diet and reduce salt intake  
            """
        )
    elif 30 <= percentage < 60:
        st.warning("⚠️ Moderate Risk of Stroke")
        st.markdown(
            """
            **Recommended Actions:**
            - 🩺 Consider visiting a doctor for preventive screening  
            - 📉 Monitor your blood pressure and glucose regularly  
            - 🏃 Start moderate exercise (walking, yoga, etc.)  
            - 🍷 Reduce alcohol consumption and quit smoking  
            """
        )
    else:
        st.success("✅ Low Risk of Stroke")
        st.markdown(
            """
            **Recommended Actions:**
            - 🥦 Maintain a healthy lifestyle  
            - 🏋️ Exercise regularly (30 mins a day)  
            - 🩸 Keep blood pressure, glucose, and cholesterol in check  
            - 🚭 Avoid excessive smoking/alcohol  
            - 🔄 Go for routine health check-ups once a year  
            """
        )
