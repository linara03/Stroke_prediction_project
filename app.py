import pickle
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import numpy as np
import random
import sklearn

def get_data():
    model, scalar = pickle.load(open('stroke_mdl.pkl','rb'))
    df = pd.read_csv('stroke_dataset.csv')
    return model, scalar, df

def YesNo(n):
    return 1 if n == 'Yes' else 0

model, scalar, df = get_data()


st.set_page_config(page_title="Stroke Predictor", layout="centered")


st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background: #f9f9f9;
        font-family: 'Segoe UI', sans-serif;
        color: #333333;
        animation: fadeIn 1s ease-in;
    }

    /* Fade-in */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* Title */
    h1 {
        font-size: 36px !important;
        color: #2E7D32 !important;
        font-weight: 600 !important;
        margin-bottom: 0.5em;
    }

    p {
        font-size: 16px;
        color: #555555;
    }

    /* Labels (sliders, radios, etc.) */
    label {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #333333 !important;
    }

    /* Inputs */
    .stSlider, .stRadio, .stSelectbox, .stNumberInput {
        margin-bottom: 1.2em;
    }

    /* Prediction box */
    .prediction-box {
        border-radius: 10px;
        padding: 20px;
        background: #ffffff;
        border: 1px solid #e0e0e0;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
    }

    .prediction-box h2 {
        margin: 0;
        font-size: 24px;
        color: #2E7D32;
        font-weight: 700;
    }

    /* Streamlit success/warning/error boxes */
    .stAlert {
        font-size: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div style="text-align: center; padding: 15px;">
        <h1>Stroke Predictor</h1>
        <p>Fill in the information below to <b>predict your risk of stroke</b>.</p>
    </div>
    """,
    unsafe_allow_html=True
)


age = st.slider('Age', 0, 100, 50)
sex = st.radio("Gender", ["Male", "Female"])
hypertension = st.radio("Hypertension", ["Yes","No"])
heart_disease = st.radio("Heart Disease", ["Yes","No"])

bmi = st.number_input("Body Mass Index (BMI)")

avg_gl_level = st.number_input("Average Glucose Level")


smoking_status = st.selectbox('Smoking Status', df['Smoking_Status'].unique())
alcohol_intake = st.number_input("Alcohol Intake (units per week)", min_value=0, max_value=50, value=0)

blood_pressure = st.number_input("Blood Pressure (mmHg)")
cholesterol = st.number_input("Cholesterol Level (mg/dL)")


family_history = st.radio("Family History of Stroke", ["Yes", "No"])

mri_result = st.number_input("MRI Result")


work_type = st.selectbox('Work Type', df['Work_Type'].unique())
residence_type = st.selectbox('Residence ', df['Residence_Type'].unique())
stress_level = st.slider('Stress Level', 0, 10, 0)

physical_activity = st.number_input("Physical Activity(hrs/week)")



# Encode gender
sex = 1 if sex == "Male" else 0

# Map smoking status
value_list = [1, 2, 3, 4]
smoking_dict = {key:value for key, value in zip(df['Smoking_Status'].unique(), value_list)}


# Map working type
value_list2 = [0,1, 2, 3, 4]
work_dict = {key:value for key, value in zip(df['Work_Type'].unique(), value_list2)}

# Map Residence_Type
residence_dict = {key: idx for idx, key in enumerate(df['Residence_Type'].unique())}
residence_encoded = residence_dict.get(residence_type)


user_values = [[
    age, sex, YesNo(hypertension), YesNo(heart_disease), avg_gl_level, bmi,
    smoking_dict.get(smoking_status), alcohol_intake, blood_pressure,
    cholesterol, YesNo(family_history), mri_result,residence_encoded,stress_level,physical_activity,work_dict.get(work_type)
]]


st.markdown("---")


if st.button("🔍 Predict My Risk"):
    prob = model.predict_proba(scalar.transform(user_values))[0][1]
    percentage = round(prob * 100, 2)

    st.markdown(
        f"""
        <div class="prediction-box" style="text-align:center;">
            <h2>Predicted Stroke Risk: {percentage}%</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    # WHO-based thresholds
    if percentage < 5:
        st.success("✅ Very Low Risk of Stroke (<5%)")
        st.markdown(
            """
            **Recommended Actions:**
            - Maintain a healthy lifestyle  
            - Exercise regularly (30 mins a day)  
            - Routine health check-ups once a year  
            """
        )

    elif 5 <= percentage < 10:
        st.success("✅ Low Risk of Stroke (5–10%)")
        st.markdown(
            """
            **Recommended Actions:**
            - Keep blood pressure, glucose, and cholesterol in check  
            - Exercise and eat a balanced diet  
            - Avoid smoking and excess alcohol  
            """
        )

    elif 10 <= percentage < 20:
        st.warning("⚠️ Moderate Risk of Stroke (10–20%)")
        st.markdown(
            """
            **Recommended Actions:**
            - Consider preventive medical screening  
            - Monitor blood pressure and glucose regularly  
            - Increase physical activity  
            - Reduce alcohol and quit smoking  
            """
        )

    elif 20 <= percentage < 30:
        st.error("⚠️ High Risk of Stroke (20–30%)")
        st.markdown(
            """
            **Recommended Actions:**
            - Visit a doctor for a detailed evaluation  
            - Follow prescribed medication if advised  
            - Adopt strict lifestyle changes (diet, exercise, no smoking)  
            """
        )

    else:  # ≥30%
        st.error("🚨 Very High Risk of Stroke (≥30%)")
        st.markdown(
            """
            **Recommended Actions:**
            - Consult a doctor immediately  
            - Get a full medical check-up (BP, glucose, cholesterol)  
            - Follow treatment strictly if prescribed  
            - Avoid smoking/alcohol completely  
            - Maintain a heart-healthy diet and reduce salt intake  
            """
        )
