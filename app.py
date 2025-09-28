import pickle
import pandas as pd
import streamlit as st

st.title("Stroke Prediction System")

# Load the best model
with open("models/random_forest.pkl", "rb") as f:  # <-- Final chosen model
    model_data = pickle.load(f)
    model = model_data[0]  # RandomForest model
    feature_names = model_data[1]  # Feature names list
    # Note: This model doesn't have scaler or encoders

# Input form
st.subheader("Enter Details")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    sex = st.selectbox("Sex", ["Female", "Male"])
    hypertension = st.selectbox("Do you have Hypertension?", ["No", "Yes"])
    heart_disease = st.selectbox("Do you have Heart Disease?", ["No", "Yes"])
    work_type = st.selectbox("Work Type", ["Children", "Govt_job", "Never_worked", "Private", "Self-employed"])
    residence_type = st.selectbox("Residence Type", ["Rural", "Urban"])
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=40.0, max_value=300.0, value=100.0)

with col2:
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    smoking_status = st.selectbox("Smoking Status", ["Currently", "Formerly", "Never", "Unknown"])
    physical_activity = st.number_input("Hours of Physical Activity per Week", min_value=0.0, max_value=50.0, value=5.0)
    alcohol_intake = st.number_input("Times consuming Alcohol per Week", min_value=0, max_value=14, value=0)
    stress_level = st.slider("Stress Level (0-10 scale)", min_value=0, max_value=10, value=5)
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
    cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=300, value=200)
    family_history = st.selectbox("Family History of Stroke", ["No", "Yes"])
    mri_result = st.number_input("MRI Result", min_value=0.0, max_value=100.0, value=50.0)

# Add prediction button
if st.button("Predict Stroke Risk", type="primary"):
    # Convert Yes/No to 1/0 for binary features
    hypertension_val = 1 if hypertension == "Yes" else 0
    heart_disease_val = 1 if heart_disease == "Yes" else 0
    
    # Convert categorical variables to numerical values
    sex_val = 1 if sex == "Male" else 0
    work_type_map = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "Children": 3, "Never_worked": 4}
    work_type_val = work_type_map[work_type]
    residence_type_val = 1 if residence_type == "Urban" else 0
    smoking_map = {"Never": 0, "Formerly": 1, "Currently": 2, "Unknown": 3}
    smoking_status_val = smoking_map[smoking_status]
    family_history_val = 1 if family_history == "Yes" else 0
    
    # Convert to dataframe with proper column names (16 features, no ever_married)
    input_data = pd.DataFrame([[
        age, sex_val, hypertension_val, heart_disease_val, work_type_val, residence_type_val,
        avg_glucose_level, bmi, smoking_status_val, physical_activity, alcohol_intake,
        stress_level, blood_pressure, cholesterol, family_history_val, mri_result
    ]], columns=feature_names)
    
    # Make prediction directly (no scaling needed for Random Forest)
    prediction = model.predict(input_data)[0]
    
    st.subheader("🎯 Prediction Result")
    if prediction == 1:
        st.error("⚠️ The patient is at risk of Stroke.")
        st.warning("Please consult with a healthcare professional for further evaluation.")
    else:
        st.success("✅ The patient is not at risk of Stroke.")
        st.info("Continue maintaining a healthy lifestyle!")
