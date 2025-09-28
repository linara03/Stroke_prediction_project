import pickle
import pandas as pd
import numpy as np
import streamlit as st

st.title("🏥 Stroke Risk Prediction System")
st.markdown("---")


# Load the fixed model
@st.cache_resource
def load_model():
    try:
        with open(r"D:/Stroke_prediction_project/models/random_forest_fixed.pkl", "rb") as f:
            model_package = pickle.load(f)
            return model_package
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


model_package = load_model()

if model_package is None:
    st.error("Failed to load model. Please check the model file.")
    st.stop()

model = model_package['model']
scaler = model_package['scaler']
feature_names = model_package['feature_names']

st.sidebar.header("Model Information")
st.sidebar.info(f"Training Accuracy: {model_package.get('training_accuracy', 'N/A'):.2%}")
st.sidebar.info(f"Test Accuracy: {model_package.get('test_accuracy', 'N/A'):.2%}")

# Input form
st.subheader("📋 Enter Patient Details")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Demographics & Medical History**")
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    sex = st.selectbox("Sex", ["Female", "Male"])
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    family_history = st.selectbox("Family History of Stroke", ["No", "Yes"])

    st.markdown("**Lifestyle**")
    work_type = st.selectbox("Work Type",
                             ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Rural", "Urban"])
    smoking_status = st.selectbox("Smoking Status",
                                  ["Never", "Formerly", "Currently", "Unknown"])

with col2:
    st.markdown("**Clinical Measurements**")
    avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)",
                                        min_value=40.0, max_value=300.0, value=100.0)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    blood_pressure = st.number_input("Systolic Blood Pressure (mmHg)",
                                     min_value=80, max_value=200, value=120)
    cholesterol = st.number_input("Total Cholesterol (mg/dL)",
                                  min_value=100, max_value=400, value=200)

    st.markdown("**Activity & Stress**")
    physical_activity = st.number_input("Physical Activity (hours/week)",
                                        min_value=0.0, max_value=50.0, value=5.0, step=0.5)
    alcohol_intake = st.number_input("Alcohol Consumption (drinks/week)",
                                     min_value=0, max_value=30, value=0)
    stress_level = st.slider("Stress Level", min_value=0, max_value=10, value=5)

    # MRI result (if your dataset has this feature)
    mri_result = st.number_input("MRI Score (if available)",
                                 min_value=0.0, max_value=100.0, value=50.0)

st.markdown("---")

# Add risk indicators
st.subheader("🔍 Risk Indicators Preview")
risk_factors = []
if age > 65: risk_factors.append("Age > 65")
if hypertension == "Yes": risk_factors.append("Hypertension")
if heart_disease == "Yes": risk_factors.append("Heart Disease")
if smoking_status == "Currently": risk_factors.append("Current Smoker")
if bmi > 30: risk_factors.append("BMI > 30")
if avg_glucose_level > 140: risk_factors.append("High Glucose")
if blood_pressure > 140: risk_factors.append("High Blood Pressure")
if cholesterol > 240: risk_factors.append("High Cholesterol")
if family_history == "Yes": risk_factors.append("Family History")

if risk_factors:
    st.warning(f"⚠️ Identified Risk Factors: {', '.join(risk_factors)}")
else:
    st.success("✅ No major risk factors identified")

# Prediction button
if st.button("🔮 Predict Stroke Risk", type="primary"):
    with st.spinner("Analyzing patient data..."):

        # Encode categorical variables to match training
        sex_val = 1 if sex == "Male" else 0
        hypertension_val = 1 if hypertension == "Yes" else 0
        heart_disease_val = 1 if heart_disease == "Yes" else 0

        # Encode work type (must match your preprocessing)
        work_type_map = {
            "Private": 0,
            "Self-employed": 1,
            "Govt_job": 2,
            "Children": 3,
            "Never_worked": 4
        }
        work_type_val = work_type_map[work_type]

        residence_type_val = 1 if residence_type == "Urban" else 0

        # Encode smoking status
        smoking_map = {
            "Never": 0,
            "Formerly": 1,
            "Currently": 2,
            "Unknown": 3
        }
        smoking_status_val = smoking_map[smoking_status]

        family_history_val = 1 if family_history == "Yes" else 0

        # Create input dataframe with exact feature order
        # IMPORTANT: This must match the exact order of features during training
        input_data = pd.DataFrame([[
            age,
            sex_val,
            hypertension_val,
            heart_disease_val,
            work_type_val,
            residence_type_val,
            avg_glucose_level,
            bmi,
            smoking_status_val,
            physical_activity,
            alcohol_intake,
            stress_level,
            blood_pressure,
            cholesterol,
            family_history_val,
            mri_result
        ]], columns=feature_names)

        # Display the input for debugging
        with st.expander("🔧 Debug: View Encoded Input"):
            st.dataframe(input_data)

        # Scale the input
        input_scaled = scaler.transform(input_data)

        # Get prediction and probabilities
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]

        # Extract probabilities
        no_stroke_prob = probabilities[0]
        stroke_prob = probabilities[1]

        # Convert to percentage
        stroke_risk_percentage = stroke_prob * 100

    # Display results
    st.markdown("---")
    st.subheader("📊 Prediction Results")

    # Main result with visual indicator
    if stroke_risk_percentage > 50:
        st.error(f"⚠️ **HIGH RISK OF STROKE**")
        st.metric("Stroke Risk Score", f"{stroke_risk_percentage:.1f}%",
                  delta=f"+{stroke_risk_percentage - 50:.1f}% above threshold")

        st.markdown("""
        ### 🚨 Immediate Recommendations:
        - **Consult a healthcare professional immediately**
        - Consider comprehensive cardiovascular evaluation
        - Review and optimize current medications
        - Implement lifestyle modifications urgently
        """)

    elif stroke_risk_percentage > 30:
        st.warning(f"⚡ **MODERATE RISK OF STROKE**")
        st.metric("Stroke Risk Score", f"{stroke_risk_percentage:.1f}%")

        st.markdown("""
        ### ⚡ Recommendations:
        - Schedule a check-up with your healthcare provider
        - Monitor blood pressure and glucose regularly
        - Consider lifestyle modifications
        - Review family history with doctor
        """)

    else:
        st.success(f"✅ **LOW RISK OF STROKE**")
        st.metric("Stroke Risk Score", f"{stroke_risk_percentage:.1f}%",
                  delta=f"-{50 - stroke_risk_percentage:.1f}% below threshold")

        st.markdown("""
        ### 💚 Recommendations:
        - Continue maintaining healthy lifestyle
        - Regular health check-ups
        - Stay physically active
        - Monitor any changes in health status
        """)

    # Probability breakdown
    st.markdown("---")
    st.subheader("📈 Risk Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("No Stroke Probability", f"{no_stroke_prob:.1%}")

    with col2:
        st.metric("Stroke Probability", f"{stroke_prob:.1%}")

    with col3:
        # Risk category
        if stroke_risk_percentage > 70:
            risk_level = "Very High"
            color = "🔴"
        elif stroke_risk_percentage > 50:
            risk_level = "High"
            color = "🟠"
        elif stroke_risk_percentage > 30:
            risk_level = "Moderate"
            color = "🟡"
        else:
            risk_level = "Low"
            color = "🟢"

        st.metric("Risk Level", f"{color} {risk_level}")

    # Progress bar for risk
    st.progress(min(stroke_prob, 1.0))

    # Additional information
    with st.expander("ℹ️ About This Prediction"):
        st.markdown("""
        This prediction is based on a machine learning model trained on historical patient data.

        **Important Notes:**
        - This is a screening tool, not a diagnostic test
        - Always consult with healthcare professionals for medical decisions
        - The model considers multiple risk factors simultaneously
        - Individual risk may vary based on factors not captured in this model

        **Model Performance:**
        - The model has been validated on test data
        - Results should be interpreted by medical professionals
        - Regular updates ensure accuracy with latest medical knowledge
        """)

# Footer
st.markdown("---")
st.caption(
    "⚕️ This tool is for educational and screening purposes only. Always consult healthcare professionals for medical advice.")
st.caption(f"Model Version: {model_package.get('version', '1.0')} | Features: {len(feature_names)}")