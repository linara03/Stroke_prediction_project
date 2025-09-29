import pickle
import pandas as pd
import numpy as np
import streamlit as st

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import datetime


# ✅ Reset session state on refresh
if "reset_done" not in st.session_state:
    st.session_state.clear()
    st.session_state["reset_done"] = True

def generate_pdf(stroke_risk_percentage, risk_level, user_inputs, file_path="stroke_report.pdf"):
    """
    Generate a PDF report for stroke prediction results.
    """
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("🏥 Stroke Risk Prediction Report", styles['Title']))
    story.append(Spacer(1, 12))

    # Date & Time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"📅 Report Generated: {current_time}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Risk results
    story.append(Paragraph(f"<b>Stroke Risk Percentage:</b> {stroke_risk_percentage:.2f}%", styles['Normal']))
    story.append(Paragraph(f"<b>Risk Category:</b> {risk_level}", styles['Normal']))
    story.append(Spacer(1, 12))

    # User Inputs
    story.append(Paragraph("<b>Patient Information:</b>", styles['Heading2']))
    for key, value in user_inputs.items():
        story.append(Paragraph(f"{key}: {value}", styles['Normal']))
    story.append(Spacer(1, 12))


    doc.build(story)
    return file_path

st.title("Stroke Riskometer")
st.markdown("A health awareness tool that helps you estimate your risk of stroke and take preventive action.")
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

# Input form
st.subheader("📋 Enter Details")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Demographics & Medical History**")
    age = st.number_input("Age", min_value=0, max_value=120)
    sex = st.selectbox("Sex", ["Select","Female", "Male"])
    hypertension = st.selectbox("Hypertension", ["Select","No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["Select","No", "Yes"])
    family_history = st.selectbox("Family History of Stroke", ["Select","No", "Yes"])

    st.markdown("**Lifestyle**")
    work_type = st.selectbox("Work Type",["Select","Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Select","Rural", "Urban"])
    smoking_status = st.selectbox("Smoking Status",["Select","Never", "Formerly", "Currently", "Unknown"])

with col2:
    st.markdown("**Clinical Measurements**")
    avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)",min_value=0.0, max_value=300.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=60.0)
    blood_pressure = st.number_input("Systolic Blood Pressure (mmHg)",min_value=0, max_value=200)
    cholesterol = st.number_input("Total Cholesterol (mg/dL)",min_value=0, max_value=400)

    st.markdown("**Activity & Stress**")
    physical_activity = st.number_input("Physical Activity (hours/week)",min_value=0.0, max_value=50.0, step=0.5)
    alcohol_intake = st.number_input("Alcohol Consumption (drinks/week)",min_value=0, max_value=30)
    stress_level = st.slider("Stress Level", min_value=0, max_value=10)

    # MRI result (if your dataset has this feature)
    mri_result = st.number_input("MRI Score (if available)",min_value=0.0, max_value=100.0)

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
            "Select": None,
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
            "Select": None,
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
        #with st.expander("🔧 Debug: View Encoded Input"):
            #st.dataframe(input_data)

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
        st.metric("Stroke Risk Score", f"{stroke_risk_percentage:.1f}%")

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
        st.metric("Stroke Risk Score", f"{stroke_risk_percentage:.1f}%")

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

        # ✅ Save results in session_state with ALL inputs
        st.session_state["stroke_risk_percentage"] = stroke_risk_percentage
        st.session_state["risk_level"] = risk_level
        st.session_state["user_inputs"] = {
            "Age": age,
            "Sex": sex,
            "Hypertension": hypertension,
            "Heart Disease": heart_disease,
            "Family History": family_history,
            "Work Type": work_type,
            "Residence Type": residence_type,
            "Smoking Status": smoking_status,
            "Average Glucose Level": avg_glucose_level,
            "BMI": bmi,
            "Blood Pressure": blood_pressure,
            "Cholesterol": cholesterol,
            "Physical Activity (hrs/week)": physical_activity,
            "Alcohol Intake (drinks/week)": alcohol_intake,
            "Stress Level": stress_level,
            "MRI Score": mri_result
        }

# PDF download button
if st.button("📄 Generate Report"):
    if "stroke_risk_percentage" not in st.session_state:
        st.error("⚠️ Please run the prediction first before generating a report.")
    else:
        pdf_path = generate_pdf(
            st.session_state["stroke_risk_percentage"],
            st.session_state["risk_level"],
            st.session_state["user_inputs"]
            )
        with open(pdf_path, "rb") as pdf_file:
            st.download_button("⬇️ Download PDF", pdf_file, file_name="stroke_report.pdf", mime="application/pdf")


# Footer
st.markdown("---")
st.caption(
    "This system is intended for health awareness and preliminary risk assessment. "
    "It does not provide a medical diagnosis. Always consult a qualified healthcare professional for"
    " medical advice and treatment decisions.")
