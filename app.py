import pickle
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import numpy as np
import random
import sklearn


def get_data():
    model,scalar=pickle.load(open('stroke_mdl.pkl','rb'))
    df=pd.read_csv('stroke_dataset.csv')
    return model,scalar,df

def YesNo(n):
    if n=='Yes':
        return 1
    else:
        return 0



model,scalar,df=get_data()
st.title("Stroke Predictor")
st.write(
    'fill in the information to predict your risk of stroke')

age=st.slider('Age',0,100,50)
sex=st.radio(label="Gender",options=["Male","Female"])
hypertension=st.radio(label="Hypertension",options=["Yes","No"])
heart_disease=st.radio(label="Heart_Disease",options=["Yes","No"])
avg_gl_level=st.slider('Average Glucose Level',0,300,value=int(df['Average_Glucose_Level'].mean()))
bmi=st.slider('Body Mass Index',0,70,value=int(df['BMI'].mean()))
smoking_status=st.selectbox('Smoking Status',df['Smoking_Status'].unique())
alcohol_intake = st.number_input("Alcohol Intake (units per week)", min_value=0, max_value=50, value=0)
blood_pressure = st.slider('Blood Pressure (mmHg)',50, 200, value=int(df['Blood_Pressure'].mean()))
cholesterol = st.slider('Cholesterol Level (mg/dL)',100, 400, value=int(df['Cholesterol'].mean()))
family_history = st.radio(label="Family History of Stroke",options=["Yes", "No"])
mri_result = st.selectbox('MRI Result',df['MRI_Result'].unique())

if sex=="Male":
    sex=1
elif sex=="Female":
    sex=0
else:
    sex=random.choice([0,1])

value_list=[1,2,3,4]
smoking_dict={key:value for key, value in zip(df['Smoking_Status'].unique(), value_list)}


user_values=[[age,sex,YesNo(hypertension),YesNo(heart_disease),avg_gl_level,bmi,smoking_dict.get(smoking_status),alcohol_intake,blood_pressure,cholesterol,YesNo(family_history),mri_result]]

if st.button('Predict'):
    # Get probability instead of just class
    prob = model.predict_proba(scalar.transform(user_values))[0][1]  # Probability of stroke
    percentage = round(prob * 100, 2)

    st.subheader(f"Predicted Stroke Risk: {percentage}%")

    # Advice based on risk level
    if percentage >= 60:
        st.error("⚠️ High Risk of Stroke")
        st.write("""
        - Please **consult a doctor immediately**  
        - Get a full medical check-up (blood pressure, glucose, cholesterol)  
        - Follow prescribed medication if any  
        - Avoid smoking and alcohol completely  
        - Maintain a healthy diet and reduce salt intake  
        """)
    elif 30 <= percentage < 60:
        st.warning("⚠️ Moderate Risk of Stroke")
        st.write("""
        - Consider visiting a doctor for preventive screening  
        - Monitor your blood pressure and glucose regularly  
        - Start moderate exercise (walking, yoga, etc.)  
        - Reduce alcohol consumption and quit smoking  
        - Improve diet: more vegetables, fruits, and whole grains  
        """)
    else:
        st.success("✅ Low Risk of Stroke")
        st.write("""
        - Maintain a **healthy lifestyle**  
        - Exercise regularly (30 mins a day)  
        - Keep blood pressure, glucose, and cholesterol in check  
        - Avoid excessive smoking/alcohol  
        - Go for routine health check-ups once a year  
        """)
