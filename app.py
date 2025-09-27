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
    result=model.predict(scalar.transform(user_values))
    risk="The stroke risk is "
    if result==1:
        risk=risk+"high"
    else:
        risk=risk+"low"
    st.subheader(risk)