import pickle
import streamlit as st
import pandas as pd
import numpy as np
import random


def get_data():
    model,scalar=pickle.load(open('stroke_mdl.pkl','rb'))
    df=pd.read_csv('stroke_dataset.csv')
    return model,scalar,df

model,scalar,df=get_data()
st.title("Stroke Predictor")
st.write(
    'fill in the information to predict your risk of stroke')