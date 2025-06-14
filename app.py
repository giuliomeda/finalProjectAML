import sys
import streamlit as st

import joblib

# definizione della nn usata 
import numpy as np

# carico il modello e la pipeline
model = joblib.load("model/xgboost_model.pkl")
pipeline_user = joblib.load("model/pipeline_for_user_input.pkl")

# Interfaccia Streamlit
st.title("🧠 Stroke Prediction Project")
st.markdown("""
Applicazione di predizione del rischio di ictus.<br>
Inserisci i tuoi dati nei campi sottostanti e premi **Predict** per ottenere una valutazione.
""", unsafe_allow_html=True)

st.image("https://storage.googleapis.com/kaggle-datasets-images/1120859/1882037/04da2fb7763e553bdf251d5adf6f88d9/dataset-cover.jpg?t=2021-01-26-19-57-05", caption="Stroke Prediction", use_container_width=True)

# Input utente
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 0, 120, 30)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    ever_married = st.selectbox("Ever Married", ["No", "Yes"])
with col2:
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input("Average Glucose Level", 0.0, 300.0, 100.0)
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])


import pandas as pd

if st.button("Predict"):
    # 1. Crea un DataFrame con i dati utente
    input_dict = {
        "gender": [gender],
        "age": [age],
        "hypertension": [1 if hypertension == "Yes" else 0],
        "heart_disease": [1 if heart_disease == "Yes" else 0],
        "ever_married": [ever_married],
        "work_type": [work_type],
        "Residence_type": [residence_type],
        "avg_glucose_level": [avg_glucose_level],
        "bmi": [bmi],
        "smoking_status": [smoking_status]
    }
    input_df = pd.DataFrame(input_dict)

    # 2. Applica la pipeline (solo transform))
    X_processed = pipeline_user.transform(input_df)

    output = model.predict(X_processed)

    #st.write(f"Predizione stroke: {'SI' if output else 'NO'}")
    if output:
        st.error("⚠️ Predizione stroke: SI")
    else:
        st.success("✅ Predizione stroke: NO")


st.markdown("---")
st.caption("Progetto Finale Applied Machine Learning -  Giulio Meda, Matteo Veronese, Pietro Zorzin")

with st.sidebar:
    st.header("Info")
    st.write("Progetto Finale Applied Machine Learning")
    st.write("Autori: Giulio Meda, Matteo Veronese, Pietro Zorzin")
with st.expander("ℹ️ Come funziona l'app?"):
    st.write("""
        Inserisci i tuoi dati e premi Predict per stimare il rischio di ictus.
        I dati non vengono salvati.
    """)