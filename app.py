import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load your trained model
model = pickle.load(open('heart_attack_model.pkl', 'rb'))

# Streamlit app title
st.title("Heart Attack Prediction")

# Define input fields
def user_input_features():
    age = st.number_input("Age", 0, 100, 50)
    sex = st.selectbox("Sex", [0, 1])  # 0 = Female, 1 = Male
    cp = st.selectbox("Chest Pain Type (CP)", [0, 1, 2, 3])
    trtbps = st.number_input("Resting Blood Pressure (TRTBPS)", 0, 200, 120)
    chol = st.number_input("Cholesterol (CHOL)", 0, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (FBS)", [0, 1])
    restecg = st.selectbox("Resting Electrocardiographic Results (RESTECG)", [0, 1, 2])
    thalachh = st.number_input("Maximum Heart Rate Achieved (THALACHH)", 0, 250, 150)
    exng = st.selectbox("Exercise Induced Angina (EXNG)", [0, 1])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    slp = st.selectbox("Slope of the Peak Exercise ST Segment (SLP)", [0, 1, 2])
    caa = st.selectbox("Number of Major Vessels (0-3) (CAA)", [0, 1, 2, 3, 4])
    thall = st.selectbox("Thalassemia (THALL)", [0, 1, 2, 3])

    # Store inputs in a dataframe
    data = {'age': age,
            'sex': sex,
            'cp': cp,
            'trtbps': trtbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalachh': thalachh,
            'exng': exng,
            'oldpeak': oldpeak,
            'slp': slp,
            'caa': caa,
            'thall': thall}
    
    features = pd.DataFrame(data, index=[0])
    
    # One-hot encode categorical variables to match the training data
    features_encoded = pd.get_dummies(features, columns=['cp', 'restecg', 'slp', 'caa', 'thall'])

    # Ensure the columns match the model training set
    all_columns = ['age', 'sex', 'trtbps', 'chol', 'thalachh', 'exng', 'oldpeak', 'fbs',
                   'cp_0', 'cp_1', 'cp_2', 'cp_3',
                   'restecg_0', 'restecg_1', 'restecg_2',
                   'slp_0', 'slp_1', 'slp_2',
                   'caa_0', 'caa_1', 'caa_2', 'caa_3', 'caa_4',
                   'thall_0', 'thall_1', 'thall_2', 'thall_3']
    
    # Add missing columns with default value 0
    for col in all_columns:
        if col not in features_encoded.columns:
            features_encoded[col] = 0
    
    # Reorder the columns to match the training data
    features_encoded = features_encoded[all_columns]
    
    return features_encoded

input_df = user_input_features()

# Display user inputs
st.subheader("User Input Features")
st.write(input_df)

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.warning("The model predicts that you are at risk of a heart attack.")
    else:
        st.success("The model predicts that you are not at risk of a heart attack.")

# Display Prediction Probability
proba = model.predict_proba(input_df)
st.subheader("Prediction Probability")
st.write(proba)
