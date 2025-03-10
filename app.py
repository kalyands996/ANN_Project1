import streamlit as st 
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
from tensorflow.keras.models import load_model


#load the models 
model = load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
with open('one_hot_geo.pkl', 'rb') as file:
    one_hot_geo = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

#Streamlit
st.title('Customer Churn Prediction')


#User input
geography = st.selectbox('Geography', one_hot_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cc = st.selectbox('Has Credit Card',[0,1])
active_mem = st.selectbox('Is active member', [0,1])


#prepare the input data 
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard': [has_cc],
    'IsActiveMember' : [active_mem],
    'EstimatedSalary' : [estimated_salary]
})

# onehot encoder for geography
geo_encoded = one_hot_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns= one_hot_geo.get_feature_names_out())

#combine encoded columns with input data 
input_data = pd.concat([input_data.reset_index(drop= True), geo_encoded_df], axis=1)

#Scale the input data
input_data_scaled = scaler.transform(input_data)

#Churn prediction
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

st.write(f'Churn Probability: {prediction_prob:.2f}')


if prediction_prob > 0.5:
    st.write('The Customer is likely to Churn.')
else:
    st.write('The Customer is unlikely to Churn')


















