import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.exceptions import NotFittedError

# Load the trained model and y transformer
model_path = 'model_pipeline.joblib'
y_transformer_path = 'y_transformer.joblib'
model_pipeline = joblib.load(model_path)
y_transformer = joblib.load(y_transformer_path)

# Title and description
st.set_page_config(page_title='Insurance Charges Prediction', layout='wide')
st.title('Insurance Charges Prediction')
st.write("Predict insurance charges by entering data manually or uploading a CSV file.")

# Sidebar for navigation
st.sidebar.header('Navigation')
option = st.sidebar.radio('Select Input Method:', ['Manual Input', 'Upload CSV'])

# Function to predict
def predict_charges(data):
    try:
        # Preprocess the input data using the model pipeline
        processed_data = model_pipeline.named_steps['preprocessor'].transform(data)
        prediction = model_pipeline.named_steps['regressor'].predict(processed_data)
        original_prediction = y_transformer.inverse_transform(prediction.reshape(-1, 1))
        return original_prediction
    except NotFittedError:
        st.error('Model not properly trained. Please retrain.')
    except Exception as e:
        st.error(f'An error occurred: {e}')

# Manual input section
if option == 'Manual Input':
    st.header('Manual Data Input')
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    sex = st.selectbox('Sex', ['male', 'female'])
    smoker = st.selectbox('Smoker', ['yes', 'no'])
    children = st.number_input('Number of Children', min_value=0, max_value=5, value=1)
    region = st.selectbox('Region', ['southeast', 'southwest', 'northwest', 'northeast'])
    bmi = st.number_input('BMI', min_value=15.0, max_value=50.0, value=25.0)

    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })

    if st.button('Predict'):
        result = predict_charges(input_data)
        if result is not None:
            st.success(f'Predicted Insurance Charges: ${result[0][0]:.2f}')

# CSV upload section
elif option == 'Upload CSV':
    st.header('Upload CSV File')
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(data.head())

        # Ensure the CSV has the correct columns
        expected_columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        if not all(col in data.columns for col in expected_columns):
            st.error(f"CSV file must contain the following columns: {', '.join(expected_columns)}")
        else:
            if st.button('Predict for Uploaded Data'):
                result = predict_charges(data)
                if result is not None:
                    data['Predicted Charges'] = result
                    st.write("Predictions:")
                    st.dataframe(data)
                    st.download_button('Download Predictions', data.to_csv(index=False), file_name='predictions.csv')
