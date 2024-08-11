import pandas as pd
import streamlit as st
import joblib
from prediction import predict
from streamlit_echarts import st_echarts
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os

# Get the directory where the script is located
base_path = os.path.dirname(__file__)

# Construct the full path to the configuration file
config_path = os.path.join(base_path, 'config.yaml')

# Load configuration from YAML file
with open(config_path, 'r') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)
    
# Initialize authenticator using data from the config file
authenticator = stauth.Authenticate(
    credentials=config['credentials'],
    cookie_name=config['cookie']['name'],
    cookie_key=config['cookie']['key'],
    cookie_expiry_days=config['cookie']['expiry_days'],
)

name, authentication_status, username = authenticator.login(location='sidebar')

if authentication_status:
    authenticator.logout('Logout', 'sidebar')
    st.write(f'Welcome *{name}*')
    
    # Place the rest of your app code here that should be accessible only after login

# Continue with the rest of your application logic
    st.title("Customer Churn Prediction App")

# Load the pre-trained model and other necessary files 
model = joblib.load('churn_model.sav')
encoder = joblib.load('encoder.pkl')
feature_names = joblib.load('feature_names.pkl')
scaler = joblib.load('scaler.pkl')  # Load the scaler used during training

# Upload the dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Ensure column names are stripped of extra spaces
    st.write("Dataset Loaded Successfully!")
    st.write(df.head())

    # Ensure customerId column is treated as a string
    df['customerId'] = df['customerId'].astype(str).str.strip().str.lower()

    # Search for a customer by customerID
    customer_id = st.text_input("Enter Customer ID to Search:").strip().lower()

    if st.button('Search Customer'):
        
            # Find the customer row
        customer_data = df[df['customerId'] == customer_id]

        if not customer_data.empty:
                st.write("Customer Found:")
                st.write(customer_data)

                # Drop unnecessary columns
                columns_to_drop = ['customerId', 'DaysLastOrder', 'CountryName', 'LanguageName']
                customer_data = customer_data.drop(columns=columns_to_drop, errors='ignore')

                # Fill NaN values with zero to match training preprocessing
                customer_data.fillna(0.0, inplace=True)

                # Handle categorical columns with the same encoder used during training
                categorical_columns = customer_data.select_dtypes(include=['object']).columns
                if not categorical_columns.empty:
                    customer_encoded = pd.DataFrame(encoder.transform(customer_data[categorical_columns]), 
                                                    columns=encoder.get_feature_names_out(categorical_columns))
                    customer_data = pd.concat([customer_data.drop(categorical_columns, axis=1), customer_encoded], axis=1)

                # Align features with the training data
                missing_cols = set(feature_names) - set(customer_data.columns)
                for col in missing_cols:
                    customer_data[col] = 0  # Add missing columns with default value 0

                customer_data = customer_data[feature_names]  # Reorder columns to match the training set

                # Scale numerical features
                numerical_columns = customer_data.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
                if not numerical_columns.empty:
                    customer_data[numerical_columns] = scaler.transform(customer_data[numerical_columns])

                # Check for NaN values and handle them
                if customer_data.isna().any().any():
                    # Handle NaNs: fill NaNs for numerical columns with mean and categorical columns with mode
                    for col in customer_data.columns:
                        if customer_data[col].isnull().any():
                            if col in categorical_columns:
                                customer_data[col].fillna(customer_data[col].mode()[0], inplace=True)
                            else:
                                customer_data[col].fillna(customer_data[col].mean(), inplace=True)

                    # Check for NaN values and handle them
                if customer_data.isna().any().any():
                    st.write("Warning: The following columns still have NaN values:")
                    for col in customer_data.columns:
                        if customer_data[col].isna().any():
                            st.write(f"{col}: {customer_data[col].isna().sum()} NaNs")
                            # Additional logic to handle NaNs if needed
                else:
                    # Prediction
                    probabilities = model.predict_proba(customer_data)[:, 1]  # Assuming class 1 is the positive class
                    prediction = (probabilities > 0.5).astype(int)
                    churn_probability = probabilities[0] * 100  # Convert to percentage
                    formatted_probability = f"{churn_probability:.2f}"  # Format to two decimals
            
                    st.metric(label="Churn Probability", value=f"{churn_probability:.2f}%", delta=None)
                    st.progress(int(churn_probability))
        
                    options = {
                        "tooltip": {"formatter": "{a} <br/>{b} : {c}%"},
                        "series": [
                            {
                                "name": "Churn Probability",
                                "type": "gauge",
                                "detail": {"formatter": "{value}%"},
                                "data": [{"value": float(formatted_probability), "name": "Churn Probability"}]
                            }
                        ]
                    }
                    st_echarts(options=options, height="400px")
        else:
                    st.error("Customer ID not found in the dataset.")
    else:
                st.info("Please enter a valid Customer ID.")

elif authentication_status == False:
    st.sidebar.error('Username/password is incorrect')
elif authentication_status == None:
    st.sidebar.warning('Please enter your username and password')