import pandas as pd
import numpy as np
import joblib
from utils.retention import get_retention_strategy, get_churn_status

def load_model():
    
    try:
        model = joblib.load('model/trained_model.pkl')
        gender_encoder = joblib.load('model/gender_encoder.pkl')
        complaint_encoder = joblib.load('model/complaint_encoder.pkl')
        return model, gender_encoder, complaint_encoder
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        print("Please run train_model.py first")
        return None, None, None        
    
def predict_churn(df):
    model, gender_encoder, complaint_encoder = load_model()

    if model is None:
        return None 

    df_pred = df.copy()
        
    # Encode categorical variables
    df_pred['Gender_Encoded'] = gender_encoder.transform(df_pred['Gender'])
    df_pred['Complaint_Encoded'] = complaint_encoder.transform(df_pred['Complaint_Raised'])


    feature_columns = [
        'Age', 'Gender_Encoded', 'Total_Spend', 'Average_Order_Value',
        'Purchase_Frequency', 'Last_Purchase_Days', 'Customer_Rating',
        'Complaint_Encoded', 'Return_Count'
    ]
    
    X = df_pred[feature_columns]    
    
    probabilities = model.predict_proba(X)[:,1]

    df_result = df.copy()
    df_result['churn_probability'] = np.round(probabilities, 3)
    df_result['churn'] = df_result['churn_probability'].apply(get_churn_status)
    df_result['retention_strategy'] = df_result['churn_probability'].apply(get_retention_strategy)
   
    return df_result

def predict_single_customer(customer_data):
    """Predict churn for a single customer"""
    df = pd.DataFrame([customer_data])
    return predict_churn(df)

