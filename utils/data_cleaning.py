import pandas as pd
import numpy as np

def validate_columns(df):
    """Validate if all required columns are present"""
    required_columns = [
        'Customer_ID', 'Age', 'Gender', 'Total_Spend', 'Average_Order_Value',
        'Purchase_Frequency', 'Last_Purchase_Days', 'Customer_Rating',
        'Complaint_Raised', 'Return_Count'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True

def clean_data(df):
    """Clean and preprocess the data"""
    # Make a copy to avoid warnings
    df_clean = df.copy()
    
    # Handle missing values
    df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
    df_clean['Total_Spend'] = df_clean['Total_Spend'].fillna(df_clean['Total_Spend'].median())
    df_clean['Average_Order_Value'] = df_clean['Average_Order_Value'].fillna(df_clean['Average_Order_Value'].median())
    df_clean['Purchase_Frequency'] = df_clean['Purchase_Frequency'].fillna(df_clean['Purchase_Frequency'].median())
    df_clean['Last_Purchase_Days'] = df_clean['Last_Purchase_Days'].fillna(df_clean['Last_Purchase_Days'].median())
    df_clean['Customer_Rating'] = df_clean['Customer_Rating'].fillna(df_clean['Customer_Rating'].median())
    df_clean['Return_Count'] = df_clean['Return_Count'].fillna(0)
    df_clean['Complaint_Raised'] = df_clean['Complaint_Raised'].fillna('No')
    
    # Convert data types
    df_clean['Age'] = df_clean['Age'].astype(int)
    df_clean['Total_Spend'] = df_clean['Total_Spend'].astype(float)
    df_clean['Average_Order_Value'] = df_clean['Average_Order_Value'].astype(float)
    df_clean['Purchase_Frequency'] = df_clean['Purchase_Frequency'].astype(int)
    df_clean['Last_Purchase_Days'] = df_clean['Last_Purchase_Days'].astype(int)
    df_clean['Customer_Rating'] = df_clean['Customer_Rating'].astype(float)
    df_clean['Return_Count'] = df_clean['Return_Count'].astype(int)
    
    # Validate ranges
    df_clean['Age'] = df_clean['Age'].clip(18, 100)
    df_clean['Customer_Rating'] = df_clean['Customer_Rating'].clip(1, 5)
    df_clean['Return_Count'] = df_clean['Return_Count'].clip(0, 20)
    
    # Standardize text
    df_clean['Gender'] = df_clean['Gender'].str.capitalize()
    df_clean['Complaint_Raised'] = df_clean['Complaint_Raised'].str.capitalize()
    
    return df_clean