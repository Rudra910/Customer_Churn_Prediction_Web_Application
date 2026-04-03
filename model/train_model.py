import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def trained_model():

    df = pd.read_csv("dataset/ecommerce_churn_100k.csv")

    gender_encoder = LabelEncoder()
    complaint_encoder = LabelEncoder()

    df["Gender_Encoded"] = gender_encoder.fit_transform(df["Gender"])
    df["Complaint_Encoded"] = complaint_encoder.fit_transform(df["Complaint_Raised"])

    feature_columns = [
        'Age', 'Gender_Encoded', 'Total_Spend', 'Average_Order_Value',
        'Purchase_Frequency', 'Last_Purchase_Days', 'Customer_Rating',
        'Complaint_Encoded', 'Return_Count'
    ]

    X = df[feature_columns]
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)


    joblib.dump(rf_model,'model/trained_model.pkl')
    joblib.dump(complaint_encoder,'model/complaint_encoder.pkl')
    joblib.dump(gender_encoder,'model/gender_encoder.pkl')
    return rf_model


train = trained_model()
