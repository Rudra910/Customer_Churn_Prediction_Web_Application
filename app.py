from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
import pandas as pd
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from sklearn.ensemble import RandomForestClassifier as rf

from model.predict import predict_churn, predict_single_customer
from utils.db_connection import create_database_and_tables, insert_input_data, insert_prediction_results, get_all_predictions
from utils.data_cleaning import validate_columns, clean_data
from utils.retention import get_retention_strategy

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'csv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Home page with upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle CSV upload and predictions"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read and validate CSV
            df = pd.read_csv(filepath)
            validate_columns(df)
            df_cleaned = clean_data(df)
            
            # Initialize database and insert data
            create_database_and_tables()
            insert_input_data(df_cleaned)
            
            # Make predictions
            df_predictions = predict_churn(df_cleaned)
            
            if df_predictions is None:
                return jsonify({'error': 'Model not loaded properly'}), 500
            
            # Store predictions
            insert_prediction_results(df_predictions)
            
            # Save to CSV for download
            df_predictions.to_csv("outputs/predictions.csv", index=False)
            
            # Generate summary
            summary = {
                'total_customers': len(df_predictions),
                'high_risk': len(df_predictions[df_predictions['churn_probability'] >= 0.75]),
                'medium_risk': len(df_predictions[(df_predictions['churn_probability'] >= 0.50) & (df_predictions['churn_probability'] < 0.75)]),
                'low_risk': len(df_predictions[df_predictions['churn_probability'] < 0.50]),
                'churn_yes': len(df_predictions[df_predictions['churn'] == 'Yes']),
                'churn_no': len(df_predictions[df_predictions['churn'] == 'No'])
            }
            
            return jsonify({
                'success': True,
                'message': 'File processed successfully',
                'summary': summary,
                'download_url': '/download'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/dashboard')
def dashboard():
    """Dashboard page with matplotlib charts"""
    try:
        # Load predictions
        df = get_all_predictions()
        
        if df.empty:
            return render_template('dashboard.html', charts={}, has_data=False)
        
        # Create risk categories
        def get_risk_category(prob):
            if prob >= 0.75:
                return 'High Risk'
            elif prob >= 0.50:
                return 'Medium Risk'
            else:
                return 'Low Risk'
        
        df['risk_category'] = df['churn_probability'].apply(get_risk_category)
        
        # Generate all charts
        charts = {}
        
        # Chart 1: Churn Distribution (Pie Chart)
        plt.figure(figsize=(5, 5))
        churn_counts = df['churn'].value_counts()
        colors = ["#fa3131", "#0066d2"] if 'Yes' in churn_counts.index else ['#0066d2', '#fa3131']
        plt.pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title('Churn Distribution', fontsize=12, fontweight='bold')
        plt.savefig("static/Churn_Distribution.png", dpi=150)
        plt.close()
        
        # Chart 2: Risk Distribution (Bar Chart)
        plt.figure(figsize=(5.2, 5))
        risk_counts = df['risk_category'].value_counts()
        risk_colors = {'High Risk': '#fa3131', 'Medium Risk': '#ffc107', 'Low Risk': '#0066d2'}
        colors = [risk_colors.get(x, '#6c757d') for x in risk_counts.index]
        
        plt.bar(risk_counts.index, risk_counts.values, color=colors)
        plt.title('Customer Risk Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Risk Category', fontsize=12)
        plt.ylabel('Number of Customers', fontsize=10)
        
        # Add value labels on bars
        for i, v in enumerate(risk_counts.values):
            plt.text(i, v + 5, str(v), ha='center', fontweight='bold')
        
        plt.savefig("static/Risk_Distribution.png", dpi=150)
        plt.close()
        
        # Chart 3: Churn_by_Gender (Grouped Bar)
        plt.figure(figsize=(5.2, 5))
        gender_churn = pd.crosstab(df['Gender'], df['churn'])
        
        x = range(len(gender_churn.index))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(5.2, 5))
        ax.bar([i - width/2 for i in x], gender_churn.get('No', [0, 0]), width, 
               label='Not Churned', color='#0066d2')
        ax.bar([i + width/2 for i in x], gender_churn.get('Yes', [0, 0]), width, 
               label='Churned', color='#fa3131')
        
        ax.set_xlabel('Gender', fontsize=12)
        ax.set_ylabel('Number of Customers', fontsize=10)
        ax.set_title('Churn by Gender', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(gender_churn.index)
        ax.legend()
        
        plt.savefig("static/Churn_by_Gender.png", dpi=150)
        plt.close()
        
        # Chart 4: Feature Importance (Horizontal Bar Chart)
        try:
            # Prepare features (same as model input)
            feature_cols = ['Age', 'Gender', 'Total_Spend', 'Average_Order_Value',
                            'Purchase_Frequency', 'Last_Purchase_Days',
                            'Customer_Rating', 'Complaint_Raised', 'Return_Count']

            df_model = df.copy()

            # Encode categorical columns
            df_model['Gender'] = df_model['Gender'].map({'Male': 1, 'Female': 0})
            df_model['Complaint_Raised'] = df_model['Complaint_Raised'].map({'Yes': 1, 'No': 0})

            X = df_model[feature_cols]
            y = df_model['churn']

            # Train Random Forest (only for importance)
            rf_model = rf(n_estimators=100, random_state=42)
            rf_model.fit(X, y)

            # Create feature importance dataframe
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=True)

            # Plot graph
            plt.figure(figsize=(5, 5))
            plt.barh(feature_importance['feature'], feature_importance['importance'],
                     color='skyblue')

            plt.xlabel('Importance')
            plt.ylabel('Features')
            # plt.title('Feature Importance')
            plt.title(' Feature Importance', fontsize=12, fontweight='bold')


            plt.tight_layout()
            plt.savefig("static/feature_importance.png")
            plt.close()

        except Exception as e:
            print("Feature importance error:", e)
        
        # Chart 5: Last Purchase Days vs Churn
        plt.figure(figsize=(5, 5))
        last_purchase_churn = df.groupby('churn')['Last_Purchase_Days'].mean()
        
        plt.bar(['Not Churned', 'Churned'], 
                [last_purchase_churn.get('No', 0), last_purchase_churn.get('Yes', 0)],
                color=['#0066d2', '#fa3131'])
        plt.xlabel('Churn Status', fontsize=12)
        plt.ylabel('Average Days Since Last Purchase', fontsize=12)
        plt.title('Last Purchase Days by Churn Status', fontsize=12, fontweight='bold')
        
        # Add value labels
        for i, v in enumerate([last_purchase_churn.get('No', 0), last_purchase_churn.get('Yes', 0)]):
            plt.text(i, v + 1, f'{v:.1f} days', ha='center', fontweight='bold')
        
        plt.savefig("static/Last_Purchase_Days_vs_Churn.png", dpi=150)
        plt.close()
        
        # Chart 6: Complaint vs Churn
        plt.figure(figsize=(5.2, 5))
        complaint_churn = pd.crosstab(df['Complaint_Raised'], df['churn'])
        
        x = range(len(complaint_churn.index))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.bar([i - width/2 for i in x], complaint_churn.get('No', [0, 0]), width,
               label='Not Churned', color='#0066d2')
        ax.bar([i + width/2 for i in x], complaint_churn.get('Yes', [0, 0]), width,
               label='Churned', color='#fa3131')
        
        ax.set_xlabel('Complaint Raised', fontsize=12)
        ax.set_ylabel('Number of Customers', fontsize=10)
        ax.set_title('Churn by Complaint Status', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(complaint_churn.index)
        ax.legend()
        
        plt.savefig("static/Complaint_vs_Churn.png", dpi=150)
        plt.close()
        
        # Get top 10 risky customers for table
        top_risky = df.nlargest(10, 'churn_probability')[['Customer_ID', 'Age', 'Gender', 
                                                           'Total_Spend', 'churn_probability', 
                                                           'risk_category','retention_strategy']]
        
        df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})
        total_customers = len(df)
        churn_rate = round(df['churn'].mean() * 100, 2)
        retained_customers = len(df[df['churn'] == 0])
        # high_risk = len(df[df['risk_category'] == 'High'])

        return render_template('dashboard.html',total_customers=total_customers,charts=charts,
        churn_rate=churn_rate,
        retained_customers=retained_customers,
        has_data=True,top_risky=top_risky.to_dict('records'))
        
    except Exception as e:
        print(f"Dashboard error: {e}")
        return render_template('dashboard.html', charts={}, has_data=False, error=str(e))

@app.route('/individual')
def individual():
    """Individual prediction page"""
    return render_template('individual.html')

@app.route('/predict-individual', methods=['POST'])
def predict_individual():
    """Handle individual customer prediction"""
    try:
        # Get form data with correct field names
        customer_data = {
            'Customer_ID': request.form.get('customer_id', 'CUST-MANUAL'),
            'Age': int(request.form.get('age')),
            'Gender': request.form.get('gender'),  # This should match your encoding
            'Total_Spend': float(request.form.get('total_spend')),
            'Average_Order_Value': float(request.form.get('avg_order_value')),
            'Purchase_Frequency': int(request.form.get('purchase_frequency')),
            'Last_Purchase_Days': int(request.form.get('last_purchase_days')),
            'Customer_Rating': float(request.form.get('customer_rating')),
            'Complaint_Raised': request.form.get('complaint_raised'),  # This should match your encoding
            'Return_Count': int(request.form.get('return_count'))
        }
        
        # Make prediction
        df_result = predict_single_customer(customer_data)
        
        if df_result is None:
            return jsonify({'error': 'Model not loaded properly'}), 500
        
        # Get the prediction result
        result = df_result.iloc[0].to_dict()
        
        prob = result['churn_probability']
        if prob >= 0.75:
            risk_level = "High Risk"
            risk_color = "#fa3131"
            risk_text = "High Risk of Churn"
        elif prob >= 0.50:
            risk_level = "Medium Risk"
            risk_color = "#ffc107"
            risk_text = "Medium Risk of Churn"
        else:
            risk_level = "Low Risk"
            risk_color = "#00d200"
            risk_text = "Low Risk of Churn"
        
        # Return JSON response
        return jsonify({
            'success': True,
            'probability': f"{prob*100:.2f}",
            'risk_level': risk_level,
            'risk_color': risk_color,
            'risk_text': risk_text,
            'churn_status': result['churn'],
            'retention_strategy': result['retention_strategy']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/download')
def download():
    """Download predictions CSV"""
    try:
        return send_from_directory(app.root_path,"outputs/predictions.csv ")
    except Exception as e:
        return jsonify({'error': 'File not found'}), 404


if __name__ == '__main__':
    create_database_and_tables()
    app.run(debug=True, port=5001)