from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
#for accountfraud

import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

app = Flask(__name__)

# Sample data for demonstration (replace with your actual data)
def load_sample_data():
    # Transaction Fraud Data
    transaction_data = pd.DataFrame({
        'Customer_ID': range(1, 1001),
        'Gender': ['M', 'F'] * 500,
        'Age': np.random.randint(18, 80, 1000),
        'State': np.random.choice(['NY', 'CA', 'TX', 'FL', 'IL'], 1000),
        'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami'], 1000),
        'Bank_Branch': np.random.choice(['Branch A', 'Branch B', 'Branch C'], 1000),
        'Account_Type': np.random.choice(['Savings', 'Checking', 'Credit'], 1000),
        'Transaction_ID': [f'TXN{i:04d}' for i in range(1, 1001)],
        'Transaction_Date': pd.date_range(start='2024-01-01', periods=1000, freq='H'),
        'Transaction_Time': pd.date_range(start='2024-01-01', periods=1000, freq='H').time,
        'Transaction_Amount': np.random.uniform(10, 10000, 1000),
        'Merchant_ID': [f'MER{i:04d}' for i in range(1, 1001)],
        'Transaction_Type': np.random.choice(['Online', 'In-Person', 'ATM'], 1000),
        'Merchant_Category': np.random.choice(['Retail', 'Food', 'Entertainment', 'Travel'], 1000),
        'Account_Balance': np.random.uniform(1000, 50000, 1000),
        'Transaction_Device': np.random.choice(['Mobile', 'Desktop', 'Tablet'], 1000),
        'Transaction_Location': np.random.choice(['Domestic', 'International'], 1000),
        'Device_Type': np.random.choice(['iOS', 'Android', 'Windows', 'Mac'], 1000),
        'Is_Fraud': np.random.choice([0, 1], 1000, p=[0.95, 0.05]),
        'Transaction_Currency': np.random.choice(['USD', 'EUR', 'GBP'], 1000),
        'Transaction_Description': [f'Transaction {i}' for i in range(1, 1001)]
    })
    return transaction_data

def load_account_sample_data():
    # Account Fraud Data

    account_data = pd.read_csv("datasets\Base.csv")
  
    return account_data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transaction_fraud')
def transaction_fraud():
    return render_template('transaction_fraud.html')

@app.route('/account_fraud')
def account_fraud():
    return render_template('account_fraud.html')

@app.route('/api/transaction_stats')
def transaction_stats():
    data = load_sample_data()
    
    # Create various visualizations
    fraud_by_type = data.groupby('Transaction_Type')['Is_Fraud'].mean().reset_index()
    fraud_by_category = data.groupby('Merchant_Category')['Is_Fraud'].mean().reset_index()
    fraud_by_device = data.groupby('Transaction_Device')['Is_Fraud'].mean().reset_index()
    
    # Create plots
    type_fig = px.bar(fraud_by_type, x='Transaction_Type', y='Is_Fraud',
                      title='Fraud Rate by Transaction Type')
    category_fig = px.pie(fraud_by_category, values='Is_Fraud', names='Merchant_Category',
                         title='Fraud Distribution by Merchant Category')
    device_fig = px.bar(fraud_by_device, x='Transaction_Device', y='Is_Fraud',
                       title='Fraud Rate by Transaction Device')
    
    return jsonify({
        'type_chart': type_fig.to_json(),
        'category_chart': category_fig.to_json(),
        'device_chart': device_fig.to_json()
    })

@app.route('/api/detect_transaction_fraud', methods=['POST'])
def detect_transaction_fraud():
    data = request.json
    # Add your fraud detection logic here
    # This is a simple example - replace with your actual model
    risk_score = np.random.random()
    is_fraud = risk_score > 0.7
    
    return jsonify({
        'risk_score': float(risk_score),
        'is_fraud': bool(is_fraud),
        'message': 'High risk transaction detected!' if is_fraud else 'Transaction appears normal'
    })

@app.route('/api/account_stats')
def account_stats():
    data = load_account_sample_data()
    
    # Create various visualizations
    fraud_by_source = data.groupby('source')['fraud_bool'].mean().reset_index()
    fraud_by_payment = data.groupby('payment_type')['fraud_bool'].mean().reset_index()
    fraud_by_os = data.groupby('device_os')['fraud_bool'].mean().reset_index()
    
    # Additional fraud patterns
    fraud_by_month = data.groupby('month')['fraud_bool'].mean().reset_index()
    risk_by_session = px.scatter(data, x='session_length_in_minutes', y='device_fraud_count',
                                color='fraud_bool', title='Risk Pattern by Session Length')
    feature_columns = ['income', 'name_email_similarity', 'prev_address_months_count', 
                       'current_address_months_count', 'customer_age', 'days_since_request', 
                       'intended_balcon_amount', 'zip_count_4w', 'proposed_credit_limit', 
                       'foreign_request', 'session_length_in_minutes', 'device_distinct_emails_8w', 
                       'device_fraud_count']
    
    X = data[feature_columns]
    y = data['fraud_bool']
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=2)  # Reduce to 2D for visualization
    X_pca = pca.fit_transform(X_scaled)

    # Train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_pca, y)
    
    # Convert PCA result to DataFrame for visualization
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['fraud_bool'] = y.astype(str)  # Convert to string for color mapping
    # Scatter plot for PCA with fraud labels
    knn_pca_fig = px.scatter(pca_df, x='PC1', y='PC2', color='fraud_bool',
                            title='K-Neighbors Fraud Detection with PCA',
                            labels={'fraud_bool': 'Fraud Status'},
                            opacity=0.8)
    
    # Create plots
    source_fig = px.bar(fraud_by_source, x='source', y='fraud_bool',
                       title='Fraud Rate by Source')
    payment_fig = px.pie(fraud_by_payment, values='fraud_bool', names='payment_type',
                        title='Fraud Distribution by Payment Type')
    os_fig = px.bar(fraud_by_os, x='device_os', y='fraud_bool',
                    title='Fraud Rate by Device OS')
    
  

    
    return jsonify({
        'source_chart': source_fig.to_json(),
        'payment_chart': payment_fig.to_json(),
        'os_chart': os_fig.to_json(),
        'knn_pca_chart': knn_pca_fig.to_json(),
        'session_chart': risk_by_session.to_json()
    })

@app.route('/api/detect_account_fraud', methods=['POST'])
def detect_account_fraud():
    data = request.json
    income = data.get("income")  
    NameEmsim = data.get("name_email_similarity")
    customerage = data.get("customer_age")
    creditrisk = data.get("cibil_score")
    creditlimit = data.get("credit_limit")
    devicefraud = data.get("device_fraud_count")

    

    df = pd.read_csv("datasets\Base.csv")
    # Load the saved scaler
    loaded_scaler = joblib.load("models\scaler.pkl")

    # Load the saved KNN model
    loaded_knn_model = joblib.load("models\knn_model.pkl")

    # New Sample Data
    new_sample = np.array([[income, NameEmsim, customerage, creditrisk, 0,devicefraud, creditlimit]])

    # Select features used for KNN
    knn_features = ["income", "name_email_similarity", "customer_age", "credit_risk_score",
                    "bank_months_count", "device_fraud_count", "proposed_credit_limit"]
    new_sample_knn = pd.DataFrame(new_sample, columns=knn_features)

    # Standardize the new sample using the loaded scaler
    new_sample_scaled = loaded_scaler.transform(new_sample_knn)
    features = [
    "income", "name_email_similarity", "customer_age", "credit_risk_score",
    "bank_months_count", "device_fraud_count", "proposed_credit_limit"
    ]
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Find 100 nearest neighbors using the loaded KNN model
    k_neighbors = 100
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree').fit(X_scaled)
    distances, indices = nbrs.kneighbors(new_sample_scaled)

    # Compute fraud risk probability
    fraud_counts = df["fraud_bool"].iloc[indices[0]].sum()
    fraud_risk_probability = fraud_counts / k_neighbors
   
    if (fraud_risk_probability >= 0.2):
        is_fraud = 1
    else:
        is_fraud = 0
        
    
    return jsonify({
        'risk_score': float(fraud_risk_probability),
        'is_fraud': int(is_fraud),
        'message': 'High risk account activity detected!' if is_fraud else 'Account activity appears normal'
    })


if __name__ == '__main__':
    app.run(debug=True) 