import pandas as pd
import numpy as np
import ipaddress
import joblib
import torch
import sys
import os
import json

from sklearn.metrics import roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'engine'))
from supervised_model import preprocess_data
from autoencoder_model import Autoencoder

def run_evaluation():
    data_path = 'data/rba-dataset.csv'
    
    # Load just the first 1M rows where we know there are 11 positives
    df_raw = pd.read_csv(data_path, nrows=1000000)
    
    # Supervised
    df_sup = preprocess_data(df_raw.copy())
    X_sup = df_sup.drop(['Is Account Takeover'], axis=1) if 'Is Account Takeover' in df_sup.columns else df_sup
    y_true = df_sup['Is Account Takeover'].astype(int)
    
    rf_model = joblib.load('models/supervised_model.joblib')
    y_prob = rf_model.predict_proba(X_sup)[:, 1]
    
    auc_sup = roc_auc_score(y_true, y_prob)
    
    # Autoencoder
    df_ae = df_raw.copy()
    df_ae['Login Hour'] = pd.to_datetime(df_ae['Login Timestamp']).dt.hour
    
    def ip_to_int(ip):
        try: return int(ipaddress.ip_address(ip))
        except: return 0
        
    df_ae['IP Address Int'] = df_ae['IP Address'].apply(ip_to_int)
    if 'Login Successful' in df_ae.columns:
        df_ae['Login Successful'] = df_ae['Login Successful'].astype(int)
        
    numeric_cols = ['ASN', 'Login Hour', 'IP Address Int', 'User ID', 'Login Successful']
    
    ae_scaler = joblib.load('models/ae_scaler.joblib')
    scaled_data = ae_scaler.transform(df_ae[numeric_cols])
    
    input_dim = scaled_data.shape[1]
    ae_model = Autoencoder(input_dim)
    ae_model.load_state_dict(torch.load('models/autoencoder.pth'))
    ae_model.eval()
    
    with torch.no_grad():
        data_tensor = torch.FloatTensor(scaled_data)
        reconstruction = ae_model(data_tensor)
        mse_loss = torch.mean((reconstruction - data_tensor)**2, dim=1).numpy()
    
    ae_auc = roc_auc_score(y_true, mse_loss)
    
    results = {
        "Supervised_ROC_AUC": float(auc_sup),
        "Autoencoder_ROC_AUC": float(ae_auc)
    }
    
    with open('eval_results.json', 'w') as f:
        json.dump(results, f)
        
    print("SUCCESS")

if __name__ == "__main__":
    run_evaluation()
