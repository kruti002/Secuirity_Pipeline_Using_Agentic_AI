import pandas as pd
import numpy as np
import ipaddress
import joblib
import torch
import sys
import os

from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'engine'))
from supervised_model import preprocess_data
from autoencoder_model import Autoencoder

def run_evaluation():
    data_path = 'data/rba-dataset.csv'
    
    # We will build a test dataset
    df_list = []
    chunk_idx = 0
    
    print("Building balanced test dataset...")
    for chunk in pd.read_csv(data_path, chunksize=1000000):
        # Skip the first chunk (training set)
        if chunk_idx == 0:
            chunk_idx += 1
            print("Skipped chunk 0 (training set)")
            continue
            
        true_rows = chunk[chunk['Is Account Takeover'] == True]
        false_rows = chunk[chunk['Is Account Takeover'] == False].sample(n=len(true_rows)*100, replace=True) 
        
        # Avoid issues if len true_rows is 0, still sample some false ones
        if len(false_rows) == 0:
            false_rows = chunk[chunk['Is Account Takeover'] == False].sample(n=5000, random_state=42)
            
        df_list.append(true_rows)
        df_list.append(false_rows)
        
        chunk_idx += 1
        
    df_raw = pd.concat(df_list, axis=0).reset_index(drop=True)
    
    print(f"Test dataset built: {df_raw.shape[0]} rows.")
    print("Labels in test dataset:")
    print(df_raw['Is Account Takeover'].value_counts())
    
    # -------------------------------------------------------------------------
    # Supervised Eval
    # -------------------------------------------------------------------------
    df_sup = preprocess_data(df_raw.copy())
    X_sup = df_sup.drop(['Is Account Takeover'], axis=1) if 'Is Account Takeover' in df_sup.columns else df_sup
    y_true = df_sup['Is Account Takeover'].astype(int)
    
    rf_model = joblib.load('models/supervised_model.joblib')
    y_prob = rf_model.predict_proba(X_sup)[:, 1]
    
    auc_sup = roc_auc_score(y_true, y_prob)
    print(f"\nSupervised ROC AUC Score: {auc_sup:.4f}")
    
    # -------------------------------------------------------------------------
    # Autoencoder Eval
    # -------------------------------------------------------------------------
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
    print(f"Autoencoder Reconstruction Loss ROC AUC Score: {ae_auc:.4f}")

if __name__ == "__main__":
    run_evaluation()
