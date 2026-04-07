import pandas as pd
import numpy as np
import ipaddress
import joblib
import torch
import sys
import os

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'engine'))
from autoencoder_model import Autoencoder

def evaluate_models(test_path='data/splits/test.csv'):
    print(f"Loading holdout test data from {test_path}...")
    
    if not os.path.exists(test_path):
        raise FileNotFoundError("Missing test.csv. Please run split_data.py first.")
        
    df_raw = pd.read_csv(test_path)
    print(f"\nHoldout dataset loaded: {df_raw.shape[0]:,} rows.")
    
    # -------------------------------------------------------------
    # 1. Evaluate Supervised Model
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("📊 Evaluating Supervised Model (Random Forest)")
    print("="*50)
    
    bundle = joblib.load('models/histgradientboosting_bundle.joblib')
    hgb_model = bundle['model']
    threshold = bundle['threshold']
    
    # Preprocess test set using the model's expected features
    from features import drop_unneeded_columns_for_ml, add_base_features
    X_sup = drop_unneeded_columns_for_ml(add_base_features(df_raw.copy()))
    if 'Is Account Takeover' in X_sup.columns:
        y_true = X_sup['Is Account Takeover']
        X_sup = X_sup.drop(columns=['Is Account Takeover'])
    else:
        y_true = np.zeros(len(X_sup))
    
    # Ensure missing categorical/numerical logic applies via pipeline
    y_prob = hgb_model.predict_proba(X_sup)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    auc = roc_auc_score(y_true, y_prob)
    print(f"\nROC AUC Score: {auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Account Takeover"], zero_division=0))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    # -------------------------------------------------------------
    # 2. Evaluate Autoencoder (Anomaly Model)
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print("🧠 Evaluating Autoencoder Model (PyTorch)")
    print("="*50)
    
    from autoencoder_model import preprocess_for_ae, Autoencoder
    ae_scaler = joblib.load('models/ae_scaler.joblib')
    scaled_data, _, _ = preprocess_for_ae(df_raw.copy(), scaler=ae_scaler, is_train=False)
    
    input_dim = scaled_data.shape[1]
    ae_model = Autoencoder(input_dim)
    ae_model.load_state_dict(torch.load('models/autoencoder.pth'))
    ae_model.eval()
    
    # Calculate Reconstruction Loss (MSE)
    with torch.no_grad():
        data_tensor = torch.FloatTensor(scaled_data)
        reconstruction = ae_model(data_tensor)
        mse_loss = torch.mean((reconstruction - data_tensor)**2, dim=1).numpy()
    
    # For an AE, we test if Higher MSE = Attack
    # We can measure AUROC using MSE as our prediction score
    ae_auc = roc_auc_score(y_true, mse_loss)
    print(f"\nReconstruction Loss ROC AUC Score: {ae_auc:.4f}")
    
    # Set an arbitrary threshold at the 95th percentile of loss to test binary metrics
    threshold = np.percentile(mse_loss, 95)
    y_pred_ae = (mse_loss > threshold).astype(int)
    
    print(f"\nClassification Report (Top 5% Highest MSE marked as Anomalies):")
    print(classification_report(y_true, y_pred_ae, target_names=["Normal", "Anomaly / ATO"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred_ae))

if __name__ == "__main__":
    evaluate_models()
