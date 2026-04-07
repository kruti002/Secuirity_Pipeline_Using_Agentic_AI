import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

# Ensure local directory imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import add_base_features, drop_unneeded_columns_for_ml

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, input_dim) 
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def preprocess_for_ae(df, scaler=None, is_train=True):
    df_feat = add_base_features(df)
    
    numeric_cols = [
        'Login Hour', 'Time_Since_Last_Login_Sec', 'Time_Gap_Deviation_Ratio',
        'User_Login_Hour_Deviation', 'User_Mean_Login_Hour_Prior',
        'First_Time_Country', 'First_Time_ASN', 'First_Time_Subnet', 'First_Time_Device_Combo',
        'First_Time_Country_Device_Combo', 'First_Time_ASN_Device_Combo',
        'User_Login_Count_Prior', 'User_Country_Count_Prior', 'User_ASN_Count_Prior',
        'Secs_Since_Last_Country_For_User', 'Secs_Since_Last_Device_For_User',
        'Login Successful', 'Fail_Count_10Min', 'Fail_Count_1Hour',
        'Success_After_Fail_Burst', 'Recent_Fail_Success_Ratio',
        # User requested stronger behavioral AE hooks
        'Recency_Country_Normalized', 'Recency_ASN_Normalized',
        'Recency_Subnet_Normalized', 'Recency_Device_Normalized',
        'Seen_Country_Before', 'Seen_ASN_Before', 
        'Seen_Subnet_Before', 'Seen_Device_Before',
        'Consecutive_Failures_Prior', 'New_Entities_10Min', 'New_Entities_1Hour'
    ]
    
    # Filter for columns that actually exist in the dataframe
    numeric_cols = [c for c in numeric_cols if c in df_feat.columns]
    
    # Catching edge-cases by filling any potential NAs gracefully
    data_to_scale = df_feat[numeric_cols].copy().fillna(0)
    
    if is_train:
        os.makedirs('models', exist_ok=True)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_to_scale)
        joblib.dump(scaler, 'models/ae_scaler.joblib')
        joblib.dump(numeric_cols, 'models/ae_numeric_cols.joblib')
    else:
        scaled_data = scaler.transform(data_to_scale)
        
    return scaled_data, numeric_cols, scaler

def train_autoencoder(train_path='data/splits/train.csv', val_path='data/splits/val.csv'):
    print(f"Loading AE train data from {train_path}...")
    if not os.path.exists(train_path):
        print("Splits not found. Please run split_data.py first.")
        return

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    
    # Train ONLY on Normal Logins
    normal_df_train = df_train[df_train['Is Account Takeover'] == False]
    normal_df_val   = df_val[df_val['Is Account Takeover'] == False]
    
    scaled_train, cols, scaler = preprocess_for_ae(normal_df_train, is_train=True)
    scaled_val, _, _ = preprocess_for_ae(normal_df_val, scaler=scaler, is_train=False)
    
    input_dim = scaled_train.shape[1]
    model = Autoencoder(input_dim)
    
    # Huber loss handles heavy tabular tails substantially better than standard MSE
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_tensor = torch.FloatTensor(scaled_train)
    val_tensor   = torch.FloatTensor(scaled_val)
    
    dataset = TensorDataset(train_tensor, train_tensor)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)
    
    print("Training Autoencoder with Early Stopping...")
    epochs = 25
    best_val_loss = float('inf')
    patience = 4
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_data, _ in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(dataloader)
        
        model.eval()
        with torch.no_grad():
            val_out = model(val_tensor)
            val_loss = criterion(val_out, val_tensor).item()
            
        print(f"Epoch [{epoch+1:02d}/{epochs}] - Train Loss: {avg_train_loss:.5f} | Val normal loss: {val_loss:.5f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            if not os.path.exists('models'):
                os.makedirs('models')
            torch.save(model.state_dict(), 'models/autoencoder.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at Epoch {epoch+1}")
                break

    print("Loading best AE model for threshold calibration...")
    model.load_state_dict(torch.load('models/autoencoder.pth'))
    model.eval()
    
    # Evaluate across the entire validation subset to generate final score distribution mapping
    scaled_val_full, _, _ = preprocess_for_ae(df_val, scaler=scaler, is_train=False)
    val_full_tensor = torch.FloatTensor(scaled_val_full)
    
    with torch.no_grad():
        reconstruction = model(val_full_tensor)
        # Final scoring metric is standard MSE, even if training used SmoothL1 Loss
        mse_loss = torch.mean((reconstruction - val_full_tensor)**2, dim=1).numpy()
    
    val_normal_mask = (df_val['Is Account Takeover'] == False).values
    val_normal_errors = mse_loss[val_normal_mask]
    
    threshold = np.percentile(val_normal_errors, 99)
    print(f"Computed an AE threshold (99th pct of normal validation): {threshold:.5f}")
    
    # Save scoring bounds to construct normalized anomaly confidence inside the pipeline
    joblib.dump({
        'threshold_99': threshold,
        'threshold_50': np.percentile(val_normal_errors, 50),
        'threshold_10': np.percentile(val_normal_errors, 10),
        'best_val_loss': best_val_loss,
        'input_dim': input_dim,
        'numeric_cols': cols
    }, 'models/ae_calibration.joblib')

    with open('models/ae_threshold.txt', 'w') as f:
         f.write(str(threshold))

if __name__ == "__main__":
    train_autoencoder()
