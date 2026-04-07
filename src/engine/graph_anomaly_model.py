import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Ensure local directory imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def normalize_score(decision_scores, low_bound, high_bound):
    """
    Normalizes Isolation Forest decision scores into a [0, 1] anomaly score.
    In Isolation Forest, lower raw scores are more anomalous.
    Mapped Output: 1.0 (Most Anomalous) -> 0.0 (Most Normal)
    """
    decision_scores = np.asarray(decision_scores)
    # Clip to bounds seen during calibration
    norm = (decision_scores - low_bound) / (high_bound - low_bound)
    norm = np.clip(norm, 0, 1)
    # Invert so higher = more anomalous
    return 1.0 - norm

def get_graph_feature_cols():
    return [
        # Drift / novelty
        'GF_New_Entity_Count', 'GF_User_Drift_Score',
        'GF_Edge_User_Subnet_New', 'GF_Edge_User_ASN_New',
        'GF_Edge_User_Country_New', 'GF_Edge_User_Device_New',
        # New Pairwise Novelty
        'GF_Edge_User_Subnet_Device_New', 'GF_Edge_User_Country_Device_New',
        'GF_Edge_User_ASN_Device_New',
        # Infrastructure Hub Degrees (log-dampened)
        'GF_Subnet_User_Count_Before', 'GF_ASN_User_Count_Before',
        'GF_Country_User_Count_Before', 'GF_Device_User_Count_Before',
        # User neighborhood size
        'GF_User_Subnet_Count_Before', 'GF_User_ASN_Count_Before',
        'GF_User_Country_Count_Before', 'GF_User_Device_Count_Before',
        # Windowed Infrastructure Pressure
        'GF_Subnet_User_Count_10Min', 'GF_Subnet_User_Count_1Hour',
        'GF_ASN_User_Count_10Min', 'GF_Device_User_Count_10Min',
        # Neighborhood Malicious Density
        'GF_Subnet_Failed_User_Count_Before', 'GF_ASN_Failed_User_Count_Before',
        'GF_Device_Failed_User_Count_Before',
        'GF_Subnet_Has_High_Failure_Pressure', 'GF_Device_Has_High_Failure_Pressure',
        # Risk propagation
        'GF_Subnet_Failure_Rate_Prior', 'GF_ASN_Failure_Rate_Prior',
        'GF_Device_Failure_Rate_Prior',
        # Temporal behavioral baselines
        'User_Login_Count_Prior', 'User_Login_Hour_Deviation',
        'Time_Gap_Deviation_Ratio',
        # Rolling activity
        'Fail_Count_10Min', 'Fail_Count_1Hour',
        'Recent_Fail_Burst', 'Success_After_Fail_Burst',
        'Time_Since_Last_Login_Sec',
    ]

def train_graph_anomaly(train_path='data/splits/train.csv', val_path='data/splits/val.csv'):
    print(f"Loading Graph Anomaly train data from {train_path}...")
    # 1. Path existence check for both splits
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"Splits not found at {train_path} or {val_path}. Please run src/tools/split_data.py first.")
        return

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    
    # 2. Guard against empty normal training set
    normal_df = df_train[df_train['Is Account Takeover'] == False].copy()
    if len(normal_df) == 0:
        print("Error: No normal training rows found for graph anomaly training. Aborting.")
        return
    
    print(f"Training rows (Normal only): {len(normal_df)}")
    
    graph_cols = get_graph_feature_cols()
    graph_cols = [c for c in graph_cols if c in df_train.columns]
    
    # 3. Feature count sanity check
    if len(graph_cols) < 10:
        print(f"Warning: Only {len(graph_cols)} graph features found. Upstream generation might have failed.")
    else:
        print(f"Using {len(graph_cols)} graph-augmented features for anomaly detection.")
    
    # 4. Switched to RobustScaler for heavy-tailed security metrics
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(normal_df[graph_cols].fillna(0))
    
    # 5. isolation Forest implementation (Tuned contamination)
    contamination = 0.01
    print(f"Training Graph Isolation Forest (Contamination={contamination})...")
    iso_model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    iso_model.fit(X_scaled)
    
    # 6. Diagnostics: Attack vs Normal separation
    print("\nEvaluating separation on Validation set...")
    df_val_normal = df_val[df_val['Is Account Takeover'] == False].copy()
    df_val_attack = df_val[df_val['Is Account Takeover'] == True].copy()
    
    if len(df_val_normal) == 0:
        df_val_normal = df_val.copy()
        
    X_val_norm_scaled = scaler.transform(df_val_normal[graph_cols].fillna(0))
    norm_scores = iso_model.decision_function(X_val_norm_scaled)
    
    if len(df_val_attack) > 0:
        X_val_att_scaled = scaler.transform(df_val_attack[graph_cols].fillna(0))
        att_scores = iso_model.decision_function(X_val_att_scaled)
        print(f"  Median Normal Score : {np.median(norm_scores):.6f}")
        print(f"  Median Attack Score : {np.median(att_scores):.6f} (Lower = More Anomalous)")
    else:
        print("  No attacks found in validation split for diagnostic comparison.")

    # 7. Calibration with zero-width guard
    low_bound = float(np.percentile(norm_scores, 1))
    high_bound = float(np.percentile(norm_scores, 99))
    if high_bound <= low_bound:
        high_bound = low_bound + 1e-6
    
    print(f"  Calibration Bounds  : low={low_bound:.6f}, high={high_bound:.6f}")
    
    # 8. Consolidate into a single Bundle artifact
    os.makedirs('models', exist_ok=True)
    bundle = {
        'scaler': scaler,
        'model': iso_model,
        'graph_cols': graph_cols,
        'low_bound': low_bound,
        'high_bound': high_bound,
        'contamination': contamination,
        'stats': {
            'norm_val_median': float(np.median(norm_scores)),
            'val_p01': low_bound,
            'val_p99': high_bound
        }
    }
    
    joblib.dump(bundle, 'models/graph_anomaly_bundle.joblib')
    
    # Backward compatibility for legacy loaders
    joblib.dump(scaler, 'models/graph_scaler.joblib')
    joblib.dump(iso_model, 'models/graph_anomaly_model.joblib')
    joblib.dump(graph_cols, 'models/graph_cols.joblib')
    joblib.dump((low_bound, high_bound), 'models/graph_calibration.joblib')
    
    print("\nGraph Anomaly Engine successfully bundled → models/graph_anomaly_bundle.joblib")

if __name__ == "__main__":
    train_graph_anomaly()
