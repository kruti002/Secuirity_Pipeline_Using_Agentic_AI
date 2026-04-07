import pandas as pd
import numpy as np
import joblib
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from features import add_base_features, drop_unneeded_columns_for_ml
from rule_engine import apply_rules
from risk_aggregator import aggregate_risk
from graph_anomaly_model import get_graph_feature_cols

def run_pipeline(eval_path='data/splits/test.csv'):
    print(f"Loading unseen test data from {eval_path}...")
    if not os.path.exists(eval_path):
        print("Test data not found. Please run src/tools/split_data.py first.")
        return

    df_raw = pd.read_csv(eval_path)
    
    print("Preprocessing for supervised model...")
    df_proc = add_base_features(df_raw)
    df_proc = drop_unneeded_columns_for_ml(df_proc)
    
    # Run supervised model
    print("Running Supervised Model...")
    bundle   = joblib.load('models/supervised_model_bundle.joblib')
    rf_model  = bundle['model']
    threshold = bundle.get('threshold', 0.30)
    X_sup = df_proc.drop(['Is Account Takeover'], axis=1) if 'Is Account Takeover' in df_proc.columns else df_proc
    model_scores = rf_model.predict_proba(X_sup)[:, 1]
    
    # Run Graph Anomaly Engine (Isolation Forest)
    print("Running Graph Anomaly Engine...")
    graph_cols = joblib.load('models/graph_cols.joblib')
    graph_scaler = joblib.load('models/graph_scaler.joblib')
    graph_iso_model = joblib.load('models/graph_anomaly_model.joblib')
    low_bound, high_bound = joblib.load('models/graph_calibration.joblib')
    
    # We execute utilizing the structural nodes already computed chronologically during split_data
    df_graph = df_raw.copy()
    
    # Isolate exact node feature dependencies
    X_graph = df_graph[graph_cols].fillna(0)
    X_scaled = graph_scaler.transform(X_graph)
    
    # Convert arbitrary Isolation Forest returns (-1 anomaly, 1 normal) into probability-like score (0 to 1 risk)
    # Calibrate using exact validation boundaries instead of volatile chunk minimums
    decision_scores = graph_iso_model.decision_function(X_scaled)
    graph_anomaly_score = 1 - np.clip((decision_scores - low_bound) / (high_bound - low_bound), 0, 1)
    
    # Run rule engine
    print("Running Rule Engine...")
    df_rules = apply_rules(df_raw.copy())
    rule_scores = df_rules['rule_score'].values
    reason_codes = df_rules['reason_codes'].values
    
    # Sub-aggregate and export pipeline
    print("Aggregating Hybrid Risk Layers...")
    df_final, user_risk = aggregate_risk(df_raw.copy(), model_scores, graph_anomaly_score, rule_scores)
    df_final['reason_codes'] = reason_codes
    
    alerts = df_final[df_final['final_risk'] >= 0.4].sort_values(by='final_risk', ascending=False)
    
    print("Exporting data for dashboard...")
    if not os.path.exists('dashboard/public'):
        os.makedirs('dashboard/public')
        
    dash_cols = [
        'User ID', 'Login Timestamp', 'IP Address', 'Country', 'Device Type',
        'Login Successful', 'model_score', 'graph_anomaly_score', 'rule_score', 
        'final_risk', 'alert_severity', 'reason_codes', 'Is Account Takeover'
    ]
    
    existing_cols = [c for c in dash_cols if c in df_final.columns]
    
    top_alerts = alerts[existing_cols].head(5000).to_dict(orient='records')
    with open('dashboard/public/soc_alerts.json', 'w') as f:
        json.dump(top_alerts, f)
        
    user_risk_out = user_risk.sort_values(by='avg_risk', ascending=False).to_dict(orient='records')
    with open('dashboard/public/user_risks.json', 'w') as f:
        json.dump(user_risk_out, f)
        
    print("Unified Hybrid Pipeline complete. Evaluation on test set exported.")
    
if __name__ == "__main__":
    run_pipeline()
