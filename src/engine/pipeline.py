import pandas as pd
import numpy as np
import joblib
import json
import os
import sys

# Ensure local directory imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from features import add_base_features, drop_unneeded_columns_for_ml
from rules import apply_rule_engine
from graph_anomaly_model import normalize_score

def run_pipeline(eval_path='data/splits/test.csv'):
    print(f"Loading unseen test data from {eval_path}...")
    if not os.path.exists(eval_path):
        print("Test data not found. Please run src/tools/split_data.py first.")
        return

    df_raw = pd.read_csv(eval_path)
    
    # 1. LOAD BUNDLES
    print("Loading Layer Bundles (Supervised + Graph)...")
    sup_bundle = joblib.load('models/supervised_model_bundle.joblib')
    graph_bundle = joblib.load('models/graph_anomaly_bundle.joblib')
    
    # 2. BEHAVIORAL LAYER (Supervised)
    print("Executing Behavioral Layer (RF)...")
    df_proc = add_base_features(df_raw.copy())
    # Dynamic feature selection based on what the model was trained with
    X_sup = df_proc[sup_bundle['numerical_cols'] + sup_bundle['categorical_cols']].fillna(0)
    model_scores = sup_bundle['model'].predict_proba(X_sup)[:, 1]
    
    # 3. GRAPH ANOMALY LAYER (Isolation Forest)
    print("Executing Graph Anomaly Layer (IF)...")
    X_graph = graph_bundle['scaler'].transform(df_raw[graph_bundle['graph_cols']].fillna(0))
    raw_graph_scores = graph_bundle['model'].decision_function(X_graph)
    # Normalize using the fixed calibration bounds [0, 1]
    graph_anomaly_score = normalize_score(
        raw_graph_scores, 
        graph_bundle['low_bound'], 
        graph_bundle['high_bound']
    )
    
    # 4. RULE ENGINE LAYER
    print("Executing SOC Rule Engine...")
    df_rules = apply_rule_engine(df_raw)
    rule_scores = df_rules['rule_score'].values
    reason_codes = df_rules['rule_reasons'].values
    
    # 5. RISK AGGREGATION & FUSION
    print("Aggregating Hybrid Risk Score...")
    # Standardised Hybrid Fusion: 50% ML, 30% Graph, 20% Rules
    final_risk = (0.5 * model_scores) + (0.3 * graph_anomaly_score) + (0.2 * rule_scores)
    # Max override for critical rules
    final_risk = np.maximum(final_risk, rule_scores * 0.8)
    
    # Create final result set
    df_final = df_raw.copy()
    df_final['model_score'] = model_scores
    df_final['graph_anomaly_score'] = graph_anomaly_score
    df_final['rule_score'] = rule_scores
    df_final['final_risk'] = final_risk
    df_final['reason_codes'] = reason_codes
    
    def assign_severity(score):
        if score >= 0.85: return 'CRITICAL'
        if score >= 0.65: return 'HIGH'
        if score >= 0.40: return 'MEDIUM'
        return 'LOW'
    
    df_final['alert_severity'] = df_final['final_risk'].apply(assign_severity)
    
    # User-level risks for the dashboard
    user_risk = df_final.groupby('User ID').agg({
        'final_risk': ['max', 'mean', 'count'],
        'model_score': 'max',
        'graph_anomaly_score': 'max'
    }).reset_index()
    user_risk.columns = ['User ID', 'max_risk', 'avg_risk', 'alert_count', 'max_model_score', 'max_graph_score']
    
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
        
    print("Unified Hybrid Pipeline complete. Data exported to dashboard/public/")
    
if __name__ == "__main__":
    run_pipeline()
