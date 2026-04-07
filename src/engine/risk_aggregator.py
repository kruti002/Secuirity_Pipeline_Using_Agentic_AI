import pandas as pd
import numpy as np

def aggregate_risk_v2(df, supervised_score, graph_anomaly_score, rule_score):
    """
    Advanced SOC Risk Fusion (v2):
    - RF is the primary ranker.
    - Graph Anomaly acts as a GATED NOVELTY BOOSTER for medium-confidence cases.
    - Rules provide explicit EXPLANATION-ONLY flags or HIGH-FIDELITY overrides.
    """
    df = df.copy()
    
    df['model_score'] = supervised_score
    df['graph_anomaly_score'] = graph_anomaly_score
    df['rule_score'] = rule_score
    
    # 1. BASE RANKING (Primary: Supervised Model)
    # We use a high weight for RF to maintain precision.
    final_risk = 0.8 * df['model_score']
    
    # 2. GATED GRAPH NOVELTY BOOST
    # We only let the graph influence the score if it shows extreme structural novelty 
    # and the RF model is in its "uncertainty band" (0.15 to 0.70).
    graph_boost = np.where(
        (df['graph_anomaly_score'] > 0.92) & 
        (df['model_score'] >= 0.15) & (df['model_score'] <= 0.70),
        0.15, # Significant boost for "hidden" anomalies
        0.0
    )
    
    # Extra boost for extreme graph novelty regardless of RF (for the "Novelty Queue")
    extreme_novelty = np.where(df['graph_anomaly_score'] > 0.98, 0.10, 0.0)
    
    # 3. RULE ENGINE OVERRIDES
    # Instead of a flat additive weight, rules provide a floor for high-fidelity signals.
    # If a critical rule triggers (e.g. Success after Fail Burst), the risk is at least 0.7.
    rule_floor = df['rule_score'] * 0.7
    
    # 4. FINAL FUSION
    final_risk = np.maximum(final_risk + graph_boost + extreme_novelty, rule_floor)
    df['final_risk'] = final_risk.clip(0, 1)
    
    # 5. SEVERITY & QUEUE ASSIGNMENT
    def assign_soc_lane(row):
        m = row['model_score']
        g = row['graph_anomaly_score']
        r = row['rule_score']
        
        if r > 0.8:
            return 'CRITICAL - Policy Violation'
        if m > 0.75:
            return 'HIGH - Known ATO Pattern'
        if g > 0.95 and m < 0.3:
            return 'HIGH - Novel Structural Anomaly'
        if row['final_risk'] > 0.5:
            return 'MEDIUM - Behavioral Review'
        return 'LOW - Baseline'
        
    df['alert_severity'] = df.apply(assign_soc_lane, axis=1)
    
    # User-level aggregation
    user_risk = df.groupby('User ID').agg({
        'final_risk': ['max', 'mean', 'count'],
        'model_score': 'max',
        'graph_anomaly_score': 'max'
    }).reset_index()
    user_risk.columns = ['User ID', 'max_risk', 'avg_risk', 'alert_count', 'max_model_score', 'max_graph_score']
    
    return df, user_risk
