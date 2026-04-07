import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# Set path for engine imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'engine'))
from graph_anomaly_model import normalize_score
from rules import apply_rule_engine
from features import add_base_features
from risk_aggregator import aggregate_risk_v2

def run_experiment(mode="full", val_data=None):
    """
    Evaluates specific hybrid risk combinations against the validation set.
    """
    # 1. LOAD MODELS
    print(f"\nEvaluating Layer Configuration: [{mode.upper()}]")
    sup_bundle = joblib.load('models/supervised_model_bundle.joblib')
    graph_bundle = joblib.load('models/graph_anomaly_bundle.joblib')
    
    # 2. MATCH FEATURES (Filter for columns present in val_data)
    num_cols = [c for c in sup_bundle['numerical_cols'] if c in val_data.columns]
    cat_cols = [c for c in sup_bundle['categorical_cols'] if c in val_data.columns]
    
    # Ensure no empty lists crash the predictor
    X_sup = val_data[num_cols + cat_cols].fillna(0)
    supervised_scores = sup_bundle['model'].predict_proba(X_sup)[:, 1]
    
    # 3. SCORE GENERATION
    # graph anomaly
    low, high = graph_bundle['low_bound'], graph_bundle['high_bound']
    X_graph = graph_bundle['scaler'].transform(val_data[graph_bundle['graph_cols']].fillna(0))
    raw_graph_scores = graph_bundle['model'].decision_function(X_graph)
    graph_anomaly_scores = normalize_score(raw_graph_scores, low, high)
    
    # rule engine (Pre-computed in main)
    rule_scores = val_data['rule_score'].values
    
    # 4. FUSE SCORES (Using Gated aggregate_risk_v2 logic)
    if mode == "rf_only":
        final_score = supervised_scores
    elif mode == "rf_graph":
        # Only fuse RF and Graph (Rules set to 0)
        df_tmp, _ = aggregate_risk_v2(val_data, supervised_scores, graph_anomaly_scores, np.zeros_like(supervised_scores))
        final_score = df_tmp['final_risk'].values
    elif mode == "rf_rules":
        # Only fuse RF and Rules (Graph set to 0)
        df_tmp, _ = aggregate_risk_v2(val_data, supervised_scores, np.zeros_like(supervised_scores), rule_scores)
        final_score = df_tmp['final_risk'].values
    elif mode == "full":
        # Full Hybrid Integration
        df_tmp, _ = aggregate_risk_v2(val_data, supervised_scores, graph_anomaly_scores, rule_scores)
        final_score = df_tmp['final_risk'].values
    
    # 5. METRICS (Ranking based P@K)
    y_true = val_data['Is Account Takeover'].astype(int).values
    pos_total = max(1, y_true.sum())
    
    roc = roc_auc_score(y_true, final_score)
    pr_auc = average_precision_score(y_true, final_score)
    
    # Calculate ranking metrics (Precision at K)
    def p_at_k(y, score, k):
        idx = np.argsort(score)[::-1][:k]
        return float(y[idx].mean())

    p50 = p_at_k(y_true, final_score, k=50)
    p100 = p_at_k(y_true, final_score, k=100)
    p500 = p_at_k(y_true, final_score, k=500)
    
    # Final Alerts (Using optimal supervised threshold as baseline)
    threshold = sup_bundle['threshold']
    y_pred = (final_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        'Mode': mode,
        'ROC AUC': round(roc, 4),
        'PR AUC': round(pr_auc, 4),
        'P@50': round(p50, 4),
        'P@100': round(p100, 4),
        'P@500': round(p500, 4),
        'TP': tp,
        'FP': fp
    }

def main():
    test_path = 'data/splits/test.csv'
    if not os.path.exists(test_path):
        print("Test split missing. Please run split_data.py first.")
        return
        
    print(f"Loading and expanding UNSEEN TEST data: {test_path}")
    df_test = pd.read_csv(test_path)
    
    # Pre-apply base features (Login Hour, etc) and SOC rules
    df_test = add_base_features(df_test)
    df_test = apply_rule_engine(df_test)
    
    modes = ["rf_only", "rf_graph", "rf_rules", "full"]
    results = []
    
    for m in modes:
        results.append(run_experiment(m, df_test))
    
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("      FINAL HYBRID PIPELINE PERFORMANCE COMPARISON (TEST SET)")
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80)
    
    # Save the results
    os.makedirs('reports', exist_ok=True)
    df_results.to_csv('reports/final_test_comparison.csv', index=False)
    print("\nComparison report saved to reports/final_test_comparison.csv")

if __name__ == "__main__":
    main()
