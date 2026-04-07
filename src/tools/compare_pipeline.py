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

def run_experiment(mode="full", val_data=None):
    """
    Evaluates specific hybrid risk combinations against the validation set.
    """
    df = val_data.copy()
    
    # 1. LOAD MODELS
    print(f"\nEvaluating Layer Configuration: [{mode.upper()}]")
    sup_bundle = joblib.load('models/supervised_model_bundle.joblib')
    graph_bundle = joblib.load('models/graph_anomaly_bundle.joblib')
    
    # 2. GENERATE COMPONENT SCORES
    # supervised
    X_sup = df[sup_bundle['numerical_cols'] + sup_bundle['categorical_cols']].fillna(0)
    supervised_scores = sup_bundle['model'].predict_proba(X_sup)[:, 1]
    
    # graph anomaly
    low, high = graph_bundle['low_bound'], graph_bundle['high_bound']
    X_graph = graph_bundle['scaler'].transform(df[graph_bundle['graph_cols']].fillna(0))
    raw_graph_scores = graph_bundle['model'].decision_function(X_graph)
    graph_anomaly_scores = normalize_score(raw_graph_scores, low, high)
    
    # rule engine
    df_rules = apply_rule_engine(df)
    rule_scores = df_rules['rule_score'].values
    
    # 3. FUSE SCORES
    if mode == "rf_only":
        final_score = supervised_scores
    elif mode == "rf_graph":
        # Weighted mean: 70% supervised + 30% anomaly
        final_score = (0.7 * supervised_scores) + (0.3 * graph_anomaly_scores)
    elif mode == "rf_rules":
        # Max-fusion: Catch anything heavy in rules
        final_score = np.maximum(supervised_scores, rule_scores)
    elif mode == "full":
        # Full hybrid fusion: 50/30/20 balance
        fused = (0.5 * supervised_scores) + (0.3 * graph_anomaly_scores) + (0.2 * rule_scores)
        # Force rules to override if they hit critical alerts
        final_score = np.maximum(fused, rule_scores * 0.8)
    
    # 4. METRICS (Ranking based P@K)
    y_true = df['Is Account Takeover'].astype(int).values
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
    val_path = 'data/splits/val.csv'
    if not os.path.exists(val_path):
        print("Validation split missing. Please run split_data.py first.")
        return
        
    df_val = pd.read_csv(val_path)
    
    modes = ["rf_only", "rf_graph", "rf_rules", "full"]
    results = []
    
    for m in modes:
        results.append(run_experiment(m, df_val))
    
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("      HYBRID PIPELINE PERFORMANCE COMPARISON (Validation Set)")
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80)
    
    # Save the results
    os.makedirs('reports', exist_ok=True)
    df_results.to_csv('reports/pipeline_comparison.csv', index=False)
    print("\nComparison report saved to reports/pipeline_comparison.csv")

if __name__ == "__main__":
    main()
