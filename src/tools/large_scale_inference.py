import os
import sys
import joblib
import pandas as pd
import numpy as np
import time
import json

# Ensure local imports work
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'engine'))
from graph_embeddings import extract_embedding_features, GraphSVDModel
from features import add_base_features, drop_unneeded_columns_for_ml
from rule_engine import apply_precision_booster

def precision_at_k(y_true, y_score, k):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    k = min(k, len(y_true))
    idx = np.argsort(y_score)[::-1][:k]
    return float(y_true[idx].mean())

def recall_at_k(y_true, y_score, k):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    total_pos = y_true.sum()
    if total_pos == 0: return 0.0
    k = min(k, len(y_true))
    idx = np.argsort(y_score)[::-1][:k]
    return float(y_true[idx].sum() / total_pos)

def run_full_dataset_inference():
    print("\n" + "="*80)
    print("      TRUE PRODUCTION SIMULATION: INFERENCE ON UNSEEN TEST SET")
    print("="*80)

    # Test split = completely unseen rows (events 4.8M to 6M)
    test_path   = 'data/splits/test.csv'
    bundle_path = 'models/supervised_model_bundle.joblib'
    svd_path    = 'models/graph_svd_model.joblib'

    if not all(os.path.exists(p) for p in [test_path, bundle_path, svd_path]):
        print("Required artifacts missing. Run split_data.py and model training first.")
        return

    print("Loading models...")
    bundle  = joblib.load(bundle_path)
    svd_model = joblib.load(svd_path)
    model_p = bundle['model']
    threshold = bundle.get('threshold', 0.5)

    print(f"Loading unseen test set from data/splits/test.csv...")
    df = pd.read_csv(test_path)
    print(f"Test set: {len(df):,} rows | {int(df['Is Account Takeover'].sum())} known ATOs")

    # Features are ALREADY computed by split_data.py — just clean up non-ML columns
    df_ml  = drop_unneeded_columns_for_ml(add_base_features(df))
    X_full = df_ml.drop(columns=['Is Account Takeover'], errors='ignore')
    y_test = df['Is Account Takeover'].astype(int).values

    # Scoring
    print(f"\n[1/2] Scoring {len(X_full):,} events with Production RF...")
    start_time = time.time()
    probs = model_p.predict_proba(X_full)[:, 1]
    elapsed = time.time() - start_time
    df['risk_score'] = probs

    print("[2/2] Extracting SVD structural telemetry...")
    emb_data = extract_embedding_features(df_ml, svd_model)

    # Apply rule-based precision booster (top-5000 slice only)
    print("[3/3] Applying Graph-Aware Precision Booster...")
    df = apply_precision_booster(df, rf_score_col='risk_score', top_n=5000)

    # Metrics
    print("\n" + "="*80)
    print("      STABILITY & RANKING REPORT")
    print("="*80)
    print(f"Score Range  RF  : min={probs.min():.4f}  median={np.median(probs):.4f}  max={probs.max():.4f}")
    final_scores = df['final_score'].values
    print(f"Score Range FINAL: min={final_scores.min():.4f}  median={np.median(final_scores):.4f}  max={final_scores.max():.4f}")
    print(f"Throughput  : {len(df)/elapsed:,.0f} events/sec")
    print()
    for label, scores in [("RF Score", probs), ("Final Score (RF+Rules)", final_scores)]:
        print(f"--- {label} ---")
        print(f"  P@10  : {precision_at_k(y_test, scores, 10):.4f}    R@10  : {recall_at_k(y_test, scores, 10):.4f}")
        print(f"  P@50  : {precision_at_k(y_test, scores, 50):.4f}    R@50  : {recall_at_k(y_test, scores, 50):.4f}")
        print(f"  P@500 : {precision_at_k(y_test, scores, 500):.4f}    R@500 : {recall_at_k(y_test, scores, 500):.4f}")
        thresh_preds = (scores >= threshold).astype(int)
        tp = int(((thresh_preds == 1) & (y_test == 1)).sum())
        fp = int(((thresh_preds == 1) & (y_test == 0)).sum())
        print(f"  TP: {tp}  FP: {fp}  (threshold={threshold:.4f})")
        print()
    print("="*80)
    # ── EXPORT: Two-Tier Alert System ────────────────────────────────────────
    os.makedirs('reports', exist_ok=True)

    # Tier 1 — CRITICAL: Rule-boosted rows with final_score > 0.50
    critical_alerts = df[df['final_score'] > 0.50].sort_values('final_score', ascending=False).reset_index(drop=True)
    critical_alerts.insert(0, 'alert_tier', 'CRITICAL')
    critical_alerts.insert(1, 'alert_rank', np.arange(1, len(critical_alerts) + 1))

    # Tier 2 — SUSPICIOUS: Top 1000 by RF score (catches stealth ATOs)
    suspicious_alerts = df.sort_values('risk_score', ascending=False).head(1000).reset_index(drop=True)
    suspicious_alerts.insert(0, 'alert_tier', 'SUSPICIOUS')
    suspicious_alerts.insert(1, 'alert_rank', np.arange(1, len(suspicious_alerts) + 1))

    critical_alerts.to_csv('reports/critical_alerts.csv', index=False)
    suspicious_alerts.to_csv('reports/suspicious_alerts.csv', index=False)

    final_scores = df['final_score'].values
    stats = {
        "status": "complete",
        "total_rows": len(df),
        "known_atos": int(y_test.sum()),
        "runtime_sec": round(elapsed, 2),
        "throughput_rows_per_sec": round(len(df) / elapsed, 2),
        "rf_score_max": round(float(probs.max()), 4),
        "final_score_max": round(float(final_scores.max()), 4),
        "critical_alerts": len(critical_alerts),
        "suspicious_alerts": len(suspicious_alerts),
        "RF_P@50":     round(precision_at_k(y_test, probs, 50), 4),
        "RF_R@500":    round(recall_at_k(y_test, probs, 500), 4),
        "FINAL_P@50":  round(precision_at_k(y_test, final_scores, 50), 4),
        "FINAL_R@500": round(recall_at_k(y_test, final_scores, 500), 4),
    }
    with open('reports/inference_performance.json', 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"\nTier 1 CRITICAL alerts  : {len(critical_alerts):>5} rows -> reports/critical_alerts.csv")
    print(f"Tier 2 SUSPICIOUS alerts: {len(suspicious_alerts):>5} rows -> reports/suspicious_alerts.csv")
    print(f"Performance stats        :        -> reports/inference_performance.json")


if __name__ == "__main__":
    run_full_dataset_inference()
