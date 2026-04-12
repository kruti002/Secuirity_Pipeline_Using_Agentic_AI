import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, 
    roc_auc_score, 
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.base import clone

# Ensure local imports work
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'engine'))
from graph_embeddings import extract_embedding_features, GraphSVDModel
from features import add_base_features, drop_unneeded_columns_for_ml

def precision_at_k(y_true, y_score, k):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(y_true) < k: k = len(y_true)
    idx = np.argsort(y_score)[::-1][:k]
    return float(y_true[idx].mean())

def recall_at_k(y_true, y_score, k):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    total_pos = y_true.sum()
    if total_pos == 0: return 0.0
    if len(y_true) < k: k = len(y_true)
    idx = np.argsort(y_score)[::-1][:k]
    return float(y_true[idx].sum() / total_pos)

def run_embedding_experiment():
    print("\n" + "="*80)
    print("      SCIENTIFIC HYBRID EVALUATION: PIPELINE + SVD EMBEDDINGS")
    print("="*80)
    
    # 1. LOAD DATA & PRODUCTION BUNDLE
    train_path = 'data/splits/train.csv'
    val_path = 'data/splits/val.csv'
    test_path = 'data/splits/test.csv'
    bundle_path = 'models/supervised_model_bundle.joblib'
    svd_model_path = 'models/graph_svd_model.joblib'
    
    if not all(os.path.exists(p) for p in [train_path, val_path, test_path, bundle_path, svd_model_path]):
        print("Missing required artifacts (splits, original pipeline bundle, or SVD model).")
        return
        
    print("Loading datasets and production pipeline bundle...")
    df_train = pd.read_csv(train_path) 
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    bundle = joblib.load(bundle_path)
    svd_model = joblib.load(svd_model_path)
    
    # 2. FEATURE PIPELINE REPLICATION
    print("Mirroring production feature engineering for all splits...")
    target_col = 'Is Account Takeover'
    df_train_proc = drop_unneeded_columns_for_ml(add_base_features(df_train))
    df_val_proc = drop_unneeded_columns_for_ml(add_base_features(df_val))
    df_test_proc = drop_unneeded_columns_for_ml(add_base_features(df_test))
    
    # 3. SVD EMBEDDING EXTRACTION
    print("Extracting SVD Embedding Features...")
    emb_train = extract_embedding_features(df_train_proc, svd_model)
    emb_val = extract_embedding_features(df_val_proc, svd_model)
    emb_test = extract_embedding_features(df_test_proc, svd_model)
    
    # 4. PREPARE X AND Y
    X_train_base = df_train_proc.drop(columns=[target_col])
    X_val_base = df_val_proc.drop(columns=[target_col])
    X_test_base = df_test_proc.drop(columns=[target_col])
    
    y_train = df_train_proc[target_col].astype(int)
    y_val = df_val_proc[target_col].astype(int)
    y_test = df_test_proc[target_col].astype(int)
    
    X_train_emb = pd.concat([X_train_base.reset_index(drop=True), emb_train.reset_index(drop=True)], axis=1)
    X_val_emb = pd.concat([X_val_base.reset_index(drop=True), emb_val.reset_index(drop=True)], axis=1)
    X_test_emb = pd.concat([X_test_base.reset_index(drop=True), emb_test.reset_index(drop=True)], axis=1)

    # 5. EXECUTE PIPELINE COMPARISON
    results = []

    def run_sub_experiment(name, X_tr, X_va, X_te):
        print(f"\n[Experiment] {name} | Feature Count: {X_tr.shape[1]}")
        model_p = clone(bundle['model'])
        model_p.fit(X_tr, y_train)
        
        # Eval
        val_probs = model_p.predict_proba(X_va)[:, 1]
        val_pr_auc = average_precision_score(y_val, val_probs)
        
        test_probs = model_p.predict_proba(X_te)[:, 1]
        
        # Threshold-based classification
        threshold = bundle.get('threshold', 0.5)
        test_preds = (test_probs >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel() if y_test.nunique() > 1 else (0,0,0,0)
        
        res = {
            'Mode': name,
            'VAL PR AUC': round(val_pr_auc, 4),
            'TEST PR AUC': round(average_precision_score(y_test, test_probs), 4),
            'P@10': round(precision_at_k(y_test, test_probs, 10), 4),
            'P@50': round(precision_at_k(y_test, test_probs, 50), 4),
            'TP': tp,
            'FP': fp,
            'ROC AUC': round(roc_auc_score(y_test, test_probs), 4) if y_test.nunique() > 1 else np.nan
        }
        return res, test_probs

    res_A, p_A = run_sub_experiment("Production Baseline", X_train_base, X_val_base, X_test_base)
    results.append(res_A)
    
    res_B, p_B = run_sub_experiment("Baseline + SVD Embeds", X_train_emb, X_val_emb, X_test_emb)
    results.append(res_B)

    # 6. FINAL COMPARISON REPORT
    df_report = pd.DataFrame(results)
    print("\n" + "="*80)
    print("      FINAL HYBRID PIPELINE EVOLUTION REPORT")
    print("="*80)
    print(df_report.to_string(index=False))
    print("="*80)
    
    df_test_proc['baseline_prob'] = p_A
    df_test_proc['svd_hybrid_prob'] = p_B
    os.makedirs('reports', exist_ok=True)
    df_test_proc.to_csv('reports/final_prediction_comparison.csv', index=False)
    print("\nIn-depth prediction comparison saved to reports/final_prediction_comparison.csv")
    
if __name__ == "__main__":
    run_embedding_experiment()
