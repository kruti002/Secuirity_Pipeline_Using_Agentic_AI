"""
compare_models.py
-----------------
Trains and evaluates two classifiers on the same train/val splits:
  1. Random Forest          (current baseline)
  2. HistGradientBoosting   (boosted-tree challenger)

Outputs a side-by-side metric table and saves both model bundles.

Usage:
    python src/tools/compare_models.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'engine'))
from features import add_base_features, drop_unneeded_columns_for_ml
from supervised_model import (
    CANDIDATE_CATEGORICAL,
    CANDIDATE_NUMERICAL,
    _make_ohe,
    precision_at_k,
    find_best_threshold,
)

TRAIN_PATH = 'data/splits/train_model.csv'
VAL_PATH   = 'data/splits/val_model.csv'
TARGET_COL = 'Is Account Takeover'


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_splits():
    assert os.path.exists(TRAIN_PATH), f"Missing {TRAIN_PATH} — run split_data.py first."
    assert os.path.exists(VAL_PATH),   f"Missing {VAL_PATH} — run split_data.py first."

    df_train = drop_unneeded_columns_for_ml(add_base_features(pd.read_csv(TRAIN_PATH)))
    df_val   = drop_unneeded_columns_for_ml(add_base_features(pd.read_csv(VAL_PATH)))

    X_train = df_train.drop(columns=[TARGET_COL])
    y_train = df_train[TARGET_COL].astype(int)
    X_val   = df_val.drop(columns=[TARGET_COL])
    y_val   = df_val[TARGET_COL].astype(int)

    # Dynamic column selection (same logic as supervised_model.py)
    categorical_cols = [c for c in CANDIDATE_CATEGORICAL if c in X_train.columns]
    numerical_cols   = [c for c in CANDIDATE_NUMERICAL   if c in X_train.columns]
    all_known = set(categorical_cols + numerical_cols)
    extra_num = [
        c for c in X_train.columns
        if c not in all_known and pd.api.types.is_numeric_dtype(X_train[c])
    ]
    numerical_cols += extra_num

    print(f"Features: {len(numerical_cols)} numeric + {len(categorical_cols)} categorical")
    print(f"\nTrain positives: {y_train.sum()} / {len(y_train)} "
          f"({y_train.mean()*100:.3f}%)")
    print(f"Val   positives: {y_val.sum()} / {len(y_val)} "
          f"({y_val.mean()*100:.3f}%)\n")

    return X_train, y_train, X_val, y_val, numerical_cols, categorical_cols


# ---------------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------------
def build_preprocessor(numerical_cols, categorical_cols):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0))
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot',  _make_ohe())
    ])
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer,     numerical_cols),
            ('cat', categorical_transformer, categorical_cols),
        ],
        remainder='drop'
    )


# ---------------------------------------------------------------------------
# MODELS TO COMPARE
# ---------------------------------------------------------------------------
def get_model_configs(numerical_cols, categorical_cols):
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)

    return {
        'RandomForest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=400,
                max_depth=16,
                min_samples_leaf=5,
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1
            ))
        ]),

        'HistGradientBoosting': Pipeline(steps=[
            ('preprocessor', build_preprocessor(numerical_cols, categorical_cols)),
            # HGB handles missing values natively — imputer still applied for OHE
            ('classifier', HistGradientBoostingClassifier(
                max_iter=400,
                max_depth=8,
                min_samples_leaf=20,
                class_weight='balanced',
                random_state=42,
            ))
        ]),
    }


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------
def evaluate(name, model, X_val, y_val, threshold):
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    roc    = roc_auc_score(y_val, y_prob)
    pr_auc = average_precision_score(y_val, y_prob)
    prec   = precision_score(y_val, y_pred, zero_division=0)
    rec    = recall_score(y_val, y_pred,    zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

    pos_total = max(1, int(y_val.sum()))
    p_at_k_results, r_at_k_results = {}, {}
    for k in [50, 100, 500, 1000]:
        if len(y_val) >= k:
            idx = np.argsort(np.asarray(y_prob))[::-1][:k]
            p_at_k_results[k] = float(np.asarray(y_val)[idx].mean())
            r_at_k_results[k] = float(np.asarray(y_val)[idx].sum()) / pos_total

    return {
        'Model':            name,
        'Threshold':        round(threshold, 4),
        'ROC AUC':          round(roc,    4),
        'PR  AUC':          round(pr_auc, 4),
        'Precision':        round(prec,   4),
        'Recall':           round(rec,    4),
        'TP':               tp,
        'FP':               fp,
        'FN':               fn,
        'TN':               tn,
        **{f'P@{k}':  round(v, 4) for k, v in p_at_k_results.items()},
        **{f'R@{k}':  round(v, 4) for k, v in r_at_k_results.items()},
        '_prob': y_prob,   # kept for saving predictions, dropped from table
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def compare_models():
    print("=" * 60)
    print("  MODEL COMPARISON: RandomForest vs HistGradientBoosting")
    print("=" * 60)

    X_train, y_train, X_val, y_val, num_cols, cat_cols = load_splits()
    models = get_model_configs(num_cols, cat_cols)

    results   = []
    threshold_map = {}

    for name, model in models.items():
        print(f"\n{'─'*50}")
        print(f"  Training: {name}")
        print(f"{'─'*50}")
        model.fit(X_train, y_train)

        y_prob    = model.predict_proba(X_val)[:, 1]
        threshold = find_best_threshold(y_val, y_prob, target_recall=0.70)
        threshold_map[name] = threshold
        print(f"  Auto-tuned threshold: {threshold:.4f}")

        metrics = evaluate(name, model, X_val, y_val, threshold)
        results.append(metrics)

        # Save each model bundle
        os.makedirs('models', exist_ok=True)
        safe_name = name.lower().replace(' ', '_')
        bundle = {
            'model':            model,
            'threshold':        threshold,
            'numerical_cols':   num_cols,
            'categorical_cols': cat_cols,
            'val_roc_auc':      metrics['ROC AUC'],
            'val_pr_auc':       metrics['PR  AUC'],
        }
        joblib.dump(bundle, f'models/{safe_name}_bundle.joblib')

        # Save val predictions for this model
        pd.DataFrame({
            'y_true': np.asarray(y_val),
            'y_prob': metrics['_prob'],
            'y_pred': (metrics['_prob'] >= threshold).astype(int)
        }).to_csv(f'models/{safe_name}_val_predictions.csv', index=False)

    # ---------------------------------------------------------------------------
    # COMPARISON TABLE
    # ---------------------------------------------------------------------------
    print(f"\n\n{'=' * 60}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 60}")

    # Drop the internal prob array before displaying
    display_cols = [
        'Model', 'Threshold',
        'ROC AUC', 'PR  AUC',
        'Precision', 'Recall',
        'TP', 'FP', 'FN',
        'P@50', 'P@100', 'P@500',
        'R@50', 'R@100', 'R@500',
    ]

    df_res = pd.DataFrame([{k: v for k, v in r.items() if k != '_prob'} for r in results])
    available = [c for c in display_cols if c in df_res.columns]
    print(df_res[available].to_string(index=False))

    # Save comparison table
    df_res.drop(columns=['_prob'], errors='ignore').to_csv(
        'models/model_comparison.csv', index=False
    )
    print("\nComparison table → models/model_comparison.csv")

    # Pick the winner by PR AUC (better metric for imbalanced ATO)
    winner = df_res.loc[df_res['PR  AUC'].idxmax(), 'Model']
    print(f"\n🏆  Best model by PR AUC: {winner}")

    return df_res


if __name__ == '__main__':
    compare_models()
