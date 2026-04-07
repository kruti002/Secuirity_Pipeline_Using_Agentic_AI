import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import add_base_features, drop_unneeded_columns_for_ml

# ---------------------------------------------------------------------------
# KNOWN FEATURE GROUPS
# Pipeline will only use columns that actually exist in the DataFrame.
# ---------------------------------------------------------------------------
CANDIDATE_CATEGORICAL = [
    'Country', 'OS Name and Version', 'Browser Name and Version', 'Device Type'
]

CANDIDATE_NUMERICAL = [
    # Temporal
    'Login Hour', 'Login_Hour_Sin', 'Login_Hour_Cos',
    'Login_Day_Of_Week', 'Is_Weekend',
    'Time_Since_Last_Login_Sec', 'Time_Gap_Deviation_Ratio',
    'User_Mean_Login_Hour_Prior', 'User_Login_Hour_Deviation',
    'User_Mean_Time_Between_Logins_Prior',
    'User_Weekend_Login_Rate_Prior', 'Weekend_Behavior_Deviation',
    # Novelty
    'First_Time_Country', 'First_Time_ASN', 'First_Time_Subnet', 'First_Time_Device_Combo',
    'First_Time_Country_Device_Combo', 'First_Time_ASN_Device_Combo',
    'First_Time_Subnet_Device_Combo', 'First_Time_Country_ASN',
    # Seen-before flags
    'Seen_Country_Before', 'Seen_ASN_Before', 'Seen_Subnet_Before', 'Seen_Device_Before',
    'Seen_Country_Device_Before', 'Seen_ASN_Device_Before', 'Seen_Subnet_Device_Before',
    # Frequency
    'User_Login_Count_Prior', 'User_Country_Count_Prior', 'User_ASN_Count_Prior',
    'User_Subnet_Count_Prior', 'User_Device_Combo_Count_Prior',
    # Recency (raw + normalized)
    'Secs_Since_Last_Country_For_User', 'Secs_Since_Last_ASN_For_User',
    'Secs_Since_Last_Subnet_For_User', 'Secs_Since_Last_Device_For_User',
    'Secs_Since_Last_Country_Device_For_User', 'Secs_Since_Last_ASN_Device_For_User',
    'Secs_Since_Last_Subnet_Device_For_User',
    'Recency_Country_Normalized', 'Recency_ASN_Normalized',
    'Recency_Subnet_Normalized', 'Recency_Device_Normalized',
    # Stability
    'Country_Change_Rate_Prior', 'ASN_Change_Rate_Prior',
    'Subnet_Change_Rate_Prior', 'Device_Change_Rate_Prior',
    # Failure baselines
    'Login Successful', 'User_Fail_Count_Prior', 'User_Success_Count_Prior',
    'User_Failure_Rate_Prior',
    # Rolling activity
    'Fail_Count_10Min', 'Fail_Count_1Hour', 'Success_Count_10Min',
    'Recent_Fail_Burst', 'Recent_Fail_Success_Ratio',
    'Success_After_Fail_Burst', 'Consecutive_Failures_Prior',
    'Success_After_Consecutive_Fails',
    # Entity explosion
    'New_Entities_10Min', 'New_Entities_1Hour',
    'New_Subnets_1Hour', 'New_Device_Combos_1Hour',
    # Maturity interactions
    'Mature_Account_New_Country', 'Mature_Account_New_Device',
    'Mature_Account_Multi_Entity_Drift',
    # Interaction features
    'New_Device_And_New_Subnet', 'New_Country_And_Fail_Burst',
    'New_ASN_And_Unusual_Hour', 'Success_After_Fail_And_New_Device',
    # Missingness
    'Missing_Country', 'Missing_ASN', 'Missing_Device_Info',
    # Graph features (if pre-baked into splits)
    'GF_New_Entity_Count', 'GF_User_Drift_Score',
    'GF_Edge_User_Subnet_New', 'GF_Edge_User_ASN_New',
    'GF_Edge_User_Country_New', 'GF_Edge_User_Device_New',
    'GF_Subnet_User_Count_Before', 'GF_ASN_User_Count_Before',
    'GF_Country_User_Count_Before', 'GF_Device_User_Count_Before',
    'GF_User_Subnet_Count_Before', 'GF_User_ASN_Count_Before',
    'GF_User_Country_Count_Before', 'GF_User_Device_Count_Before',
    'GF_Subnet_Failure_Rate_Prior', 'GF_ASN_Failure_Rate_Prior',
    'GF_Device_Failure_Rate_Prior',
]


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _make_ohe():
    """Version-safe OneHotEncoder: sparse_output (sklearn>=1.2) or sparse (older)."""
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)


def precision_at_k(y_true, y_score, k):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    idx = np.argsort(y_score)[::-1][:k]
    return float(y_true[idx].mean())


def find_best_threshold(y_true, y_prob, target_recall=0.70):
    """
    Return the threshold with highest precision among all points
    where recall >= target_recall. Falls back to 0.30 if none qualify.
    """
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    # The last element of prec/rec is a sentinel with no matching threshold
    candidates = [
        (p, r, t)
        for p, r, t in zip(prec[:-1], rec[:-1], thresholds)
        if r >= target_recall
    ]
    if not candidates:
        return 0.30
    best_p, best_r, best_t = max(candidates, key=lambda x: x[0])
    return float(best_t)


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
def train_supervised_model(
    train_path='data/splits/train.csv',
    val_path='data/splits/val.csv'
):
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("Pre-computed splits not found. Please run src/tools/split_data.py first!")
        return None

    print(f"Loading train data from {train_path}...")
    df_train = pd.read_csv(train_path)
    df_val   = pd.read_csv(val_path)

    df_train = drop_unneeded_columns_for_ml(add_base_features(df_train))
    df_val   = drop_unneeded_columns_for_ml(add_base_features(df_val))

    # Guard: ensure target column exists in both splits
    target_col = 'Is Account Takeover'
    if target_col not in df_train.columns or target_col not in df_val.columns:
        print(f"Missing target column: '{target_col}' — aborting.")
        return None

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col].astype(int)
    X_val   = df_val.drop(columns=[target_col])
    y_val   = df_val[target_col].astype(int)

    # --- Class distribution check ---
    print("\nTrain label distribution:")
    print(y_train.value_counts(normalize=True).round(4))
    print("\nValidation label distribution:")
    print(y_val.value_counts(normalize=True).round(4))

    # --- Dynamic feature selection (robust to missing/added columns) ---
    categorical_cols = [c for c in CANDIDATE_CATEGORICAL if c in X_train.columns]
    
    # Identify numerical columns dynamically
    numerical_cols = [
        c for c in X_train.columns 
        if c not in categorical_cols and pd.api.types.is_numeric_dtype(X_train[c])
    ]
    
    # Print feature count sanity check
    print(f"\nLocked: {len(numerical_cols)} numerical + {len(categorical_cols)} categorical features.")
    if len(numerical_cols) < 20:
         print("Warning: Low numerical feature count detected. Check upstream engineering.")

    # --- Preprocessing pipelines ---
    # Trees handle unscaled data fine, but we include imputers for robustness
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0))
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', _make_ohe())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )

    # --- Model ---
    classifier = RandomForestClassifier(
        n_estimators=400,
        max_depth=16,
        min_samples_leaf=5,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    print("\nTraining Supervised Model (ATO Detection)...")
    model.fit(X_train, y_train)

    # --- Evaluation ---
    y_prob = model.predict_proba(X_val)[:, 1]

    # Tune threshold on validation for best high-recall point
    threshold = find_best_threshold(y_val, y_prob, target_recall=0.70)
    print(f"\nAuto-tuned threshold (≥70% recall target): {threshold:.4f}")

    y_pred = (y_prob >= threshold).astype(int)

    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_pred, digits=4,
                                 target_names=['Normal', 'ATO']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    from sklearn.metrics import precision_score, recall_score

    roc  = roc_auc_score(y_val, y_prob)
    pr   = average_precision_score(y_val, y_prob)
    print(f"\nValidation ROC AUC : {roc:.4f}")
    print(f"Validation PR  AUC : {pr:.4f}")
    print(f"Threshold precision: {precision_score(y_val, y_pred, zero_division=0):.4f}")
    print(f"Threshold recall   : {recall_score(y_val, y_pred, zero_division=0):.4f}")

    print("\nPrecision@K / Recall@K (top-K ranked by score):")
    pos_total = int(y_val.sum())
    for k in [50, 100, 500, 1000]:
        if len(y_val) >= k:
            p_at_k = precision_at_k(y_val, y_prob, k)
            # Recall@K = hits in top-K / total positives
            idx    = np.argsort(np.asarray(y_prob))[::-1][:k]
            r_at_k = float(np.asarray(y_val)[idx].sum()) / max(1, pos_total)
            print(f"  @{k:<4d}  Precision: {p_at_k:.4f}  Recall: {r_at_k:.4f}")

    # --- Save model bundle (model + metadata) ---
    os.makedirs('models', exist_ok=True)
    bundle = {
        'model':            model,
        'threshold':        threshold,
        'categorical_cols': categorical_cols,
        'numerical_cols':   numerical_cols,
        'val_roc_auc':      roc,
        'val_pr_auc':       pr,
    }
    joblib.dump(bundle, 'models/supervised_model_bundle.joblib')
    joblib.dump(model,  'models/supervised_model.joblib')   # backward-compat
    print("\nModel bundle → models/supervised_model_bundle.joblib")

    # --- Validation prediction export ---
    val_out = pd.DataFrame({
        'y_true': np.asarray(y_val),
        'y_prob': y_prob,
        'y_pred': y_pred
    })
    val_out.to_csv('models/supervised_val_predictions.csv', index=False)
    print("Val predictions  → models/supervised_val_predictions.csv")

    # --- Feature importance export ---
    try:
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        importances   = model.named_steps['classifier'].feature_importances_
        feat_imp = pd.DataFrame({
            'feature':    feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        feat_imp.to_csv('models/supervised_feature_importance.csv', index=False)
        print("Feature importance → models/supervised_feature_importance.csv")
        print("\nTop 15 features:")
        print(feat_imp.head(15).to_string(index=False))
    except Exception as e:
        print(f"Feature importance export skipped: {e}")

    return bundle


if __name__ == "__main__":
    train_supervised_model()
