import os
import sys
import pandas as pd
import numpy as np
import json

# Ensure local imports work
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'engine'))
from features import add_behavioral_features, add_base_features, drop_unneeded_columns_for_ml
from graph_features import add_graph_features

def analyze_feature_drift():
    print("\n" + "="*80)
    print("      FEATURE DRIFT DIAGNOSTIC: STATEFUL HORIZON ANALYSIS")
    print("="*80)
    
    raw_data_path = 'data/rba-dataset.csv'
    if not os.path.exists(raw_data_path):
        print("Raw dataset not found.")
        return

    # Load enough rows to observe maturity drift (600k is a safe memory limit)
    print("Loading 600k rows for drift analysis...")
    df = pd.read_csv(raw_data_path).head(600000)
    df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'])
    df = df.sort_values('Login Timestamp').reset_index(drop=True)

    print("Executing full feature engine...")
    df = add_behavioral_features(df)
    df = add_graph_features(df)
    df = add_base_features(df)

    # Suspect features for drift
    target_features = [
        'User_Login_Count_Prior',
        'User_Country_Count_Prior',
        'User_ASN_Count_Prior',
        'User_Subnet_Count_Prior',
        'User_Device_Combo_Count_Prior',
        'Fail_Count_1Hour', 
        'Time_Gap_Deviation_Ratio',
        'User_Mean_Time_Between_Logins_Prior',
        'GF_User_Drift_Score'
    ]
    
    target_features = [f for f in target_features if f in df.columns]
    
    print(f"\nAnalyzing drift across {len(target_features)} suspects...")
    
    # Label prevalence check
    if 'Is Account Takeover' in df.columns:
        print("\n[Label Prevalence]")
        print(f"      Early 100k: {df.head(100000)['Is Account Takeover'].mean():.6%}")
        print(f"      Late 100k : {df.tail(100000)['Is Account Takeover'].mean():.6%}")

    early_df = df.head(100000)
    late_df = df.tail(100000)
    
    drift_stats = {}
    
    for f in target_features:
        early_stats = early_df[f].describe(percentiles=[0.5, 0.95, 0.99])
        late_stats = late_df[f].describe(percentiles=[0.5, 0.95, 0.99])
        
        eps = 1e-9
        drift_stats[f] = {
            "early": {
                "mean": round(early_stats['mean'], 4),
                "median": round(early_stats['50%'], 4),
                "p95": round(early_stats['95%'], 4),
                "p99": round(early_stats['99%'], 4),
                "max": round(early_stats['max'], 4)
            },
            "late": {
                "mean": round(late_stats['mean'], 4),
                "median": round(late_stats['50%'], 4),
                "p95": round(late_stats['95%'], 4),
                "p99": round(late_stats['99%'], 4),
                "max": round(late_stats['max'], 4)
            },
            "ratio_mean": round(late_stats['mean'] / (early_stats['mean'] + eps), 2),
            "ratio_p99": round(late_stats['99%'] / (early_stats['99%'] + eps), 2),
            "ratio_max": round(late_stats['max'] / (early_stats['max'] + eps), 2)
        }

    # Print Report
    print("\n" + "-"*100)
    print(f"{'Feature':<30} | {'Early P99':<10} | {'Late P99':<10} | {'Drift (Mean)':<12} | {'Drift (P99)'}")
    print("-"*100)
    for f, s in drift_stats.items():
        print(f"{f:<30} | {s['early']['p99']:<10} | {s['late']['p99']:<10} | {s['ratio_mean']:<12} | {s['ratio_p99']}x")
    print("-"*100)

    # Save Diagnostic
    os.makedirs('reports', exist_ok=True)
    with open('reports/feature_drift_diagnostic.json', 'w') as f:
        json.dump(drift_stats, f, indent=4)
    print("\nDetailed diagnostic saved to reports/feature_drift_diagnostic.json")

if __name__ == "__main__":
    analyze_feature_drift()
