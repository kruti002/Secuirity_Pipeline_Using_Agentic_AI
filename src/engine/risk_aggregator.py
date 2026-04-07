import pandas as pd
import numpy as np

def aggregate_risk(df, supervised_score, graph_anomaly_score, rule_score):
    """
    Combines layers using isolation-graph structures.
    """
    df = df.copy()
    
    df['model_score'] = supervised_score
    df['graph_anomaly_score'] = graph_anomaly_score
    df['rule_score'] = rule_score
    
    # Base weighted risk aggregation:
    df['final_risk'] = (0.55 * df['model_score'] + 
                        0.25 * df['graph_anomaly_score'] + 
                        0.20 * df['rule_score'])
                        
    # Conditional Boosts
    boosts = pd.Series(0.0, index=df.index)
    
    boosts += np.where((df['model_score'] > 0.8) & (df['graph_anomaly_score'] > 0.8), 0.1, 0.0)
    
    if 'GF_New_Entity_Count' in df.columns:
        boosts += np.where(df['GF_New_Entity_Count'] >= 3, 0.05, 0.0)
        
    if 'Recent_Fail_Burst' in df.columns and 'GF_Edge_User_Device_New' in df.columns:
        boosts += np.where((df['Recent_Fail_Burst'] >= 3) & (df['GF_Edge_User_Device_New'] == 1), 0.05, 0.0)
        
    if 'GF_Subnet_User_Count_Before' in df.columns and 'GF_Subnet_Failure_Rate_Prior' in df.columns:
        boosts += np.where((df['GF_Subnet_User_Count_Before'] > 50) & (df['GF_Subnet_Failure_Rate_Prior'] > 0.5), 0.1, 0.0)
        
    df['final_risk'] = (df['final_risk'] + boosts).clip(0, 1)
    
    # Assign Explicit Layers
    def assign_severity(row):
        m = row['model_score']
        g = row['graph_anomaly_score']
        r = row['rule_score']
        
        # 1. Known-pattern high confidence
        if m > 0.8 and g > 0.5:
            return 'CRITICAL - Known Compromise'
        # 2. Novel behavioral anomaly
        elif m < 0.4 and g > 0.90:
            return 'HIGH - Novel Graph Structure'
        # Base falls
        elif row['final_risk'] >= 0.85: return 'CRITICAL'
        elif row['final_risk'] >= 0.65: return 'HIGH'
        elif row['final_risk'] >= 0.40: return 'MEDIUM'
        else: return 'LOW - Standard Review'
        
    df['alert_severity'] = df.apply(assign_severity, axis=1)
    
    # User-level aggregation tracking both layers separately
    user_risk = df.groupby('User ID').agg({
        'final_risk': ['max', 'mean', 'count'],
        'model_score': 'max',
        'graph_anomaly_score': 'max'
    }).reset_index()
    
    user_risk.columns = ['User ID', 'max_risk', 'avg_risk', 'alert_count', 'max_model_score', 'max_graph_score']
    
    return df, user_risk

if __name__ == "__main__":
    df = pd.DataFrame({'User ID': [100, 100, 101, 102], 'EventID': [1,2,3,4]})
    m = np.array([0.9, 0.2, 0.4, 0.0])
    a = np.array([0.9, 0.98, 0.3, 0.1])
    r = np.array([0.1, 0.0, 0.0, 0.9])
    
    df_res, user_res = aggregate_risk(df, m, a, r)
    print("Aggregate Risk Sample:")
    print(df_res[['User ID', 'model_score', 'graph_anomaly_score', 'rule_score', 'final_risk', 'alert_severity']])
