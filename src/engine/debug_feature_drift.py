import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from engine.features import BehavioralFeatureEngine
from engine.graph_features import GraphFeatureEngine

def analyze_chunk(df, stage_name):
    suspects = [
        'User_Login_Count_Prior', 'User_Country_Count_Prior', 'User_ASN_Count_Prior',
        'User_Subnet_Count_Prior', 'User_Device_Combo_Count_Prior',
        'Fail_Count_10Min', 'Fail_Count_1Hour',
        'Time_Since_Last_Login_Sec', 'User_Mean_Time_Between_Logins_Prior',
        'Time_Gap_Deviation_Ratio'
    ]
    
    # Filter available
    cols = [c for c in suspects if c in df.columns]
    stats = df[cols].describe(percentiles=[0.5, 0.95, 0.99]).T
    print(f"\n--- Feature Stats: {stage_name} ---")
    print(stats[['min', '50%', '95%', '99%', 'max']].to_string())
    return stats

def run_diagnostic(csv_path, chunksize=100000, target_chunks=[0, 25, 49]):
    beh_engine = BehavioralFeatureEngine()
    graph_engine = GraphFeatureEngine()
    
    print(f"Starting Diagnostic on {csv_path}...")
    chunks = pd.read_csv(
        csv_path, 
        chunksize=chunksize, 
        parse_dates=['Login Timestamp'],
        dtype={'User ID': str, 'Country': str, 'ASN': str}
    )
    
    all_stats = {}
    
    for i, chunk in enumerate(chunks):
        # 1. Pre-process
        # chunk = chunk.sort_values('Login Timestamp') # Assuming chronological for debug
        chunk_proc = chunk.rename(columns={
            'User ID': 'User_ID',
            'Login Timestamp': 'Login_Timestamp',
            'Login Successful': 'Login_Successful'
        })
        chunk_proc['Login_Successful'] = chunk_proc['Login_Successful'].fillna(True).astype(int)
        
        # 2. Features
        chunk_beh = beh_engine.process_chunk(chunk_proc)
        chunk_graph = graph_engine.process_chunk(chunk_beh)
        
        # 3. Analyze specific chunks
        if i in target_chunks:
            stage = f"Chunk {i} ({i*chunksize/1e6:.1f}M rows)"
            all_stats[i] = analyze_chunk(chunk_graph, stage)
            
        if i > max(target_chunks):
            break
            
        print(f"[{datetime.now()}] Finished chunk {i}...")

    return all_stats

if __name__ == "__main__":
    csv_path = 'data/rba-dataset.csv'
    if os.path.exists(csv_path):
        run_diagnostic(csv_path, target_chunks=[0, 25, 49])
    else:
        print("Dataset not found at data/rba-dataset.csv")
