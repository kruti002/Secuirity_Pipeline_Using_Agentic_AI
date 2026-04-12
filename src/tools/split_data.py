import pandas as pd
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'engine'))
from features import add_behavioral_features
from graph_features import add_graph_features

def build_data_splits(data_path='data/rba-dataset.csv', sample_size=6000000):
    """
    Ultra Horizon Splitting: 6M rows with 20% test split.
    """
    needed_cols = [
        'Login Timestamp', 'User ID', 'IP Address', 'Country', 'ASN',
        'Device Type', 'OS Name and Version', 'Browser Name and Version',
        'Login Successful', 'Is Account Takeover',
        'Region', 'City', 'User Agent String', 'Round-Trip Time [ms]', 'Is Attack IP'
    ]
    
    print(f"Loading {sample_size:,} rows from raw dataset...")
    t0 = time.time()
    
    df = pd.read_csv(
        data_path, 
        nrows=sample_size,
        usecols=lambda c: c in needed_cols
    )
    
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.2f}s")
    
    # Sort chronologically — required for correct 'first time seen' tracking
    df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'])
    df = df.sort_values('Login Timestamp').reset_index(drop=True)

    os.makedirs('data/intermediate', exist_ok=True)
    os.makedirs('data/splits', exist_ok=True)

    # 1. GRAPH FEATURES FIRST (To ensure clean column naming / state)
    print("\n[1/2] Building graph infrastructure features...")
    t1 = time.time()
    df = add_graph_features(df)
    print(f"      Graph features done in {time.time()-t1:.2f}s")

    # 2. BEHAVIORAL FEATURES (novelty, recency, stabilized counts)
    print("\n[2/2] Building behavioral features...")
    t2 = time.time()
    df = add_behavioral_features(df)
    print(f"      Behavioral features done in {time.time()-t2:.2f}s")

    # Time-Base 70/10/20 Split for 6M rows
    print("\n[3/3] Building 70/10/20 temporal evaluation splits...")
    
    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.80)

    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    print(f"\nSaving temporal splits to data/splits/")
    print(f"  Train : {len(train_df):>7,} rows  ({int(train_df['Is Account Takeover'].sum())} ATOs)")
    print(f"  Val   : {len(val_df):>7,} rows  ({int(val_df['Is Account Takeover'].sum())} ATOs)")
    print(f"  Test  : {len(test_df):>7,} rows  ({int(test_df['Is Account Takeover'].sum())} ATOs)")

    train_df.to_csv('data/splits/train.csv', index=False)
    val_df.to_csv('data/splits/val.csv',   index=False)
    test_df.to_csv('data/splits/test.csv',  index=False)

    print("\nDone. Stabilized splits saved.")

if __name__ == "__main__":
    build_data_splits()
