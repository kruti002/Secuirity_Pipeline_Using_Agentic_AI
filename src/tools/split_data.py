import pandas as pd
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'engine'))
from features import add_behavioral_features
from graph_features import add_graph_features

def build_data_splits(data_path='data/rba-dataset.csv', sample_size=1000000):
    """
    sample_size=1000000 captures a wide behavioral baseline and enough ATO interactions.
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

    behavioral_path = 'data/intermediate/behavioral.csv'
    graph_path = 'data/intermediate/behavioral_graph.csv'

    # 3. BEHAVIORAL FEATURES (novelty, recency, rolling windows)
    if os.path.exists(behavioral_path):
        print("\n[1/2] Loading cached behavioral features...")
        df = pd.read_csv(behavioral_path)
        df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'])
    else:
        print("\n[1/2] Building behavioral features...")
        t1 = time.time()
        df = add_behavioral_features(df)
        print(f"      Behavioral features done in {time.time()-t1:.2f}s")
        df.to_csv(behavioral_path, index=False)

    # 4. GRAPH FEATURES (infrastructure pressure, pairwise novelty)
    if os.path.exists(graph_path):
        print("\n[2/2] Loading cached graph features...")
        df = pd.read_csv(graph_path)
        df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'])
    else:
        print("\n[2/2] Building graph infrastructure features...")
        t2 = time.time()
        df = add_graph_features(df)
        print(f"      Graph features done in {time.time()-t2:.2f}s")
        df.to_csv(graph_path, index=False)

    # Path B: Time-Realism Chronological Splitting
    print("\n[3/3] Building temporal evaluation splits...")
    
    pos_idx = df[df['Is Account Takeover'] == 1].index
    
    if len(pos_idx) < 3:
        print("WARNING: Not enough ATO occurrences to populate Train/Val/Test! Increasing sample_size or using Path A is required.")
        train_end = int(len(df) * 0.70)
        val_end   = int(len(df) * 0.85)
        train_df = df.iloc[:train_end]
        val_df   = df.iloc[train_end:val_end]
        test_df  = df.iloc[val_end:]
    else:
        # Force the splits to fall exactly around the ATO timestamps to guarantee distribution
        n_pos = len(pos_idx)
        train_bound = pos_idx[int(n_pos * 0.70)]
        val_bound   = pos_idx[int(n_pos * 0.85)]

        train_df = df.iloc[:train_bound]
        val_df   = df.iloc[train_bound:val_bound]
        test_df  = df.iloc[val_bound:]

    print(f"\nSaving temporal splits to data/splits/")
    print(f"  Train : {len(train_df):>7,} rows  ({train_df['Is Account Takeover'].sum()} ATOs)")
    print(f"  Val   : {len(val_df):>7,} rows  ({val_df['Is Account Takeover'].sum()} ATOs)")
    print(f"  Test  : {len(test_df):>7,} rows  ({test_df['Is Account Takeover'].sum()} ATOs)")

    train_df.to_csv('data/splits/train.csv', index=False)
    val_df.to_csv('data/splits/val.csv',   index=False)
    test_df.to_csv('data/splits/test.csv',  index=False)

    print("\nDone. Time-aware splits saved.")

if __name__ == "__main__":
    build_data_splits()

