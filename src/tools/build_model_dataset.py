import pandas as pd
import os
from sklearn.model_selection import train_test_split

def build_modeling_dataset(data_path='data/intermediate/behavioral_graph.csv'):
    print(f"Loading fully generated features from {data_path}...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}. Run split_data.py first.")
        
    df_full = pd.read_csv(data_path)
    
    # ---------------------------------------------------------------------------
    # EXTRACT SUPERVISED MODELING DATASET
    # ---------------------------------------------------------------------------
    pos_df = df_full[df_full['Is Account Takeover'] == 1]
    neg_df = df_full[df_full['Is Account Takeover'] == 0]
    
    print(f"Total Positives (ATO): {len(pos_df)}")
    print(f"Total Negatives (Benign): {len(neg_df)}")
    
    # Stratify by sampling Negatives at an imbalanced but learnable ratio (50x)
    desired_neg = min(len(pos_df) * 50, len(neg_df))
    # Safety net: ensure there is enough baseline for meaningful negative classification
    if desired_neg < 50000:
        desired_neg = min(50000, len(neg_df))
        
    print(f"Sampling {desired_neg} negative events to create a class-balanced modeling baseline...")
    neg_sample = neg_df.sample(n=desired_neg, random_state=42)
    
    df_model = pd.concat([pos_df, neg_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("Splitting into localized Train/Val sequence for supervised learning operations...")
    train_model, val_model = train_test_split(
        df_model,
        test_size=0.2,
        stratify=df_model['Is Account Takeover'],
        random_state=42
    )

    os.makedirs('data/splits', exist_ok=True)
    
    print("\nSaving stratified learning splits to data/splits/")
    print(f"  Train (Supervised) : {len(train_model):>7,} rows  ({train_model['Is Account Takeover'].sum()} ATOs)")
    print(f"  Val (Supervised)   : {len(val_model):>7,} rows  ({val_model['Is Account Takeover'].sum()} ATOs)")

    train_model.to_csv('data/splits/train_model.csv', index=False)
    val_model.to_csv('data/splits/val_model.csv', index=False)
    print("\nDone! Stratified Supervised Modeling sets successfully separated from time-realism test sequence.")

if __name__ == '__main__':
    build_modeling_dataset()
