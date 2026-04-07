import pandas as pd
import numpy as np

def apply_rules(df):
    """
    Apply security heuristics as specified by the SOC requirements.
    This creates an explicit risk signal and reason codes.
    """
    df = df.copy()
    df['rule_score'] = 0.0
    df['reason_codes'] = ""
    
    # Sort for history-based rules
    df = df.sort_values(by=['User ID', 'Login Timestamp'])
    
    # 1. Attack IP + Success
    mask1 = (df['Is Attack IP'] == True) & (df['Login Successful'] == True)
    df.loc[mask1, 'rule_score'] += 0.8
    df.loc[mask1, 'reason_codes'] += "CRITICAL: Success from Attack IP; "
    
    # 2. New Country + Success (Simplified: user has less than 2 distinct countries in history)
    # This would be better if we compared against *their* previous countries.
    user_country_counts = df.groupby('User ID')['Country'].transform('nunique')
    mask2 = (user_country_counts > 1) & (df['Login Successful'] == True) # This is a weak rule for now
    # More robust: check if this country matches the user's "dominant" country
    
    # 3. Failed -> Success (Brute-force)
    # Check if the previous event for this user was a Failure
    df['prev_success'] = df.groupby('User ID')['Login Successful'].shift(1)
    df['prev_time'] = df.groupby('User ID')['Login Timestamp'].shift(1)
    time_diff = (pd.to_datetime(df['Login Timestamp']) - pd.to_datetime(df['prev_time'])).dt.total_seconds()
    mask3 = (df['Login Successful'] == True) & (df['prev_success'] == False) & (time_diff < 300) # within 5 mins
    df.loc[mask3, 'rule_score'] += 0.5
    df.loc[mask3, 'reason_codes'] += "HIGH: Possible Brute-force (Fail->Success); "
    
    # 4. Off-hours + New Device
    # Off-hours: 11 PM to 5 AM
    df['is_off_hours'] = pd.to_datetime(df['Login Timestamp']).dt.hour.apply(lambda h: h < 5 or h > 22)
    # Device rule: simplified as "Device Type is bot or unknown" or novelty
    mask4 = (df['is_off_hours'] == True) & (df['Device Type'].isin(['bot', 'unknown']))
    df.loc[mask4, 'rule_score'] += 0.4
    df.loc[mask4, 'reason_codes'] += "MEDIUM: Off-hours login from suspicious device; "
    
    # Cap rule_score at 1.0 safely
    df['rule_score'] = df['rule_score'].clip(0, 1)
    
    return df

if __name__ == "__main__":
    # Test on existing data
    import os
    data_path = 'data/rba-dataset.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df_rules = apply_rules(df)
        print("Rule engine sample output:")
        print(df_rules[df_rules['rule_score'] > 0][['User ID', 'IP Address', 'rule_score', 'reason_codes']].head())
