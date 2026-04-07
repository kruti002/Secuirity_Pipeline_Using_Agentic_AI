import pandas as pd
import numpy as np

def apply_rule_engine(df):
    """
    Applies logic-based SOC rules to generate a deterministic Risk Score [0, 1].
    Rules catch known attack patterns that might be under-represented in ML training.
    """
    df = df.copy()
    rule_scores = []
    reasons = []
    
    for row in df.itertuples(index=False):
        score = 0.0
        row_reasons = []
        
        # Rule 1: Infrastructure Pressure (Burst of users on one subnet)
        if getattr(row, 'GF_Subnet_User_Count_10Min', 0) > np.log1p(10):
            score += 0.4
            row_reasons.append("High Subnet Pressure (10m)")
            
        # Rule 2: Device Reuse Across Users
        if getattr(row, 'GF_Device_User_Count_10Min', 0) > np.log1p(5):
            score += 0.3
            row_reasons.append("Shared Device Anomaly")
            
        # Rule 3: Success after multiple failures (Brute force success)
        if getattr(row, 'Success_After_Fail_Burst', 0) == 1:
            score += 0.6
            row_reasons.append("Post-Burst Login Success")
            
        # Rule 4: Mature Account Drift (Drastic shift for a known user)
        if getattr(row, 'Mature_Account_Multi_Entity_Drift', 0) == 1:
            score += 0.4
            row_reasons.append("High-Confidence Account Drift")

        # Rule 5: Failed burst + New Country
        if getattr(row, 'New_Country_And_Fail_Burst', 0) == 1:
            score += 0.5
            row_reasons.append("Cross-Border Brute Force")

        # Rule 6: Known Bad Actor (Simulated via Is Attack IP if present)
        if getattr(row, 'Is Attack IP', False) == True:
            score += 0.9
            row_reasons.append("Known Malicious Infrastructure")

        final_score = min(1.0, score)
        rule_scores.append(final_score)
        reasons.append("; ".join(row_reasons) if row_reasons else "Baseline")
        
    df['rule_score'] = rule_scores
    df['rule_reasons'] = reasons
    return df
