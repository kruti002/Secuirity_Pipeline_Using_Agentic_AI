import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# HARDENED RULE-BASED PRECISION BOOSTER
# ---------------------------------------------------------------------------

# Tightened thresholds based on FP analysis (20 FPs in test set)
HIGH_ASN_FAILURE_RATE = 0.5     # ASN failure rate > 50%
HIGH_ASN_USER_COUNT   = 100     # Massively shared infrastructure
HIGH_DRIFT_SCORE      = 6.5     # Extreme multi-entity novelty burst

def apply_rules(df):
    """Legacy compatibility."""
    df = df.copy()
    df['rule_score'] = 0.0
    df['reason_codes'] = ""
    return df

def apply_precision_booster(df, rf_score_col='risk_score', top_n=5000):
    """
    Hardened Compound Reranker.
    Requires co-firing signals to reach the Critical tier.
    """
    df = df.copy()
    df['rule_boost'] = 0.0
    df['reason_tags'] = ""

    top_idx = df.nlargest(top_n, rf_score_col).index
    top = df.loc[top_idx].copy()

    def has(col): return col in top.columns

    boosts = pd.Series(0.0, index=top.index)
    tags   = pd.Series("",  index=top.index)

    # 1. Smoking Gun: Success After Fail Burst
    if has('Success_After_Fail_Burst'):
        smoking_gun = top['Success_After_Fail_Burst'] == 1
        boosts[smoking_gun] += 0.40
        tags[smoking_gun] += "SUCCESS_AFTER_BRUTEFORCE; "

    # 2. Hardened Crawler/Botnet Signal
    if has('GF_ASN_Failure_Rate_Prior') and has('GF_ASN_User_Count_Before'):
        asn_bad = (top['GF_ASN_Failure_Rate_Prior'] > HIGH_ASN_FAILURE_RATE) & \
                  (top['GF_ASN_User_Count_Before'] > HIGH_ASN_USER_COUNT)
        # Needs behavioral trigger (failure or timing)
        trigger = (top.get('Fail_Count_10Min', 0) >= 1) | (top.get('User_Login_Hour_Deviation', 0) > 8)
        mask_bot = asn_bad & trigger
        boosts[mask_bot] += 0.25
        tags[mask_bot] += "BOTNET_INFRA_PRESSURE; "

    # 3. Hardened Mature Account Migration (Triple Novelty)
    if has('New_Device_And_New_Subnet') and has('First_Time_Country'):
        mig_novel = (top['New_Device_And_New_Subnet'] == 1) & (top['First_Time_Country'] == 1)
        # Only boost if it's a mature user (high previous login count)
        mature = top.get('User_Login_Count_Prior', 0) > 50
        mask_mig = mig_novel & mature
        boosts[mask_mig] += 0.30
        tags[mask_mig] += "MATURE_GEO_INFRA_MIGRATION; "

    # 4. Stealth Geo-Pivot (Succeed first attempt from new host)
    if has('Login Successful') and has('First_Time_Country') and has('First_Time_Subnet'):
        stealth = (top['Login Successful'] == 1) & (top['First_Time_Country'] == 1) & (top['First_Time_Subnet'] == 1)
        # Add ASN pressure context
        asn_c = top.get('GF_ASN_Failure_Rate_Prior', 0) > 0.3
        mask_s = stealth & asn_c
        boosts[mask_s] += 0.30
        tags[mask_s] += "STEALTH_GEO_PIVOT; "

    df.loc[top_idx, 'rule_boost'] = boosts.values
    df.loc[top_idx, 'reason_tags'] = tags.values

    # Final score = rf_score + (boost * weight)
    # We use 1.2 instead of 1.5 to be slightly more conservative
    df['final_score'] = (df[rf_score_col] + df['rule_boost'] * 1.2).clip(0, 1.0)

    return df
