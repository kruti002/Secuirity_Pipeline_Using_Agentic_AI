from collections import defaultdict, deque
import pandas as pd
import ipaddress

# ---------------------------------------------------------------------------
# MAIN BEHAVIORAL ENGINE
# Must run over the FULL chronological dataset BEFORE splitting.
# Every feature is computed strictly from prior events only.
# ---------------------------------------------------------------------------

class BehavioralFeatureEngine:
    """
    Maintains chronological state for behavioral feature engineering.
    Can process data in chunks while preserving user history.
    """
    def __init__(self):
        # ---- Stateful dictionaries (keyed by user) ----
        self.user_seen_countries   = defaultdict(set)
        self.user_seen_asns        = defaultdict(set)
        self.user_seen_subnets     = defaultdict(set)
        self.user_seen_devices     = defaultdict(set)
        
        self.user_seen_country_dev = defaultdict(set)
        self.user_seen_asn_dev     = defaultdict(set)
        self.user_seen_subnet_dev  = defaultdict(set)
        self.user_seen_country_asn = defaultdict(set)

        self.user_last_country_ts      = defaultdict(lambda: None)
        self.user_last_asn_ts          = defaultdict(lambda: None)
        self.user_last_subnet_ts       = defaultdict(lambda: None)
        self.user_last_device_ts       = defaultdict(lambda: None)
        self.user_last_country_dev_ts  = defaultdict(lambda: None)
        self.user_last_asn_dev_ts      = defaultdict(lambda: None)
        self.user_last_subnet_dev_ts   = defaultdict(lambda: None)

        self.user_login_sum_hours = defaultdict(float)
        self.user_first_login_ts  = defaultdict(lambda: None)
        self.user_last_login_ts   = defaultdict(lambda: None)
        self.user_login_count     = defaultdict(int)
        self.user_fail_count      = defaultdict(int)
        self.user_success_count   = defaultdict(int)
        self.user_consec_fails    = defaultdict(int)
        self.user_weekend_logins  = defaultdict(int)

        self.user_event_log_10m    = defaultdict(deque)
        self.user_event_log_1h     = defaultdict(deque)
        self.user_fail_10m         = defaultdict(int)
        self.user_succ_10m         = defaultdict(int)
        self.user_fail_1h          = defaultdict(int)

        self.user_entity_log_10m   = defaultdict(deque)
        self.user_entity_log_1h    = defaultdict(deque)
        self.user_entity_10m_count = defaultdict(int)
        self.user_entity_1h_subnet = defaultdict(int)
        self.user_entity_1h_device = defaultdict(int)

    def process_chunk(self, df):
        import math
        
        # Prepare helper columns
        if 'IP Address' in df.columns:
            df['IP_Subnet'] = df['IP Address'].astype(str).apply(
                lambda x: '.'.join(x.split('.')[:3]) if '.' in x else 'Unknown'
            )
        else:
            df['IP_Subnet'] = 'Unknown'

        combo_cols = [c for c in ['Device Type', 'OS Name and Version', 'Browser Name and Version'] if c in df.columns]
        if combo_cols:
            df['Device_Combo'] = df[combo_cols].astype(str).agg('-'.join, axis=1)
        else:
            df['Device_Combo'] = 'Unknown'

        rows = []
        for row in df.itertuples(index=True):
            user    = getattr(row, 'User_ID', 'Unknown')
            country = str(getattr(row, 'Country', 'Unknown'))
            asn     = str(getattr(row, 'ASN', 'Unknown'))
            subnet  = getattr(row, 'IP_Subnet', 'Unknown')
            device  = getattr(row, 'Device_Combo', 'Unknown')
            success = int(getattr(row, 'Login_Successful', 1)) if pd.notna(getattr(row, 'Login_Successful', 1)) else 1
            ts      = getattr(row, 'Login_Timestamp', None)
            
            if ts is None: continue
            now_sec = ts.timestamp()
            hour    = ts.hour
            dow     = ts.dayofweek
            is_weekend = int(dow >= 5)

            feat = {}
            total_prior = self.user_login_count[user]

            feat['Missing_Country']     = int(pd.isna(getattr(row, 'Country', None)))
            feat['Missing_ASN']         = int(pd.isna(getattr(row, 'ASN', None)))
            feat['Missing_Device_Info'] = 0 
            feat['User_Login_Count_Prior'] = math.log1p(total_prior)
            feat['Login_Hour_Sin'] = math.sin(2 * math.pi * hour / 24)
            feat['Login_Hour_Cos'] = math.cos(2 * math.pi * hour / 24)
            feat['Login_Day_Of_Week'] = dow
            feat['Is_Weekend']        = is_weekend
            feat['User_Weekend_Login_Rate_Prior'] = (self.user_weekend_logins[user]/total_prior if total_prior > 0 else 0.0)
            feat['Weekend_Behavior_Deviation'] = int((is_weekend == 1 and feat['User_Weekend_Login_Rate_Prior'] < 0.1) or (is_weekend == 0 and feat['User_Weekend_Login_Rate_Prior'] > 0.9))

            feat['First_Time_Country']      = int(country not in self.user_seen_countries[user])
            feat['First_Time_ASN']          = int(asn     not in self.user_seen_asns[user])
            feat['First_Time_Subnet']       = int(subnet  not in self.user_seen_subnets[user])
            feat['First_Time_Device_Combo'] = int(device  not in self.user_seen_devices[user])

            feat['User_Country_Count_Prior']      = math.log1p(len(self.user_seen_countries[user]))
            feat['User_ASN_Count_Prior']          = math.log1p(len(self.user_seen_asns[user]))
            feat['User_Subnet_Count_Prior']       = math.log1p(len(self.user_seen_subnets[user]))
            feat['User_Device_Combo_Count_Prior'] = math.log1p(len(self.user_seen_devices[user]))

            def secs_since(last_ts_sec): return max(0.0, now_sec - last_ts_sec) if last_ts_sec is not None else 0.0

            first_ts = self.user_first_login_ts[user]
            last_ts  = self.user_last_login_ts[user]
            
            mean_gap = (last_ts - first_ts) / (total_prior - 1) if total_prior >= 2 else 1.0
            if mean_gap == 0: mean_gap = 1.0

            feat['Seen_Country_Before'] = int(self.user_last_country_ts[(user, country)] is not None)
            feat['Secs_Since_Last_Country_For_User'] = secs_since(self.user_last_country_ts[(user, country)])
            feat['Recency_Country_Normalized'] = feat['Secs_Since_Last_Country_For_User'] / mean_gap
            
            feat['Seen_ASN_Before'] = int(self.user_last_asn_ts[(user, asn)] is not None)
            feat['Secs_Since_Last_ASN_For_User'] = secs_since(self.user_last_asn_ts[(user, asn)])
            feat['Recency_ASN_Normalized'] = feat['Secs_Since_Last_ASN_For_User'] / mean_gap

            feat['Seen_Subnet_Before'] = int(self.user_last_subnet_ts[(user, subnet)] is not None)
            feat['Secs_Since_Last_Subnet_For_User'] = secs_since(self.user_last_subnet_ts[(user, subnet)])
            feat['Recency_Subnet_Normalized'] = feat['Secs_Since_Last_Subnet_For_User'] / mean_gap

            feat['Seen_Device_Before'] = int(self.user_last_device_ts[(user, device)] is not None)
            feat['Secs_Since_Last_Device_For_User'] = secs_since(self.user_last_device_ts[(user, device)])
            feat['Recency_Device_Normalized'] = feat['Secs_Since_Last_Device_For_User'] / mean_gap

            feat['Seen_Country_Device_Before'] = int(self.user_last_country_dev_ts[(user, country, device)] is not None)
            feat['Secs_Since_Last_Country_Device_For_User'] = secs_since(self.user_last_country_dev_ts[(user, country, device)])
            feat['Seen_ASN_Device_Before'] = int(self.user_last_asn_dev_ts[(user, asn, device)] is not None)
            feat['Secs_Since_Last_ASN_Device_For_User'] = secs_since(self.user_last_asn_dev_ts[(user, asn, device)])
            feat['Seen_Subnet_Device_Before'] = int(self.user_last_subnet_dev_ts[(user, subnet, device)] is not None)
            feat['Secs_Since_Last_Subnet_Device_For_User'] = secs_since(self.user_last_subnet_dev_ts[(user, subnet, device)])

            feat['First_Time_Country_Device_Combo'] = int((country, device) not in self.user_seen_country_dev[user])
            feat['First_Time_ASN_Device_Combo']     = int((asn, device)     not in self.user_seen_asn_dev[user])
            feat['First_Time_Subnet_Device_Combo']  = int((subnet, device)  not in self.user_seen_subnet_dev[user])
            feat['First_Time_Country_ASN']          = int((country, asn)    not in self.user_seen_country_asn[user])

            if total_prior >= 2:
                m_hour = self.user_login_sum_hours[user] / total_prior
                feat['User_Mean_Login_Hour_Prior'] = m_hour
                rdiff = abs(hour - m_hour)
                feat['User_Login_Hour_Deviation'] = min(rdiff, 24 - rdiff)
            else:
                feat['User_Mean_Login_Hour_Prior'] = float(hour)
                feat['User_Login_Hour_Deviation'] = 0.0

            if total_prior >= 2:
                feat['User_Mean_Time_Between_Logins_Prior'] = min(mean_gap, 100.0) # CLIP
                feat['Time_Since_Last_Login_Sec'] = (now_sec - last_ts)
                # CLIP HUGE DRIFT
                feat['Time_Gap_Deviation_Ratio'] = min(feat['Time_Since_Last_Login_Sec'] / mean_gap, 100.0) 
            else:
                feat['User_Mean_Time_Between_Logins_Prior'] = 0.0
                feat['Time_Since_Last_Login_Sec'] = (now_sec - last_ts) if last_ts is not None else 0.0
                feat['Time_Gap_Deviation_Ratio'] = 0.0

            feat['User_Fail_Count_Prior']    = self.user_fail_count[user]
            feat['User_Success_Count_Prior'] = self.user_success_count[user]
            feat['User_Failure_Rate_Prior']  = (self.user_fail_count[user] / total_prior if total_prior > 0 else 0.0)
            feat['Consecutive_Failures_Prior']    = self.user_consec_fails[user]
            feat['Success_After_Consecutive_Fails'] = int(success == 1 and self.user_consec_fails[user] >= 3)

            ev_10m = self.user_event_log_10m[user]
            while ev_10m and (now_sec - ev_10m[0][0]) > 600:
                _, rf = ev_10m.popleft(); 
                if rf == 1: self.user_fail_10m[user] -= 1
                else: self.user_succ_10m[user] -= 1
            ev_1h = self.user_event_log_1h[user]
            while ev_1h and (now_sec - ev_1h[0][0]) > 3600:
                _, rf = ev_1h.popleft(); 
                if rf == 1: self.user_fail_1h[user] -= 1

            f10 = self.user_fail_10m[user]
            s10 = self.user_succ_10m[user]
            feat['Fail_Count_10Min'] = f10
            feat['Fail_Count_1Hour'] = self.user_fail_1h[user]
            feat['Success_Count_10Min'] = s10
            feat['Recent_Fail_Burst'] = f10
            feat['Success_After_Fail_Burst'] = int(success == 1 and f10 >= 3)
            feat['Recent_Fail_Success_Ratio'] = f10 / (f10 + s10) if (f10 + s10) > 0 else 0.0

            et10 = self.user_entity_log_10m[user]
            while et10 and (now_sec - et10[0][0]) > 600:
                et10.popleft(); self.user_entity_10m_count[user] -= 1
            et1h = self.user_entity_log_1h[user]
            while et1h and (now_sec - et1h[0][0]) > 3600:
                _, re = et1h.popleft()
                if re == 'subnet': self.user_entity_1h_subnet[user] -= 1
                if re == 'device': self.user_entity_1h_device[user] -= 1

            feat['New_Entities_10Min'] = self.user_entity_10m_count[user]
            feat['New_Entities_1Hour'] = len(et1h)
            feat['New_Subnets_1Hour'] = self.user_entity_1h_subnet[user]
            feat['New_Device_Combos_1Hour'] = self.user_entity_1h_device[user]

            # Rate counts: clip denominator to prevent dilution on mature users
            rate_base = min(total_prior, 100.0) 
            n = max(1, rate_base)
            feat['Country_Change_Rate_Prior'] = len(self.user_seen_countries[user]) / n
            feat['ASN_Change_Rate_Prior']     = len(self.user_seen_asns[user]) / n
            feat['Subnet_Change_Rate_Prior']  = len(self.user_seen_subnets[user]) / n
            feat['Device_Change_Rate_Prior']  = len(self.user_seen_devices[user]) / n

            mature = int(total_prior >= 20)
            feat['Mature_Account_New_Country'] = mature & feat['First_Time_Country']
            feat['Mature_Account_New_Device'] = mature & feat['First_Time_Device_Combo']
            feat['Mature_Account_Multi_Entity_Drift'] = int(mature and (feat['First_Time_Country'] + feat['First_Time_ASN'] + feat['First_Time_Subnet'] + feat['First_Time_Device_Combo']) >= 2)

            feat['New_Device_And_New_Subnet'] = feat['First_Time_Device_Combo'] & feat['First_Time_Subnet']
            feat['New_Country_And_Fail_Burst'] = feat['First_Time_Country'] & int(f10 >= 3)
            feat['New_ASN_And_Unusual_Hour'] = feat['First_Time_ASN'] & int(feat['User_Login_Hour_Deviation'] > 4)
            feat['Success_After_Fail_And_New_Device'] = feat['Success_After_Fail_Burst'] & feat['First_Time_Device_Combo']
            
            rows.append(feat)

            # UPDATE STATE
            if feat['First_Time_Subnet']: self.user_entity_log_10m[user].append((now_sec, 'subnet')); self.user_entity_log_1h[user].append((now_sec, 'subnet')); self.user_entity_10m_count[user] += 1; self.user_entity_1h_subnet[user] += 1
            if feat['First_Time_Device_Combo']: self.user_entity_log_10m[user].append((now_sec, 'device')); self.user_entity_log_1h[user].append((now_sec, 'device')); self.user_entity_10m_count[user] += 1; self.user_entity_1h_device[user] += 1
            if feat['First_Time_Country']: self.user_entity_log_10m[user].append((now_sec, 'country')); self.user_entity_log_1h[user].append((now_sec, 'country')); self.user_entity_10m_count[user] += 1
            if feat['First_Time_ASN']: self.user_entity_log_10m[user].append((now_sec, 'asn')); self.user_entity_log_1h[user].append((now_sec, 'asn')); self.user_entity_10m_count[user] += 1

            self.user_seen_countries[user].add(country); self.user_seen_asns[user].add(asn); self.user_seen_subnets[user].add(subnet); self.user_seen_devices[user].add(device)
            self.user_seen_country_dev[user].add((country, device)); self.user_seen_asn_dev[user].add((asn, device)); self.user_seen_subnet_dev[user].add((subnet, device)); self.user_seen_country_asn[user].add((country, asn))
            self.user_last_country_ts[(user, country)] = now_sec; self.user_last_asn_ts[(user, asn)] = now_sec; self.user_last_subnet_ts[(user, subnet)] = now_sec; self.user_last_device_ts[(user, device)] = now_sec
            self.user_last_country_dev_ts[(user, country, device)] = now_sec; self.user_last_asn_dev_ts[(user, asn, device)] = now_sec; self.user_last_subnet_dev_ts[(user, subnet, device)] = now_sec
            
            self.user_login_sum_hours[user] += hour
            if self.user_first_login_ts[user] is None:
                self.user_first_login_ts[user] = now_sec
            self.user_last_login_ts[user] = now_sec
            self.user_login_count[user] += 1; self.user_weekend_logins[user] += is_weekend
            if success == 0: self.user_fail_10m[user] += 1; self.user_fail_1h[user] += 1; self.user_fail_count[user] += 1; self.user_consec_fails[user] += 1
            else: self.user_succ_10m[user] += 1; self.user_success_count[user] += 1; self.user_consec_fails[user] = 0
            self.user_event_log_10m[user].append((now_sec, 1 if success==0 else 0)); self.user_event_log_1h[user].append((now_sec, 1 if success==0 else 0))

        feat_df = pd.DataFrame(rows)
        return pd.concat([df.reset_index(drop=True), feat_df], axis=1)

def add_behavioral_features(df):
    """Legacy wrapper for the new stateful engine."""
    engine = BehavioralFeatureEngine()
    # Rename for itertuples safety
    df = df.rename(columns={'User ID': 'User_ID', 'Login Timestamp': 'Login_Timestamp', 'Login Successful': 'Login_Successful'})
    return engine.process_chunk(df)

def add_base_features(df):
    df = df.copy()
    if 'Login Timestamp' in df.columns:
        df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'])
        df['Login Hour'] = df['Login Timestamp'].dt.hour
    return df

def drop_unneeded_columns_for_ml(df):
    drop_cols = [
        'IP Address', 'Region', 'City', 'Login Timestamp', 'Login_Timestamp',
        'User Agent String', 'Round-Trip Time [ms]', 'User ID', 'User_ID',
        'Is Attack IP', 'IP_Subnet', 'Device_Combo', 'ASN',
    ]
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
