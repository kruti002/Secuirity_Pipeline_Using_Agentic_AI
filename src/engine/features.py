from collections import defaultdict, deque
import pandas as pd
import ipaddress

# ---------------------------------------------------------------------------
# MAIN BEHAVIORAL ENGINE
# Must run over the FULL chronological dataset BEFORE splitting.
# Every feature is computed strictly from prior events only.
# ---------------------------------------------------------------------------

def add_behavioral_features(df):
    df = df.copy()

    if 'Login Timestamp' not in df.columns or 'User ID' not in df.columns:
        return df

    df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'])
    df = df.sort_values('Login Timestamp').reset_index(drop=True)

    # ---- Build entity combo columns first ----
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

    # ---- Stateful dictionaries (keyed by user) ----
    # Novelty sets
    user_seen_countries   = defaultdict(set)
    user_seen_asns        = defaultdict(set)
    user_seen_subnets     = defaultdict(set)
    user_seen_devices     = defaultdict(set)
    # Pairwise novelty sets
    user_seen_country_dev = defaultdict(set)
    user_seen_asn_dev     = defaultdict(set)
    user_seen_subnet_dev  = defaultdict(set)
    user_seen_country_asn = defaultdict(set)

    # Recency: last-seen timestamp per (user, entity) and pairwise
    user_last_country_ts      = defaultdict(lambda: None)
    user_last_asn_ts          = defaultdict(lambda: None)
    user_last_subnet_ts       = defaultdict(lambda: None)
    user_last_device_ts       = defaultdict(lambda: None)
    user_last_country_dev_ts  = defaultdict(lambda: None)
    user_last_asn_dev_ts      = defaultdict(lambda: None)
    user_last_subnet_dev_ts   = defaultdict(lambda: None)

    # Login-hour history (list for mean)
    user_login_hours      = defaultdict(list)
    # Login timestamps (for time-gap baseline)
    user_login_times      = defaultdict(list)

    # Account maturity
    user_login_count      = defaultdict(int)

    # Failure / success totals for per-user baseline
    user_fail_count       = defaultdict(int)
    user_success_count    = defaultdict(int)

    # Consecutive failure streak tracker
    user_consec_fails     = defaultdict(int)

    # Weekend login count
    user_weekend_logins   = defaultdict(int)

    # Rolling burst tracking
    user_event_log_10m    = defaultdict(deque)
    user_event_log_1h     = defaultdict(deque)
    user_fail_10m         = defaultdict(int)
    user_succ_10m         = defaultdict(int)
    user_fail_1h          = defaultdict(int)

    # Entity burst tracking
    user_entity_log_10m   = defaultdict(deque)
    user_entity_log_1h    = defaultdict(deque)
    user_entity_10m_count = defaultdict(int)
    user_entity_1h_subnet = defaultdict(int)
    user_entity_1h_device = defaultdict(int)

    rows = []

    import math

    for row in df.to_dict('records'):
        user    = row['User ID']
        country = str(row.get('Country', 'Unknown'))
        asn     = str(row.get('ASN', 'Unknown'))
        subnet  = row['IP_Subnet']
        device  = row['Device_Combo']
        success = int(row.get('Login Successful', 1)) if pd.notna(row.get('Login Successful')) else 1
        ts      = row['Login Timestamp']
        now_sec = ts.timestamp()
        hour    = ts.hour
        dow     = ts.dayofweek          # 0=Monday … 6=Sunday
        is_weekend = int(dow >= 5)

        feat = {}

        # ================================================================
        # MISSINGNESS FLAGS
        # ================================================================
        feat['Missing_Country']     = int(pd.isna(row.get('Country')))
        feat['Missing_ASN']         = int(pd.isna(row.get('ASN')))
        feat['Missing_Device_Info'] = int(
            all(pd.isna(row.get(c)) for c in ['Device Type', 'OS Name and Version', 'Browser Name and Version'])
        )

        # ================================================================
        # ACCOUNT MATURITY
        # ================================================================
        feat['User_Login_Count_Prior'] = user_login_count[user]

        # ================================================================
        # CIRCULAR TIME FEATURES
        # ================================================================
        feat['Login_Hour_Sin'] = math.sin(2 * math.pi * hour / 24)
        feat['Login_Hour_Cos'] = math.cos(2 * math.pi * hour / 24)
        feat['Login_Day_Of_Week'] = dow
        feat['Is_Weekend']        = is_weekend

        # Per-user weekend login rate prior
        total_prior = user_login_count[user]
        feat['User_Weekend_Login_Rate_Prior'] = (
            user_weekend_logins[user] / total_prior if total_prior > 0 else 0.0
        )
        # Is this an unusual day-type for this user?
        feat['Weekend_Behavior_Deviation'] = int(
            (is_weekend == 1 and feat['User_Weekend_Login_Rate_Prior'] < 0.1)
            or
            (is_weekend == 0 and feat['User_Weekend_Login_Rate_Prior'] > 0.9)
        )

        # ================================================================
        # NOVELTY FEATURES (binary, first-time flags)
        # ================================================================
        feat['First_Time_Country']      = int(country not in user_seen_countries[user])
        feat['First_Time_ASN']          = int(asn     not in user_seen_asns[user])
        feat['First_Time_Subnet']       = int(subnet  not in user_seen_subnets[user])
        feat['First_Time_Device_Combo'] = int(device  not in user_seen_devices[user])

        # ================================================================
        # FREQUENCY FEATURES (how many unique entities seen before)
        # ================================================================
        feat['User_Country_Count_Prior']     = len(user_seen_countries[user])
        feat['User_ASN_Count_Prior']         = len(user_seen_asns[user])
        feat['User_Subnet_Count_Prior']      = len(user_seen_subnets[user])
        feat['User_Device_Combo_Count_Prior'] = len(user_seen_devices[user])

        # ================================================================
        # RECENCY FEATURES (seconds since this entity was last used)
        # ================================================================
        # Use 0.0 for never-seen (not -1) so tree models don't treat it as a real distance.
        # Pair each with an explicit Seen_Before flag for cleaner splitting.
        def secs_since(last_ts_sec):
            if last_ts_sec is None:
                return 0.0
            return max(0.0, now_sec - last_ts_sec)

        mean_gap = feat.get('User_Mean_Time_Between_Logins_Prior', 1.0) or 1.0

        # Seen-before flags (1 = have history, 0 = first time)
        feat['Seen_Country_Before'] = int(user_last_country_ts[(user, country)] is not None)
        feat['Seen_ASN_Before']     = int(user_last_asn_ts[(user, asn)]         is not None)
        feat['Seen_Subnet_Before']  = int(user_last_subnet_ts[(user, subnet)]   is not None)
        feat['Seen_Device_Before']  = int(user_last_device_ts[(user, device)]   is not None)

        # Raw recency in seconds
        feat['Secs_Since_Last_Country_For_User'] = secs_since(user_last_country_ts[(user, country)])
        feat['Secs_Since_Last_ASN_For_User']     = secs_since(user_last_asn_ts[(user, asn)])
        feat['Secs_Since_Last_Subnet_For_User']  = secs_since(user_last_subnet_ts[(user, subnet)])
        feat['Secs_Since_Last_Device_For_User']  = secs_since(user_last_device_ts[(user, device)])

        # User-relative normalized recency (how many "typical gaps" ago was this entity last seen)
        feat['Recency_Country_Normalized'] = feat['Secs_Since_Last_Country_For_User'] / mean_gap
        feat['Recency_ASN_Normalized']     = feat['Secs_Since_Last_ASN_For_User']     / mean_gap
        feat['Recency_Subnet_Normalized']  = feat['Secs_Since_Last_Subnet_For_User']  / mean_gap
        feat['Recency_Device_Normalized']  = feat['Secs_Since_Last_Device_For_User']  / mean_gap

        # ================================================================
        # PAIRWISE RECENCY
        # ================================================================
        feat['Seen_Country_Device_Before'] = int(user_last_country_dev_ts[(user, country, device)] is not None)
        feat['Seen_ASN_Device_Before']     = int(user_last_asn_dev_ts[(user, asn, device)]         is not None)
        feat['Seen_Subnet_Device_Before']  = int(user_last_subnet_dev_ts[(user, subnet, device)]   is not None)

        feat['Secs_Since_Last_Country_Device_For_User'] = secs_since(user_last_country_dev_ts[(user, country, device)])
        feat['Secs_Since_Last_ASN_Device_For_User']     = secs_since(user_last_asn_dev_ts[(user, asn, device)])
        feat['Secs_Since_Last_Subnet_Device_For_User']  = secs_since(user_last_subnet_dev_ts[(user, subnet, device)])

        # ================================================================
        # PAIRWISE NOVELTY (combinations never seen before)
        # ================================================================
        feat['First_Time_Country_Device_Combo'] = int((country, device) not in user_seen_country_dev[user])
        feat['First_Time_ASN_Device_Combo']     = int((asn, device)     not in user_seen_asn_dev[user])
        feat['First_Time_Subnet_Device_Combo']  = int((subnet, device)  not in user_seen_subnet_dev[user])
        feat['First_Time_Country_ASN']          = int((country, asn)    not in user_seen_country_asn[user])

        # ================================================================
        # USER-RELATIVE TEMPORAL BASELINES
        # ================================================================
        prior_hours = user_login_hours[user]
        if len(prior_hours) >= 2:
            mean_hour = sum(prior_hours) / len(prior_hours)
            feat['User_Mean_Login_Hour_Prior'] = mean_hour
            # Circular distance: handles 23→01 correctly (distance=2, not 22)
            raw_diff = abs(hour - mean_hour)
            feat['User_Login_Hour_Deviation'] = min(raw_diff, 24 - raw_diff)
        else:
            feat['User_Mean_Login_Hour_Prior'] = hour
            feat['User_Login_Hour_Deviation']  = 0.0

        prior_times = user_login_times[user]
        n_prior = len(prior_times)
        if n_prior >= 2:
            # sum(gaps) mathematically telescopes to prior_times[last] - prior_times[first]
            mean_gap = (prior_times[-1] - prior_times[0]) / (n_prior - 1)
            feat['User_Mean_Time_Between_Logins_Prior'] = mean_gap
            current_gap = (now_sec - prior_times[-1])
            feat['Time_Since_Last_Login_Sec'] = current_gap
            feat['Time_Gap_Deviation_Ratio']  = (current_gap / mean_gap) if mean_gap > 0 else 0.0
        else:
            feat['User_Mean_Time_Between_Logins_Prior'] = 0.0
            feat['Time_Since_Last_Login_Sec'] = (now_sec - prior_times[-1]) if n_prior else 0.0
            feat['Time_Gap_Deviation_Ratio'] = 0.0

        # ================================================================
        # PER-USER FAILURE RATE BASELINE (all-history)
        # ================================================================
        feat['User_Fail_Count_Prior']    = user_fail_count[user]
        feat['User_Success_Count_Prior'] = user_success_count[user]
        feat['User_Failure_Rate_Prior']  = (
            user_fail_count[user] / total_prior if total_prior > 0 else 0.0
        )

        # ================================================================
        # CONSECUTIVE FAILURE STREAK
        # ================================================================
        feat['Consecutive_Failures_Prior']    = user_consec_fails[user]
        feat['Success_After_Consecutive_Fails'] = int(
            success == 1 and user_consec_fails[user] >= 3
        )

        # ================================================================
        # ROLLING ACTIVITY WINDOWS (10-min, 1-hour) O(1)
        # ================================================================
        ev_10m = user_event_log_10m[user]
        ev_1h  = user_event_log_1h[user]

        while ev_10m and (now_sec - ev_10m[0][0]) > 600:
            _, rem_f = ev_10m.popleft()
            if rem_f == 1: user_fail_10m[user] -= 1
            else:          user_succ_10m[user] -= 1

        while ev_1h and (now_sec - ev_1h[0][0]) > 3600:
            _, rem_f = ev_1h.popleft()
            if rem_f == 1: user_fail_1h[user] -= 1

        fail_10min = user_fail_10m[user]
        succ_10min = user_succ_10m[user]
        fail_1hour = user_fail_1h[user]

        feat['Fail_Count_10Min']    = fail_10min
        feat['Fail_Count_1Hour']    = fail_1hour
        feat['Success_Count_10Min'] = succ_10min
        feat['Recent_Fail_Burst']   = fail_10min   # backward-compat alias

        feat['Success_After_Fail_Burst'] = int(success == 1 and fail_10min >= 3)
        feat['Recent_Fail_Success_Ratio'] = (
            fail_10min / (fail_10min + succ_10min)
            if (fail_10min + succ_10min) > 0 else 0.0
        )

        # ================================================================
        # SHORT-WINDOW ENTITY EXPLOSION O(1)
        # ================================================================
        ent_10m = user_entity_log_10m[user]
        ent_1h  = user_entity_log_1h[user]

        while ent_10m and (now_sec - ent_10m[0][0]) > 600:
            ent_10m.popleft()
            user_entity_10m_count[user] -= 1
            
        while ent_1h and (now_sec - ent_1h[0][0]) > 3600:
            _, rem_e = ent_1h.popleft()
            if rem_e == 'subnet': user_entity_1h_subnet[user] -= 1
            if rem_e == 'device': user_entity_1h_device[user] -= 1

        feat['New_Entities_10Min'] = user_entity_10m_count[user]
        feat['New_Entities_1Hour'] = len(ent_1h)
        feat['New_Subnets_1Hour'] = user_entity_1h_subnet[user]
        feat['New_Device_Combos_1Hour'] = user_entity_1h_device[user]

        # ================================================================
        # ACCOUNT STABILITY RATES
        # ================================================================
        n = max(1, total_prior)
        feat['Country_Change_Rate_Prior'] = len(user_seen_countries[user]) / n
        feat['ASN_Change_Rate_Prior']     = len(user_seen_asns[user]) / n
        feat['Subnet_Change_Rate_Prior']  = len(user_seen_subnets[user]) / n
        feat['Device_Change_Rate_Prior']  = len(user_seen_devices[user]) / n

        # ================================================================
        # MATURITY-AWARE NOVELTY INTERACTIONS
        # ================================================================
        mature = int(total_prior >= 20)
        feat['Mature_Account_New_Country']       = mature & feat['First_Time_Country']
        feat['Mature_Account_New_Device']        = mature & feat['First_Time_Device_Combo']
        feat['Mature_Account_Multi_Entity_Drift'] = int(
            mature and (
                feat['First_Time_Country'] +
                feat['First_Time_ASN'] +
                feat['First_Time_Subnet'] +
                feat['First_Time_Device_Combo']
            ) >= 2
        )

        # ================================================================
        # INTERACTION FEATURES
        # ================================================================
        feat['New_Device_And_New_Subnet']         = feat['First_Time_Device_Combo'] & feat['First_Time_Subnet']
        feat['New_Country_And_Fail_Burst']        = feat['First_Time_Country'] & int(fail_10min >= 3)
        feat['New_ASN_And_Unusual_Hour']          = feat['First_Time_ASN'] & int(feat['User_Login_Hour_Deviation'] > 4)
        feat['Success_After_Fail_And_New_Device'] = feat['Success_After_Fail_Burst'] & feat['First_Time_Device_Combo']

        rows.append(feat)

        # ================================================================
        # UPDATE STATE (always AFTER computing features for this row)
        # ================================================================
        # Track explosion events for new entities this row
        if feat['First_Time_Subnet']:       
            user_entity_log_10m[user].append((now_sec, 'subnet'))
            user_entity_log_1h[user].append((now_sec, 'subnet'))
            user_entity_10m_count[user] += 1
            user_entity_1h_subnet[user] += 1
        if feat['First_Time_Device_Combo']: 
            user_entity_log_10m[user].append((now_sec, 'device'))
            user_entity_log_1h[user].append((now_sec, 'device'))
            user_entity_10m_count[user] += 1
            user_entity_1h_device[user] += 1
        if feat['First_Time_Country']:      
            user_entity_log_10m[user].append((now_sec, 'country'))
            user_entity_log_1h[user].append((now_sec, 'country'))
            user_entity_10m_count[user] += 1
        if feat['First_Time_ASN']:          
            user_entity_log_10m[user].append((now_sec, 'asn'))
            user_entity_log_1h[user].append((now_sec, 'asn'))
            user_entity_10m_count[user] += 1

        user_seen_countries[user].add(country)
        user_seen_asns[user].add(asn)
        user_seen_subnets[user].add(subnet)
        user_seen_devices[user].add(device)

        user_seen_country_dev[user].add((country, device))
        user_seen_asn_dev[user].add((asn, device))
        user_seen_subnet_dev[user].add((subnet, device))
        user_seen_country_asn[user].add((country, asn))

        user_last_country_ts[(user, country)]           = now_sec
        user_last_asn_ts[(user, asn)]                   = now_sec
        user_last_subnet_ts[(user, subnet)]             = now_sec
        user_last_device_ts[(user, device)]             = now_sec
        user_last_country_dev_ts[(user, country, device)] = now_sec
        user_last_asn_dev_ts[(user, asn, device)]          = now_sec
        user_last_subnet_dev_ts[(user, subnet, device)]    = now_sec

        user_login_hours[user].append(hour)
        user_login_times[user].append(now_sec)
        user_login_count[user] += 1
        user_weekend_logins[user] += is_weekend

        is_fail = 1 if success == 0 else 0
        user_event_log_10m[user].append((now_sec, is_fail))
        user_event_log_1h[user].append((now_sec, is_fail))
        if is_fail:
            user_fail_10m[user] += 1
            user_fail_1h[user]  += 1
            user_fail_count[user]    += 1
            user_consec_fails[user]  += 1
        else:
            user_succ_10m[user] += 1
            user_success_count[user] += 1
            user_consec_fails[user]   = 0   # reset on any success

    feat_df = pd.DataFrame(rows, index=df.index)
    return pd.concat([df, feat_df], axis=1)


# ---------------------------------------------------------------------------
# BASE FEATURES (called after behavioral features are already baked in)
# ---------------------------------------------------------------------------
def add_base_features(df):
    """
    Light transforms needed at inference time.
    Behavioral features must already be baked into the CSV by split_data.py.
    """
    df = df.copy()
    if 'Login Timestamp' in df.columns:
        df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'])
        df['Login Hour'] = df['Login Timestamp'].dt.hour
    if 'Login Successful' in df.columns:
        df['Login Successful'] = df['Login Successful'].astype(int)
    return df


# ---------------------------------------------------------------------------
# COLUMN DROPPER
# ---------------------------------------------------------------------------
def drop_unneeded_columns_for_ml(df):
    """
    Drops raw strings, identifiers, and leaky / duplicate columns.
    Keeps only model-ready features.
    """
    drop_cols = [
        'IP Address', 'Region', 'City', 'Login Timestamp',
        'User Agent String', 'Round-Trip Time [ms]', 'User ID',
        'Is Attack IP',      # lives in Rule Engine only
        'IP_Subnet',         # string; used via First_Time_Subnet
        'Device_Combo',      # string; used via First_Time_Device_Combo
        'ASN',               # categorical identity, not continuous
    ]
    return df.drop(columns=[c for c in drop_cols if c in df.columns])
