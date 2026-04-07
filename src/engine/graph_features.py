from collections import defaultdict, deque
import pandas as pd
import numpy as np

class UniqueWindow:
    def __init__(self, max_time):
        self.max_time = max_time
        self.q = deque()
        self.counts = defaultdict(int)
        self.unique_count = 0
        
    def trim(self, now_sec):
        while self.q and (now_sec - self.q[0][0]) > self.max_time:
            _, old_val = self.q.popleft()
            self.counts[old_val] -= 1
            if self.counts[old_val] == 0:
                del self.counts[old_val]
                self.unique_count -= 1
                
    def add(self, now_sec, val):
        self.q.append((now_sec, val))
        if self.counts[val] == 0:
            self.unique_count += 1
        self.counts[val] += 1

def add_graph_features(df):
    """
    Simulates a dynamic Interaction Graph by extracting evolving 
    relational features structurally over time. Includes paired novelty
    and rolling architectural pressures.
    """
    df = df.copy()
    
    # Must sort by time to build historical state correctly
    df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'])
    df = df.sort_values('Login Timestamp').reset_index(drop=True)

    # Rename columns to ensure itertuples works reliably with snake_case
    df = df.rename(columns={
        'User ID': 'User_ID',
        'Login Timestamp': 'Login_Timestamp',
        'Login Successful': 'Login_Successful'
    })

    user_seen_subnets = defaultdict(set)
    user_seen_asns = defaultdict(set)
    user_seen_countries = defaultdict(set)
    user_seen_devices = defaultdict(set)
    
    # Pairwise seen sets
    user_seen_subnet_device = defaultdict(set)
    user_seen_country_device = defaultdict(set)
    user_seen_asn_device = defaultdict(set)

    subnet_seen_users = defaultdict(set)
    asn_seen_users = defaultdict(set)
    country_seen_users = defaultdict(set)
    device_seen_users = defaultdict(set)
    
    subnet_failed_users = defaultdict(set)
    asn_failed_users = defaultdict(set)
    device_failed_users = defaultdict(set)

    subnet_fail_count = defaultdict(int)
    subnet_total_count = defaultdict(int)
    asn_fail_count = defaultdict(int)
    asn_total_count = defaultdict(int)
    device_fail_count = defaultdict(int)
    device_total_count = defaultdict(int)
    
    # Rolling pressure engines (Requires lambda encapsulation for defaultdict)
    subnet_10m = defaultdict(lambda: UniqueWindow(600))
    subnet_1h  = defaultdict(lambda: UniqueWindow(3600))
    asn_10m    = defaultdict(lambda: UniqueWindow(600))
    device_10m = defaultdict(lambda: UniqueWindow(600))

    rows = []

    print("Computing temporal graph features across full history...")
    # Using itertuples for faster iteration over large dataframe
    for row in df.itertuples(index=False):
        user = row.User_ID
        # Using getattr since some columns might have spaces or special chars if not renamed
        subnet = getattr(row, 'IP_Subnet', 'Unknown')
        asn = getattr(row, 'ASN', 'Unknown')
        country = getattr(row, 'Country', 'Unknown')
        device = getattr(row, 'Device_Combo', 'Unknown')
        success = int(row.Login_Successful)
        
        ts_sec = row.Login_Timestamp.timestamp()

        # Complex pair markers
        sub_dev = f"{subnet}|{device}"
        cnty_dev = f"{country}|{device}"
        asn_dev = f"{asn}|{device}"

        feat = {}

        # User Neighborhood Counts (Log dampened for isolation tree limits)
        feat['GF_User_Subnet_Count_Before'] = np.log1p(len(user_seen_subnets[user]))
        feat['GF_User_ASN_Count_Before'] = np.log1p(len(user_seen_asns[user]))
        feat['GF_User_Country_Count_Before'] = np.log1p(len(user_seen_countries[user]))
        feat['GF_User_Device_Count_Before'] = np.log1p(len(user_seen_devices[user]))

        # Edge Novelty (Have they used this specific infrastructure before?)
        feat['GF_Edge_User_Subnet_New'] = int(subnet not in user_seen_subnets[user])
        feat['GF_Edge_User_ASN_New'] = int(asn not in user_seen_asns[user])
        feat['GF_Edge_User_Country_New'] = int(country not in user_seen_countries[user])
        feat['GF_Edge_User_Device_New'] = int(device not in user_seen_devices[user])
        
        # Pairwise Novelty
        feat['GF_Edge_User_Subnet_Device_New'] = int(sub_dev not in user_seen_subnet_device[user])
        feat['GF_Edge_User_Country_Device_New'] = int(cnty_dev not in user_seen_country_device[user])
        feat['GF_Edge_User_ASN_Device_New'] = int(asn_dev not in user_seen_asn_device[user])

        # Drift score
        feat['GF_New_Entity_Count'] = (
            feat['GF_Edge_User_Subnet_New'] +
            feat['GF_Edge_User_ASN_New'] +
            feat['GF_Edge_User_Country_New'] +
            feat['GF_Edge_User_Device_New']
        )
        feat['GF_User_Drift_Score'] = (
            1.0 * feat['GF_Edge_User_Subnet_New'] +
            1.0 * feat['GF_Edge_User_ASN_New'] +
            0.75 * feat['GF_Edge_User_Country_New'] +
            1.25 * feat['GF_Edge_User_Device_New'] +
            1.5 * feat['GF_Edge_User_Subnet_Device_New']
        )

        # Baseline Infrastructure Hub Degrees
        feat['GF_Subnet_User_Count_Before'] = np.log1p(len(subnet_seen_users[subnet]))
        feat['GF_ASN_User_Count_Before'] = np.log1p(len(asn_seen_users[asn]))
        feat['GF_Country_User_Count_Before'] = np.log1p(len(country_seen_users[country]))
        feat['GF_Device_User_Count_Before'] = np.log1p(len(device_seen_users[device]))
        
        # Historical Risk Propagation (Ratio calculations)
        feat['GF_Subnet_Failure_Rate_Prior'] = (
            subnet_fail_count[subnet] / subnet_total_count[subnet] if subnet_total_count[subnet] > 0 else 0.0
        )
        feat['GF_ASN_Failure_Rate_Prior'] = (
            asn_fail_count[asn] / asn_total_count[asn] if asn_total_count[asn] > 0 else 0.0
        )
        feat['GF_Device_Failure_Rate_Prior'] = (
            device_fail_count[device] / device_total_count[device] if device_total_count[device] > 0 else 0.0
        )
        
        # Neighborhood Malicious Density
        feat['GF_Subnet_Failed_User_Count_Before'] = np.log1p(len(subnet_failed_users[subnet]))
        feat['GF_ASN_Failed_User_Count_Before'] = np.log1p(len(asn_failed_users[asn]))
        feat['GF_Device_Failed_User_Count_Before'] = np.log1p(len(device_failed_users[device]))
        
        feat['GF_Subnet_Has_High_Failure_Pressure'] = int(feat['GF_Subnet_Failed_User_Count_Before'] > 2.0 and feat['GF_Subnet_Failure_Rate_Prior'] > 0.5)
        feat['GF_Device_Has_High_Failure_Pressure'] = int(feat['GF_Device_Failed_User_Count_Before'] > 1.5 and feat['GF_Device_Failure_Rate_Prior'] > 0.5)

        # Windowed Infrastructure Pressure Generation
        sub10 = subnet_10m[subnet]
        sub1h = subnet_1h[subnet]
        a10   = asn_10m[asn]
        d10   = device_10m[device]
        
        sub10.trim(ts_sec)
        sub1h.trim(ts_sec)
        a10.trim(ts_sec)
        d10.trim(ts_sec)
        
        feat['GF_Subnet_User_Count_10Min'] = np.log1p(sub10.unique_count)
        feat['GF_Subnet_User_Count_1Hour'] = np.log1p(sub1h.unique_count)
        feat['GF_ASN_User_Count_10Min']    = np.log1p(a10.unique_count)
        feat['GF_Device_User_Count_10Min'] = np.log1p(d10.unique_count)

        rows.append(feat)

        # Update historical state AFTER calculating features for this current event
        user_seen_subnets[user].add(subnet)
        user_seen_asns[user].add(asn)
        user_seen_countries[user].add(country)
        user_seen_devices[user].add(device)
        
        user_seen_subnet_device[user].add(sub_dev)
        user_seen_country_device[user].add(cnty_dev)
        user_seen_asn_device[user].add(asn_dev)

        subnet_seen_users[subnet].add(user)
        asn_seen_users[asn].add(user)
        country_seen_users[country].add(user)
        device_seen_users[device].add(user)

        subnet_total_count[subnet] += 1
        asn_total_count[asn] += 1
        device_total_count[device] += 1
        
        sub10.add(ts_sec, user)
        sub1h.add(ts_sec, user)
        a10.add(ts_sec, user)
        d10.add(ts_sec, user)

        if success == 0:
            subnet_fail_count[subnet] += 1
            asn_fail_count[asn] += 1
            device_fail_count[device] += 1
            
            subnet_failed_users[subnet].add(user)
            asn_failed_users[asn].add(user)
            device_failed_users[device].add(user)

    graph_feat_df = pd.DataFrame(rows)
    return pd.concat([df, graph_feat_df], axis=1)
