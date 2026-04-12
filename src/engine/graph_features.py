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

class GraphFeatureEngine:
    def __init__(self):
        self.user_seen_subnets = defaultdict(set)
        self.user_seen_asns = defaultdict(set)
        self.user_seen_countries = defaultdict(set)
        self.user_seen_devices = defaultdict(set)
        
        # Pairwise seen sets
        self.user_seen_subnet_device = defaultdict(set)
        self.user_seen_country_device = defaultdict(set)
        self.user_seen_asn_device = defaultdict(set)

        self.subnet_seen_users = defaultdict(set)
        self.asn_seen_users = defaultdict(set)
        self.country_seen_users = defaultdict(set)
        self.device_seen_users = defaultdict(set)
        
        self.subnet_failed_users = defaultdict(set)
        self.asn_failed_users = defaultdict(set)
        self.device_failed_users = defaultdict(set)

        self.subnet_fail_count = defaultdict(int)
        self.subnet_total_count = defaultdict(int)
        self.asn_fail_count = defaultdict(int)
        self.asn_total_count = defaultdict(int)
        self.device_fail_count = defaultdict(int)
        self.device_total_count = defaultdict(int)
        
        # Rolling pressure engines
        self.subnet_10m = defaultdict(lambda: UniqueWindow(600))
        self.subnet_1h  = defaultdict(lambda: UniqueWindow(3600))
        self.asn_10m    = defaultdict(lambda: UniqueWindow(600))
        self.device_10m = defaultdict(lambda: UniqueWindow(600))

    def process_chunk(self, df):
        """
        Computes features for a chunk of data and updates internal state.
        Assumes df is already sorted by timestamp.
        """
        # Rename columns to ensure itertuples works reliably with snake_case
        df_proc = df.rename(columns={
            'User ID': 'User_ID',
            'Login Timestamp': 'Login_Timestamp',
            'Login Successful': 'Login_Successful'
        })

        rows = []
        for row in df_proc.itertuples(index=False):
            user = row.User_ID
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
            # User Neighborhood Counts
            feat['GF_User_Subnet_Count_Before'] = np.log1p(len(self.user_seen_subnets[user]))
            feat['GF_User_ASN_Count_Before'] = np.log1p(len(self.user_seen_asns[user]))
            feat['GF_User_Country_Count_Before'] = np.log1p(len(self.user_seen_countries[user]))
            feat['GF_User_Device_Count_Before'] = np.log1p(len(self.user_seen_devices[user]))

            # Edge Novelty
            feat['GF_Edge_User_Subnet_New'] = int(subnet not in self.user_seen_subnets[user])
            feat['GF_Edge_User_ASN_New'] = int(asn not in self.user_seen_asns[user])
            feat['GF_Edge_User_Country_New'] = int(country not in self.user_seen_countries[user])
            feat['GF_Edge_User_Device_New'] = int(device not in self.user_seen_devices[user])
            feat['GF_Edge_User_Subnet_Device_New'] = int(sub_dev not in self.user_seen_subnet_device[user])
            feat['GF_Edge_User_Country_Device_New'] = int(cnty_dev not in self.user_seen_country_device[user])
            feat['GF_Edge_User_ASN_Device_New'] = int(asn_dev not in self.user_seen_asn_device[user])

            feat['GF_New_Entity_Count'] = feat['GF_Edge_User_Subnet_New'] + feat['GF_Edge_User_ASN_New'] + feat['GF_Edge_User_Country_New'] + feat['GF_Edge_User_Device_New']
            feat['GF_User_Drift_Score'] = (1.0 * feat['GF_Edge_User_Subnet_New'] + 1.0 * feat['GF_Edge_User_ASN_New'] + 0.75 * feat['GF_Edge_User_Country_New'] + 1.25 * feat['GF_Edge_User_Device_New'] + 1.5 * feat['GF_Edge_User_Subnet_Device_New'])
            
            # Drift/Pressure
            feat['GF_Subnet_User_Count_Before'] = np.log1p(len(self.subnet_seen_users[subnet]))
            feat['GF_ASN_User_Count_Before'] = np.log1p(len(self.asn_seen_users[asn]))
            feat['GF_Country_User_Count_Before'] = np.log1p(len(self.country_seen_users[country]))
            feat['GF_Device_User_Count_Before'] = np.log1p(len(self.device_seen_users[device]))
            
            feat['GF_Subnet_Failure_Rate_Prior'] = (self.subnet_fail_count[subnet] / self.subnet_total_count[subnet] if self.subnet_total_count[subnet] > 0 else 0.0)
            feat['GF_ASN_Failure_Rate_Prior'] = (self.asn_fail_count[asn]/self.asn_total_count[asn] if self.asn_total_count[asn]>0 else 0.0)
            feat['GF_Device_Failure_Rate_Prior'] = (self.device_fail_count[device]/self.device_total_count[device] if self.device_total_count[device]>0 else 0.0)

            feat['GF_Subnet_Failed_User_Count_Before'] = np.log1p(len(self.subnet_failed_users[subnet]))
            feat['GF_ASN_Failed_User_Count_Before'] = np.log1p(len(self.asn_failed_users[asn]))
            feat['GF_Device_Failed_User_Count_Before'] = np.log1p(len(self.device_failed_users[device]))

            # Windowed pressure
            s10 = self.subnet_10m[subnet]
            s1h = self.subnet_1h[subnet]
            a10 = self.asn_10m[asn]
            d10 = self.device_10m[device]
            s10.trim(ts_sec); s1h.trim(ts_sec); a10.trim(ts_sec); d10.trim(ts_sec)
            
            feat['GF_Subnet_User_Count_10Min'] = np.log1p(s10.unique_count)
            feat['GF_Subnet_User_Count_1Hour'] = np.log1p(s1h.unique_count)
            feat['GF_ASN_User_Count_10Min']    = np.log1p(a10.unique_count)
            feat['GF_Device_User_Count_10Min'] = np.log1p(d10.unique_count)
            
            feat['GF_Subnet_Has_High_Failure_Pressure'] = int(feat['GF_Subnet_Failed_User_Count_Before'] > 2.0 and feat['GF_Subnet_Failure_Rate_Prior'] > 0.5)
            feat['GF_Device_Has_High_Failure_Pressure'] = int(feat['GF_Device_Failed_User_Count_Before'] > 1.5 and feat['GF_Device_Failure_Rate_Prior'] > 0.5)

            rows.append(feat)

            # Update state
            self.user_seen_subnets[user].add(subnet)
            self.user_seen_asns[user].add(asn)
            self.user_seen_countries[user].add(country)
            self.user_seen_devices[user].add(device)
            self.user_seen_subnet_device[user].add(sub_dev)
            self.subnet_seen_users[subnet].add(user)
            self.asn_seen_users[asn].add(user)
            self.device_seen_users[device].add(user)
            self.subnet_total_count[subnet] += 1
            self.asn_total_count[asn] += 1
            self.device_total_count[device] += 1
            s10.add(ts_sec, user)
            
            if success == 0:
                self.subnet_fail_count[subnet] += 1
                self.asn_fail_count[asn] += 1
                self.device_fail_count[device] += 1
                self.subnet_failed_users[subnet].add(user)

        graph_feat_df = pd.DataFrame(rows)
        return pd.concat([df.reset_index(drop=True), graph_feat_df], axis=1)

def add_graph_features(df):
    """Legacy wrapper."""
    engine = GraphFeatureEngine()
    return engine.process_chunk(df)
