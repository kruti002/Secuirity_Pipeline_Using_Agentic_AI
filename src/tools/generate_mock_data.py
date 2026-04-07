import pandas as pd
import numpy as np
import random
import time

def generate_mock_data(n_rows=100000):
    print(f"Generating {n_rows} rows of mock RBA data...")
    
    # Pre-defined lists for realism
    countries = ['US', 'DE', 'GB', 'FR', 'JP', 'CN', 'BR', 'CA', 'AU', 'IN']
    ip_bases = ['192.168.', '10.0.', '172.16.', '8.8.', '1.1.', '13.107.', '20.190.', '40.126.']
    devices = ['desktop', 'mobile', 'tablet', 'bot', 'unknown']
    os_list = ['Windows 10', 'macOS 10.15', 'Android 11', 'iOS 14', 'Linux', 'Windows 7']
    browsers = ['Chrome 91.0', 'Firefox 89.0', 'Safari 14.1', 'Edge 91.0', 'Opera 77.0']
    
    data = []
    
    # User distribution
    user_ids = list(range(1000, 2000)) # 1000 active users
    
    # Attack stats
    attack_ips = set()
    for _ in range(50):
        attack_ips.add(f"{random.choice(ip_bases)}{random.randint(0,255)}.{random.randint(0,255)}")

    for i in range(n_rows):
        user_id = random.choice(user_ids)
        ip = f"{random.choice(ip_bases)}{random.randint(0,255)}.{random.randint(0,255)}"
        country = random.choice(countries)
        asn = random.randint(100, 600000)
        
        # Scenario: Some users have fixed behavior
        if user_id < 1200: # Stable users
            ip = f"192.168.1.{user_id % 255}"
            country = 'US'
            asn = 15169 # Google ASN
            device = 'desktop'
            os_ver = 'Windows 10'
            browser = 'Chrome 91.0'
        else:
            device = random.choice(devices)
            os_ver = random.choice(os_list)
            browser = random.choice(browsers)

        is_attack_ip = ip in attack_ips
        
        # Login success rate
        # Higher failure rate for attack IPs
        if is_attack_ip:
            login_success = random.random() > 0.8
        else:
            login_success = random.random() > 0.1
            
        # Is Account Takeover (ATO)
        # Randomly assign ATO to some events involving attack IPs or weird behavior
        is_ato = False
        if is_attack_ip and login_success and random.random() > 0.5:
            is_ato = True
        elif not is_attack_ip and login_success and random.random() > 0.995:
            is_ato = True
            
        timestamp = int(time.time()) - random.randint(0, 86400 * 30) # Last 30 days
        
        row = {
            'IP Address': ip,
            'Country': country,
            'Region': f"Region_{country}",
            'City': f"City_{random.randint(0,100)}",
            'ASN': asn,
            'User Agent String': f"Mozilla/5.0 ({os_ver}; {browser})",
            'OS Name and Version': os_ver,
            'Browser Name and Version': browser,
            'Device Type': device,
            'User ID': user_id,
            'Login Timestamp': timestamp,
            'Round-Trip Time [ms]': random.randint(10, 500),
            'Login Successful': login_success,
            'Is Attack IP': is_attack_ip,
            'Is Account Takeover': is_ato
        }
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv('data/rba-dataset.csv', index=False)
    print("Data saved to data/rba-dataset.csv")

if __name__ == "__main__":
    generate_mock_data()
