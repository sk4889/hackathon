import os
import json
import pandas as pd
import numpy as np
from faker import Faker
from math import radians, sin, cos, sqrt, atan2

# Set random seed for reproducibility
Faker.seed(42)
np.random.seed(42)
fake = Faker()

# Country coordinates for distance calculations
country_coords = {
    'US': (37.09024, -95.712891),
    'IN': (20.593684, 78.96288),
    'UK': (55.378051, -3.435973),
    'CA': (56.130366, -106.346771),
    'RU': (61.52401, 105.318756),
    'CN': (35.86166, 104.195397),
    'NG': (9.081999, 8.675277)
}

# Predefined lists for consistent data
device_types = ['desktop', 'laptop', 'tablet', 'phone']
company_names = ['Acme Corp', 'Globex Inc', 'Soylent Solutions', 'Initech', 'Umbrella Corp']
occupations = ['Software Engineer', 'Accountant', 'Marketing Manager', 'Sales Representative', 'Teacher']

def haversine(coord1, coord2):
    """Calculate distance between two coordinates in kilometers."""
    R = 6371.0  # Earth radius in km
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def generate_custom_ip():
    """Generate a custom IP address in 192.10.x.y format."""
    return f"192.10.{np.random.randint(0, 256)}.{np.random.randint(0, 256)}"

def generate_data(num_records=1000):
    """Generate customer profiles and transaction data."""
    num_customers = max(1, num_records // 10)

    # Generate customer profiles
    customer_profiles = pd.DataFrame({
        'customer_id': [fake.uuid4() for _ in range(num_customers)],
        'customer_type': np.random.choice(['individual', 'corporate'], num_customers, p=[0.8, 0.2]),
        'risk_rating': np.random.choice(['low', 'medium', 'high'], num_customers, p=[0.7, 0.2, 0.1]),
        'nationality': np.random.choice(list(country_coords.keys()), num_customers),
        'occupation': np.random.choice(occupations, num_customers),
        'ip_address': [generate_custom_ip() for _ in range(num_customers)],
        'location': np.random.choice(list(country_coords.keys()), num_customers),
        'usual_amount_mean': np.round(np.random.lognormal(mean=5, sigma=1, size=num_customers), 2),
        'usual_frequency_per_day': np.random.randint(1, 5, num_customers),
        'account_age_days': np.random.randint(1, 1000, num_customers),
        'last_update_days': np.random.randint(1, 365, num_customers)
    })

    # Generate transactions
    transactions = []
    for _, customer in customer_profiles.iterrows():
        num_tx = np.random.randint(1, max(2, num_records // num_customers + 1))
        base_time = fake.date_time_this_year()
        for i in range(num_tx):
            time_offset = np.random.randint(-10, 11) * 3600
            tx_time = pd.Timestamp(base_time) + pd.Timedelta(seconds=time_offset)
            prev_country = customer['location']
            new_country = np.random.choice(list(country_coords.keys()))
            time_diff_hours = abs(time_offset / 3600)
            distance = haversine(country_coords[prev_country], country_coords[new_country])
            impossible_travel = distance / time_diff_hours > 900 if time_diff_hours > 0 else False
            hour_diff = abs(tx_time.hour - customer['usual_frequency_per_day'])
            unusual_timing = hour_diff > 6
            amount = np.round(np.random.lognormal(mean=np.log(customer['usual_amount_mean']), sigma=1), 2)
            deviation = abs(amount - customer['usual_amount_mean']) / customer['usual_amount_mean']
            is_anomaly = (customer['risk_rating'] == 'high' or impossible_travel or deviation > 1 or unusual_timing)
            transactions.append({
                'customer_id': customer['customer_id'],
                'timestamp': tx_time,
                'transaction_amount': amount,
                'transaction_type': np.random.choice(['cash_deposit', 'wire_transfer', 'remittance']),
                'transaction_frequency': np.random.randint(1, customer['usual_frequency_per_day'] + 2),
                'transaction_channel': np.random.choice(['branch', 'online', 'ATM']),
                'counterparty_name': np.random.choice(company_names),
                'counterparty_country': np.random.choice(list(country_coords.keys())),
                'payment_method': np.random.choice(['cash', 'digital']),
                'originating_country': new_country,
                'destination_country': np.random.choice(list(country_coords.keys())),
                'sanctioned_country': np.random.choice([True, False], p=[0.1, 0.9]),
                'deviation_from_profile': deviation,
                'unusual_timing': unusual_timing,
                'structuring': np.random.choice([True, False], p=[0.05, 0.95]),
                'rapid_movement': np.random.choice([True, False], p=[0.1, 0.9]),
                'sanctions_list_hit': np.random.choice([True, False], p=[0.05, 0.95]),
                'pep_match': np.random.choice([True, False], p=[0.1, 0.9]),
                'negative_media': np.random.choice([True, False], p=[0.05, 0.95]),
                'ip_address': generate_custom_ip(),
                'device_type': np.random.choice(device_types),
                'account_age_days': customer['account_age_days'],
                'last_update_days': customer['last_update_days'],
                'failed_attempts': np.random.randint(0, 5),
                'impossible_travel': impossible_travel,
                'is_anomaly': int(is_anomaly)
            })

    transaction_df = pd.DataFrame(transactions)
    return customer_profiles, transaction_df

def save_data(customer_profiles, transaction_df, output_dir='data/raw'):
    """Save customer profiles and transactions to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    customer_profiles.to_csv(f'{output_dir}/customer_profiles.csv', index=False)
    transaction_df.to_csv(f'{output_dir}/transaction_data.csv', index=False)
    print(f"Saved data to {output_dir}/customer_profiles.csv and {output_dir}/transaction_data.csv")

def save_retrain_data(customer_profiles, transaction_df, output_dir='data/retrain'):
    """Save transaction data as JSON for API retraining."""
    os.makedirs(output_dir, exist_ok=True)
    transactions = transaction_df.to_dict(orient='records')
    y = transaction_df['is_anomaly'].tolist()
    retrain_data = {'X': transactions, 'y': y}
    with open(f'{output_dir}/retrain_data.json', 'w') as f:
        json.dump(retrain_data, f, indent=2, default=str)
    print(f"Saved retrain data to {output_dir}/retrain_data.json")

if __name__ == "__main__":
    profiles, transactions = generate_data(1000)
    save_data(profiles, transactions)
    save_retrain_data(profiles, transactions)