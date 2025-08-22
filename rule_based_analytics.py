import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Country coordinates (lat, long)
country_coords = {
    'US': (37.09024, -95.712891),
    'IN': (20.593684, 78.96288),
    'UK': (55.378051, -3.435973),
    'CA': (56.130366, -106.346771),
    'RU': (61.52401, 105.318756),
    'CN': (35.86166, 104.195397),
    'NG': (9.081999, 8.675277)
}

def customer_centric_view(customer_profiles, transaction_df, col):
    # Validate inputs
    if customer_profiles.empty or transaction_df.empty:
        col.error("Customer profiles or transaction data is empty")
        return
    
    required_cols_profiles = ['customer_id', 'device_type', 'nationality', 'risk_rating']
    if not all(col in customer_profiles.columns for col in required_cols_profiles):
        col.error(f"Missing required columns in customer_profiles: {[col for col in required_cols_profiles if col not in customer_profiles.columns]}")
        return
    
    required_cols_transactions = ['customer_id', 'timestamp', 'transaction_amount', 'device_type', 'counterparty_country', 'originating_country', 'is_anomaly']
    if not all(col in transaction_df.columns for col in required_cols_transactions):
        col.error(f"Missing required columns in transaction_df: {[col for col in required_cols_transactions if col not in transaction_df.columns]}")
        return
    
    col.subheader("Customer-Centric Fraudulent Transactions")
    col.write("Overview of customers with potential fraud indicators based on transaction patterns.")
    
    try:
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(transaction_df['timestamp']):
            transaction_df['timestamp'] = pd.to_datetime(transaction_df['timestamp'], errors='coerce')
        
        # Merge data
        merged_df = transaction_df.merge(customer_profiles, on='customer_id', how='left', suffixes=('_tx', '_profile'))
        
        # Rule-based anomaly detection
        merged_df['device_type_mismatch'] = merged_df['device_type_tx'] != merged_df['device_type_profile']
        merged_df['counterparty_risk'] = merged_df['counterparty_country'].isin(['RU', 'CN', 'NG'])
        
        # Calculate risk score (weighted sum of anomalies)
        merged_df['risk_score'] = (merged_df['is_anomaly'] * 2 + 
                                 merged_df['device_type_mismatch'] * 1 + 
                                 merged_df['counterparty_risk'] * 1.5)
        
        # Aggregate by customer
        agg_df = merged_df.groupby('customer_id').agg({
            'is_anomaly': 'sum',
            'device_type_mismatch': 'sum',
            'counterparty_risk': 'sum',
            'risk_score': 'mean',
            'transaction_amount': 'mean',
            'nationality': 'first',
            'risk_rating': 'first'
        }).reset_index()
        
        # Rename columns for business readability
        agg_df = agg_df.rename(columns={
            'is_anomaly': 'Base Anomaly',
            'device_type_mismatch': 'Device Mismatch',
            'counterparty_risk': 'High-Risk Country',
            'transaction_amount': 'Avg Transaction Amount',
            'nationality': 'Nationality',
            'risk_rating': 'Risk Rating'
        })
        
        # Format transaction amount as currency
        agg_df['Avg Transaction Amount'] = agg_df['Avg Transaction Amount'].apply(lambda x: f"${x:,.2f}")
        
        # Calculate total anomalies
        agg_df['Total Anomalies'] = (agg_df['Base Anomaly'] + 
                                    agg_df['Device Mismatch'] + 
                                    agg_df['High-Risk Country'])
        
        col.write("Customer Anomaly Summary (Top 10 High-Risk Customers)")
        col.write("Shows customers with highest anomaly counts and risk scores.")
        col.dataframe(agg_df.sort_values('risk_score', ascending=False)[
            ['customer_id', 'Base Anomaly', 'Device Mismatch', 'High-Risk Country', 
             'Total Anomalies', 'risk_score', 'Avg Transaction Amount', 'Nationality', 'Risk Rating']
        ].head(10).style.format({'risk_score': '{:.2f}'}))
        
        # Stacked Bar Chart: Customer count by total anomalies
        col.write("Customer Count by Total Anomalies")
        col.write("Shows the number of customers with 0, 1, 2, or more anomalies.")
        anomaly_counts = agg_df['Total Anomalies'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(4, 3))
        anomaly_counts.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title("Customer Count by Total Anomalies")
        ax.set_xlabel("Total Anomalies")
        ax.set_ylabel("Customer Count")
        plt.tight_layout()
        col.pyplot(fig)
        
        # Bar Chart: Average transaction amount by risk rating
        col.write("Average Transaction Amount by Risk Rating")
        col.write("Compares average transaction amounts across customer risk ratings.")
        risk_amounts = merged_df.groupby('risk_rating')['transaction_amount'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.barplot(data=risk_amounts, x='risk_rating', y='transaction_amount', ax=ax)
        ax.set_title("Avg Transaction Amount by Risk Rating")
        ax.set_ylabel("Amount ($)")
        ax.set_xlabel("Risk Rating")
        plt.tight_layout()
        col.pyplot(fig)
        
        # Heatmap: Correlation between anomalies and transaction amount
        col.write("Correlation of Anomalies and Transaction Amount")
        col.write("Shows relationships between anomaly types and transaction amounts.")
        corr_cols = ['is_anomaly', 'device_type_mismatch', 'counterparty_risk', 'transaction_amount']
        corr_matrix = merged_df[corr_cols].corr()
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', annot_kws={'size': 8}, ax=ax)
        ax.set_title("Correlation Heatmap")
        plt.tight_layout()
        col.pyplot(fig)
        
        # Device Type Mismatch Insights
        col.write("Device Type Mismatch Insights")
        col.write("Shows frequency of transactions where device type differs from customer profile.")
        device_mismatch_counts = merged_df.groupby('device_type_tx')['device_type_mismatch'].sum().reset_index()
        device_mismatch_counts = device_mismatch_counts.rename(columns={'device_type_mismatch': 'Device Mismatch Count'})
        col.bar_chart(device_mismatch_counts.set_index('device_type_tx'))
    
    except Exception as e:
        col.error(f"Error in customer-centric view: {str(e)}")

def transactional_view(customer_profiles, transaction_df, col):
    # Validate inputs
    if customer_profiles.empty or transaction_df.empty:
        col.error("Customer profiles or transaction data is empty")
        return
    
    required_cols_profiles = ['customer_id', 'device_type', 'nationality', 'risk_rating']
    if not all(col in customer_profiles.columns for col in required_cols_profiles):
        col.error(f"Missing required columns in customer_profiles: {[col for col in required_cols_profiles if col not in customer_profiles.columns]}")
        return
    
    required_cols_transactions = ['customer_id', 'timestamp', 'transaction_amount', 'device_type', 'counterparty_country', 'originating_country', 'is_anomaly']
    if not all(col in transaction_df.columns for col in required_cols_transactions):
        col.error(f"Missing required columns in transaction_df: {[col for col in required_cols_transactions if col not in transaction_df.columns]}")
        return
    
    col.subheader("Transactional View")
    col.write("Detailed view of transactions flagged as anomalous.")
    
    try:
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(transaction_df['timestamp']):
            transaction_df['timestamp'] = pd.to_datetime(transaction_df['timestamp'], errors='coerce')
        
        # Merge data
        merged_df = transaction_df.merge(customer_profiles, on='customer_id', how='left', suffixes=('_tx', '_profile'))
        
        # Rule-based anomaly detection
        merged_df['device_type_mismatch'] = merged_df['device_type_tx'] != merged_df['device_type_profile']
        merged_df['counterparty_risk'] = merged_df['counterparty_country'].isin(['RU', 'CN', 'NG'])
        
        # Calculate anomaly severity
        merged_df['anomaly_severity'] = (merged_df['is_anomaly'] + 
                                       merged_df['device_type_mismatch'] + 
                                       merged_df['counterparty_risk'])
        
        # Filter anomalous transactions
        anomalous_tx = merged_df[merged_df[['is_anomaly', 'device_type_mismatch', 'counterparty_risk']].sum(axis=1) > 0]
        
        # Rename columns for business readability
        display_df = anomalous_tx[['customer_id', 'timestamp', 'transaction_amount', 
                                 'device_type_mismatch', 'counterparty_risk', 'anomaly_severity']].copy()
        display_df = display_df.rename(columns={
            'is_anomaly': 'Base Anomaly',
            'device_type_mismatch': 'Device Mismatch',
            'counterparty_risk': 'High-Risk Country',
            'transaction_amount': 'Amount',
            'anomaly_severity': 'Anomaly Severity'
        })
        
        # Format transaction amount as currency
        display_df['Amount'] = display_df['Amount'].apply(lambda x: f"${x:,.2f}")
        
        col.write("Anomalous Transactions (Top 10)")
        col.write("Transactions with at least one anomaly flag. High severity (≥2) indicates multiple red flags.")
        col.dataframe(display_df.head(10).style.apply(
            lambda x: ['background-color: #ffcccc' if x['Anomaly Severity'] >= 2 else '' for _ in x],
            axis=1
        ))
        
        # Line Chart: Anomalous transactions over time
        if not anomalous_tx.empty and 'timestamp' in anomalous_tx.columns and anomalous_tx['timestamp'].notnull().any():
            anomalous_tx['date'] = anomalous_tx['timestamp'].dt.date
            time_counts = anomalous_tx.groupby('date').size().reset_index(name='Transaction Count')
            if len(time_counts['date'].unique()) >= 2:  # Require at least 2 unique dates
                col.write("Anomalous Transactions Over Time")
                col.write("Shows the trend of anomalous transactions by date.")
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.lineplot(data=time_counts, x='date', y='Transaction Count', ax=ax)
                ax.set_title("Anomalous Transactions Over Time")
                ax.set_xlabel("Date")
                ax.set_ylabel("Transaction Count")
                plt.xticks(rotation=45)
                plt.tight_layout()
                col.pyplot(fig)
            else:
                col.write("Insufficient data for Anomalous Transactions Over Time (requires multiple dates).")
        else:
            col.write("No valid timestamp data for Anomalous Transactions Over Time.")
        
        # Bar Chart: Transaction count by counterparty country
        col.write("Anomalous Transactions by Counterparty Country")
        col.write("Shows which countries are involved in anomalous transactions.")
        country_counts = anomalous_tx.groupby('counterparty_country').size().reset_index(name='Transaction Count')
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.barplot(data=country_counts, x='counterparty_country', y='Transaction Count', ax=ax)
        ax.set_title("Anomalous Transactions by Counterparty Country")
        ax.set_xlabel("Counterparty Country")
        ax.set_ylabel("Transaction Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        col.pyplot(fig)
        
        # Geographic Visualization
        col.write("Geographic Distribution of Anomalous Transactions")
        col.write("Map of originating countries for anomalous transactions (up to 100).")
        with col:
            m = folium.Map(location=[20, 0], zoom_start=2)
            for _, row in anomalous_tx.head(100).iterrows():
                if row['originating_country'] in country_coords:
                    folium.Marker(
                        location=country_coords[row['originating_country']],
                        popup=f"Customer: {row['customer_id']}<br>Amount: {row['transaction_amount']:.2f}<br>Anomaly: {row['is_anomaly']}",
                        icon=folium.Icon(color='red' if row['is_anomaly'] else 'blue')
                    ).add_to(m)
            folium_static(m)
    
    except Exception as e:
        col.error(f"Error in transactional view: {str(e)}")