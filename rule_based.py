import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import seaborn as sns
import matplotlib.pyplot as plt

# Country coordinates for mapping
country_coords = {
    'US': (37.09024, -95.712891), 'IN': (20.593684, 78.96288),
    'UK': (55.378051, -3.435973), 'CA': (56.130366, -106.346771),
    'RU': (61.52401, 105.318756), 'CN': (35.86166, 104.195397),
    'NG': (9.081999, 8.675277)
}

def show_customer_centric(profiles, transactions, column):
    """Display customer-centric analysis of fraudulent transactions."""
    if profiles.empty or transactions.empty:
        column.error("No customer or transaction data available")
        return

    column.subheader("Customer-Centric Fraud Analysis")
    try:
        # Merge data
        merged_df = transactions.merge(profiles, on='customer_id', how='left')
        anomalous_tx = merged_df[merged_df['is_anomaly'] == 1]

        # Display summary
        column.write(f"Total Anomalous Transactions: {len(anomalous_tx)}")
        column.write("Top Risky Customers:")
        risky_customers = anomalous_tx.groupby('customer_id').size().reset_index(name='Anomaly Count')
        column.dataframe(risky_customers.head(5))

        # Plot risk by nationality
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        sns.countplot(data=anomalous_tx, x='nationality', ax=ax)
        ax.set_title('Anomalous Transactions by Nationality', fontsize=10)
        ax.set_xlabel('Nationality', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.tick_params(labelsize=7)
        plt.xticks(rotation=45)
        plt.tight_layout(pad=0.3)
        column.pyplot(fig)
    except Exception as e:
        column.error(f"Error in customer-centric analysis: {e}")

def show_transactional(profiles, transactions, column):
    """Display transactional analysis of fraudulent transactions."""
    if profiles.empty or transactions.empty:
        column.error("No customer or transaction data available")
        return

    column.subheader("Transactional Fraud Analysis")
    try:
        anomalous_tx = transactions[transactions['is_anomaly'] == 1]

        # Plot transaction amounts
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        sns.histplot(data=anomalous_tx, x='transaction_amount', bins=20, ax=ax)
        ax.set_title('Anomalous Transaction Amounts', fontsize=10)
        ax.set_xlabel('Amount', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.tick_params(labelsize=7)
        column.pyplot(fig)

        # Geographic map
        column.write("Geographic Distribution")
        m = folium.Map(location=[20, 0], zoom_start=2)
        for _, row in anomalous_tx.head(100).iterrows():
            if row['originating_country'] in country_coords:
                folium.Marker(
                    location=country_coords[row['originating_country']],
                    popup=f"Customer: {row['customer_id']}<br>Amount: {row['transaction_amount']:.2f}",
                    icon=folium.Icon(color='red')
                ).add_to(m)
        folium_static(m)
    except Exception as e:
        column.error(f"Error in transactional analysis: {e}")