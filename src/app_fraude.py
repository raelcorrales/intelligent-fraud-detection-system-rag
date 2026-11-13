import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import altair as alt  # For robust embedding visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# --- Conditional Imports for t-SNE ---
try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False
    st.warning("The `scikit-learn` package is not installed. t-SNE embedding visualization is disabled. Install via `pip install scikit-learn`.")

# --- Streamlit Application Configuration ---
st.set_page_config(
    page_title="FraudAI - Transaction Analytics",
    page_icon="⚡",
    layout="wide"
)

# --- Sample Data Generation Function (for demonstration) ---
@st.cache_data
def generate_sample_data(num_records=5000):
    """Generates a synthetic DataFrame with banking transaction features."""
    np.random.seed(42)

    risk_levels_consistent = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    data = {
        'TransactionID': [f'TXN{i:05d}' for i in range(num_records)],
        'AccountID': [f'ACC{np.random.randint(1, 500):03d}' for _ in range(num_records)],
        'TransactionDate': pd.to_datetime(pd.date_range('2024-01-01', periods=num_records, freq='h').to_numpy() +
                                         pd.to_timedelta(np.random.randint(-24, 24, num_records), unit='h')),
        'TransactionAmount': np.round(np.random.lognormal(mean=3, sigma=1.0, size=num_records), 2) * 10,
        'TransactionType': np.random.choice(['Credit', 'Debit'], num_records, p=[0.3, 0.7]),
        'Location': np.random.choice([
            'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
            'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
            'Austin', 'Jacksonville', 'Fort Worth', 'Columbus', 'Charlotte'
        ], num_records),
        'DeviceID': np.random.choice([f'DEV{i:03d}' for i in range(50)], num_records),
        'IP Address': [f'192.168.{np.random.randint(0,255)}.{np.random.randint(0,255)}' for _ in range(num_records)],
        'MerchantID': np.random.choice([f'MERCH{i:03d}' for i in range(200)], num_records),
        'AccountBalance': np.round(np.random.lognormal(mean=7, sigma=0.8, size=num_records), 2) * 100,
        'PreviousTransactionDate': pd.to_datetime(pd.date_range('2023-12-01', periods=num_records, freq='h').to_numpy() +
                                                  pd.to_timedelta(np.random.randint(-720, 0, num_records), unit='h')),
        'Channel': np.random.choice(['Online', 'ATM', 'Branch', 'Mobile'], num_records, p=[0.4, 0.25, 0.2, 0.15]),
        'CustomerAge': np.random.randint(18, 70, num_records),
        'CustomerOccupation': np.random.choice([
            'Engineer', 'Doctor', 'Student', 'Retired', 'Artist',
            'Teacher', 'Businessman', 'Analyst', 'Consultant', 'Chef', 'Lawyer'
        ], num_records),
        'TransactionDuration': np.round(np.random.uniform(5, 120, num_records), 0),
        'LoginAttempts': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], num_records, p=[0.5, 0.2, 0.1, 0.05, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01]),
        'is_fraudulent': np.random.choice([True, False], num_records, p=[0.05, 0.95]),
        'transaction_details_str_for_llm': ['Sample transaction contextual string' for _ in range(num_records)],
        'transaction_embedding_shape': [np.random.rand(128).tolist() for _ in range(num_records)],
        'fraud_context': [np.random.choice([
            None,
            'Unusual overnight transaction activity',
            'Repeated failed authentication attempts prior to transaction',
            'High-value transaction in unnotified location',
            'Transaction executed via flagged proxy IP address',
            'Spending pattern sharply deviates from customer baseline',
            'Unrecognized device signature for account history',
            'Abnormally rapid transaction duration'
        ], 1, p=[0.8, 0.03, 0.05, 0.02, 0.03, 0.04, 0.02, 0.01])[0] for _ in range(num_records)],
        'explanation': [np.random.choice([
            None,
            'System identified anomalous spending deviation exceeding habitual limits for this channel and occupation.',
            'Alert triggered by anomalous execution duration coupled with IP mismatch from customer geographic history.',
            'Repeated failed login attempts followed by high-value debit. Context matches account takeover patterns.',
            'Suspicious transaction executed from an unregistered device with elevated transaction frequency.',
            'Transaction executed from an unnotified geographic location with substantial account balance depletion.'
        ], 1, p=[0.8, 0.05, 0.05, 0.04, 0.03, 0.03])[0] for _ in range(num_records)],
        'detected_risk_level': [np.random.choice(risk_levels_consistent) if np.random.rand() < 0.2 else None for _ in range(num_records)],
        'detected_rule_tags': [np.random.choice([['geolocation', 'behavioral'], ['authentication'], ['spending-pattern', 'timing'], ['device']], 1, p=[0.25,0.25,0.25,0.25])[0] if np.random.rand() < 0.1 else None for _ in range(num_records)]
    }

    df = pd.DataFrame(data)

    account_initial_balance = {acc_id: np.random.lognormal(mean=7, sigma=0.8) * 100 for acc_id in df['AccountID'].unique()}
    df['AccountBalance'] = df['AccountID'].map(account_initial_balance)
    df = df.sort_values(by=['AccountID', 'TransactionDate']).reset_index(drop=True)

    for index, row in df.iterrows():
        if row['TransactionType'] == 'Debit':
            df.loc[index, 'AccountBalance'] -= row['TransactionAmount']
        elif row['TransactionType'] == 'Credit':
            df.loc[index, 'AccountBalance'] += row['TransactionAmount']
        df.loc[index, 'AccountBalance'] = max(0, df.loc[index, 'AccountBalance'])

    df.loc[df['is_fraudulent'] == False, ['fraud_context', 'explanation', 'detected_risk_level', 'detected_rule_tags']] = None

    return df

# --- Data Loading and Processing ---
@st.cache_data
def load_transactions_data(file_source):
    """
    Parses and cleans JSON Lines data into a Pandas DataFrame.
    """
    try:
        if isinstance(file_source, str):
            if not os.path.exists(file_source):
                st.error(f"Error: File not found at path: `{file_source}`")
                return None
            df = pd.read_json(file_source, lines=True)
        else:
            df = pd.read_json(file_source, lines=True)

        for col in ['TransactionDate', 'PreviousTransactionDate']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        df.dropna(subset=['TransactionDate'], inplace=True)

        numeric_cols = ['TransactionAmount', 'AccountBalance', 'LoginAttempts', 'CustomerAge', 'TransactionDuration']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['TransactionAmount', 'AccountBalance', 'LoginAttempts', 'CustomerAge'], inplace=True)

        if 'is_fraudulent' in df.columns:
            df['is_fraudulent'] = df['is_fraudulent'].astype(bool)
        else:
            df['is_fraudulent'] = False

        fraud_related_cols = ['fraud_context', 'explanation', 'detected_risk_level', 'detected_rule_tags']
        for col in fraud_related_cols:
            if col not in df.columns:
                df[col] = None

        df.loc[df['is_fraudulent'] == False, fraud_related_cols] = None

        return df
    except ValueError as e:
        st.error(f"Error parsing JSON Lines file: Ensure proper schema and formatting. Details: `{e}`")
        return None
    except Exception as e:
        st.error(f"Unexpected error loading file: `{e}`")
        return None

# --- Main Dashboard Header ---
st.title("⚡ FraudAI - Banking Transaction Analytics")
st.markdown("Interactive fraud detection dashboard. Explore transactional KPIs, cybersecurity risk signals, and customer demographics.")

# Data Ingestion Section
st.markdown("---")
st.subheader("Data Ingestion")
uploaded_file = st.file_uploader(
    "Upload processed JSON Lines (.jsonl) file",
    type=["jsonl"],
    help="Drag and drop your .jsonl file here. Include `is_fraudulent`, `fraud_context`, `explanation`, and `detected_risk_level` for complete analysis."
)

df_transactions = None
if uploaded_file is not None:
    st.info("File uploaded successfully. Loading data...")
    df_transactions = load_transactions_data(uploaded_file)
else:
    st.info("Upload a file or generate synthetic demo data to explore the dashboard.")
    if st.button("Generate Demo Data"):
        df_transactions = generate_sample_data()
        st.cache_data.clear()
        st.rerun()

if df_transactions is None or df_transactions.empty:
    st.warning("No data loaded. Please upload a file or generate demo data.")
    st.stop()

st.success(f"Transaction data loaded successfully: {len(df_transactions)} total records.")

# --- Filtering Options ---
st.markdown("---")
st.subheader("Filter Options")

col_filter1, col_filter2, col_filter3, col_filter4 = st.columns(4)

with col_filter1:
    min_date_data = df_transactions['TransactionDate'].min().date()
    max_date_data = df_transactions['TransactionDate'].max().date()
    date_range_selected = st.date_input(
        "Date Range",
        value=(min_date_data, max_date_data),
        min_value=min_date_data,
        max_value=max_date_data
    )
    if len(date_range_selected) == 2:
        df_filtered = df_transactions[(df_transactions['TransactionDate'].dt.date >= date_range_selected[0]) &
                                     (df_transactions['TransactionDate'].dt.date <= date_range_selected[1])]
    else:
        df_filtered = df_transactions.copy()

with col_filter2:
    if 'TransactionType' in df_filtered.columns:
        selected_types = st.multiselect(
            "Transaction Type",
            options=df_filtered['TransactionType'].unique(),
            default=df_filtered['TransactionType'].unique()
        )
        df_filtered = df_filtered[df_filtered['TransactionType'].isin(selected_types)]

with col_filter3:
    if 'Channel' in df_filtered.columns:
        selected_channels = st.multiselect(
            "Channel",
            options=df_filtered['Channel'].unique(),
            default=df_filtered['Channel'].unique()
        )
        df_filtered = df_filtered[df_filtered['Channel'].isin(selected_channels)]

with col_filter4:
    show_fraud_only = st.checkbox("Show Fraudulent Transactions Only", value=False)
    if show_fraud_only:
        df_filtered = df_filtered[df_filtered['is_fraudulent'] == True]

if df_filtered.empty:
    st.warning("No transactions match the selected filters.")
    st.stop()

st.info(f"Displaying {len(df_filtered)} filtered transactions.")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔒 Cybersecurity & Fraud", "👥 Customer Insights"])

# TAB 1: OVERVIEW
with tab1:
    st.header("Transactional Performance Overview")
    st.markdown("Key transactional metrics, volume trends, and channel distributions.")

    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        st.metric("Total Transactions", f"{len(df_filtered):,}")
    with col_kpi2:
        st.metric("Total Volume Transacted", f"${df_filtered['TransactionAmount'].sum():,.2f}")
    with col_kpi3:
        st.metric("Average Transaction Amount", f"${df_filtered['TransactionAmount'].mean():,.2f}")

    st.subheader("Historical Volume and Transaction Trends")
    if 'TransactionDate' in df_filtered.columns and 'TransactionAmount' in df_filtered.columns:
        time_agg_general = st.radio("Time Aggregation:", ('Day', 'Week', 'Month'), key='time_agg_general', horizontal=True)
        resample_freq = 'D' if time_agg_general == 'Day' else ('W' if time_agg_general == 'Week' else 'M')
        df_trend_general = df_filtered.set_index('TransactionDate').resample(resample_freq).agg(
            Total_Transactions=('TransactionID', 'count'),
            Transacted_Volume=('TransactionAmount', 'sum')
        ).reset_index()

        fig_general_trend = go.Figure()
        fig_general_trend.add_trace(go.Scatter(x=df_trend_general['TransactionDate'], y=df_trend_general['Total_Transactions'], mode='lines+markers', name='Total Transactions', line=dict(color='blue')))
        fig_general_trend.add_trace(go.Scatter(x=df_trend_general['TransactionDate'], y=df_trend_general['Transacted_Volume'], mode='lines+markers', name='Volume ($)', yaxis='y2', line=dict(color='green')))
        fig_general_trend.update_layout(
            title='Transaction Volume & Count Trends',
            xaxis_title='Date',
            yaxis_title='Total Transactions',
            yaxis2=dict(title='Volume ($)', overlaying='y', side='right'),
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig_general_trend, use_container_width=True)

    col_general1, col_general2 = st.columns(2)
    with col_general1:
        st.subheader("Transactions by Type")
        if 'TransactionType' in df_filtered.columns:
            type_counts_general = df_filtered['TransactionType'].value_counts().reset_index()
            type_counts_general.columns = ['TransactionType', 'Count']
            fig_general_type = px.pie(type_counts_general, names='TransactionType', values='Count',
                                      title='Proportion by Transaction Type',
                                      color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_general_type, use_container_width=True)

    with col_general2:
        st.subheader("Transactions by Channel")
        if 'Channel' in df_filtered.columns:
            channel_counts_general = df_filtered['Channel'].value_counts().reset_index()
            channel_counts_general.columns = ['Channel', 'Count']
            fig_general_channel = px.bar(channel_counts_general, x='Count', y='Channel', orientation='h',
                                         title='Transaction Count by Channel',
                                         labels={'Count': 'Count', 'Channel': 'Channel'},
                                         color='Channel',
                                         color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_general_channel, use_container_width=True)

# TAB 2: CYBERSECURITY & FRAUD
with tab2:
    st.header("🔒 Cybersecurity & Fraud: Deep Anomaly Analysis")
    st.markdown("Designed for fraud operations and risk teams to investigate alerts, explainability, and attack signatures.")

    fraud_df = df_filtered[df_filtered['is_fraudulent'] == True]

    st.subheader("Fraud Metrics & Detection KPIs")
    col_fraud_kpi1, col_fraud_kpi2, col_fraud_kpi3 = st.columns(3)
    with col_fraud_kpi1:
        st.metric("Fraudulent Transactions Detected", f"{len(fraud_df):,}")
    with col_fraud_kpi2:
        if 'TransactionAmount' in fraud_df.columns:
            st.metric("Total Fraud Volume", f"${fraud_df['TransactionAmount'].sum():,.2f}")
        else:
            st.metric("Total Fraud Volume", "N/A")
    with col_fraud_kpi3:
        if len(df_filtered) > 0:
            st.metric("Fraud Rate (%)", f"{len(fraud_df) / len(df_filtered) * 100:,.2f}%")
        else:
            st.metric("Fraud Rate (%)", "0.00%")

    if fraud_df.empty:
        st.info("No fraudulent transactions identified within the current filter parameters.")
    else:
        st.markdown("---")
        st.subheader("Visual Anomaly Detection (Scatter Plots)")
        col_anom1, col_anom2 = st.columns(2)
        with col_anom1:
            if 'TransactionAmount' in df_filtered.columns and 'TransactionDuration' in df_filtered.columns:
                fig_amount_duration = px.scatter(
                    df_filtered, x='TransactionDuration', y='TransactionAmount', color='is_fraudulent',
                    title='Amount vs. Transaction Duration',
                    labels={'TransactionDuration': 'Duration (seconds)', 'TransactionAmount': 'Amount ($)'},
                    hover_data=['TransactionID', 'AccountID', 'TransactionType', 'fraud_context', 'explanation', 'detected_risk_level'],
                    color_discrete_map={True: 'red', False: 'blue'},
                    size='TransactionAmount', log_y=True
                )
                st.plotly_chart(fig_amount_duration, use_container_width=True)

        with col_anom2:
            if 'LoginAttempts' in df_filtered.columns and 'TransactionAmount' in df_filtered.columns:
                fig_login_amount = px.scatter(
                    df_filtered, x='LoginAttempts', y='TransactionAmount', color='is_fraudulent',
                    title='Amount vs. Login Attempts',
                    labels={'LoginAttempts': 'Login Attempts', 'TransactionAmount': 'Amount ($)'},
                    hover_data=['TransactionID', 'AccountID', 'TransactionType', 'Location', 'fraud_context', 'explanation', 'detected_risk_level'],
                    color_discrete_map={True: 'red', False: 'blue'},
                    size='TransactionAmount', log_y=True
                )
                st.plotly_chart(fig_login_amount, use_container_width=True)

        st.markdown("---")
        st.subheader("Fraud Trends & Geographic Hotspots")
        col_fraud_dist1, col_fraud_dist2 = st.columns(2)
        with col_fraud_dist1:
            if 'Channel' in fraud_df.columns:
                fig_fraud_channel = px.pie(fraud_df, names='Channel', title='Fraud by Channel', hole=0.3, color_discrete_sequence=px.colors.qualitative.Bold)
                st.plotly_chart(fig_fraud_channel, use_container_width=True)

        with col_fraud_dist2:
            if 'TransactionDate' in fraud_df.columns:
                fraud_trend_daily = fraud_df.set_index('TransactionDate').resample('D').agg(Fraud_Count=('TransactionID', 'count')).reset_index()
                fig_fraud_trend = px.line(
                    fraud_trend_daily, x='TransactionDate', y='Fraud_Count',
                    title='Daily Fraud Incidents',
                    labels={'TransactionDate': 'Date', 'Fraud_Count': 'Incidents'},
                    markers=True, line_shape='spline', color_discrete_sequence=['red']
                )
                st.plotly_chart(fig_fraud_trend, use_container_width=True)

        col_hotspot1, col_hotspot2 = st.columns(2)
        with col_hotspot1:
            if 'Location' in fraud_df.columns:
                city_to_state = {
                    'New York': 'NY', 'Los Angeles': 'CA', 'Chicago': 'IL', 'Houston': 'TX',
                    'Phoenix': 'AZ', 'Philadelphia': 'PA', 'San Antonio': 'TX', 'San Diego': 'CA',
                    'Dallas': 'TX', 'San Jose': 'CA', 'Austin': 'TX', 'Jacksonville': 'FL',
                    'Fort Worth': 'TX', 'Columbus': 'OH', 'Charlotte': 'NC', 'Memphis': 'TN',
                    'Denver': 'CO', 'Seattle': 'WA', 'Miami': 'FL', 'Boston': 'MA',
                    'Detroit': 'MI', 'Washington': 'DC', 'Nashville': 'TN', 'Portland': 'OR',
                    'Las Vegas': 'NV', 'Atlanta': 'GA', 'Orlando': 'FL', 'Minneapolis': 'MN'
                }
                fraud_df_copy = fraud_df.copy()
                fraud_df_copy['State'] = fraud_df_copy['Location'].map(city_to_state)
                state_fraud_counts = fraud_df_copy.dropna(subset=['State'])['State'].value_counts().reset_index()
                state_fraud_counts.columns = ['State', 'FraudulentTransactions']

                if not state_fraud_counts.empty:
                    fig_state_fraud = px.choropleth(
                        state_fraud_counts,
                        locations='State',
                        locationmode='USA-states',
                        color='FraudulentTransactions',
                        scope='usa',
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title='Fraud Frequency by US State'
                    )
                    st.plotly_chart(fig_state_fraud, use_container_width=True)

        with col_hotspot2:
            if 'DeviceID' in fraud_df.columns:
                device_fraud_counts = fraud_df['DeviceID'].value_counts().nlargest(10).reset_index()
                device_fraud_counts.columns = ['DeviceID', 'FraudulentTransactions']
                fig_fraud_dev = px.bar(
                    device_fraud_counts, x='DeviceID', y='FraudulentTransactions',
                    title='Top 10 High-Risk Devices',
                    labels={'DeviceID': 'Device ID', 'FraudulentTransactions': 'Fraud Count'},
                    color='DeviceID', color_discrete_sequence=px.colors.qualitative.G10
                )
                st.plotly_chart(fig_fraud_dev, use_container_width=True)

        st.markdown("---")
        st.subheader("Detailed Fraudulent Transaction Investigation")
        display_columns = [
            'TransactionID', 'TransactionDate', 'TransactionAmount', 'AccountID',
            'is_fraudulent', 'detected_risk_level', 'detected_rule_tags',
            'Location', 'Channel', 'CustomerAge', 'explanation'
        ]
        available_display_columns = [col for col in display_columns if col in fraud_df.columns]

        if not fraud_df.empty:
            risk_level_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "N/A"]
            fraud_df['detected_risk_level_ordered'] = pd.Categorical(fraud_df['detected_risk_level'].fillna("N/A"), categories=risk_level_order, ordered=True)
            table_df = fraud_df.sort_values('detected_risk_level_ordered', ascending=True)[available_display_columns]

            if 'detected_rule_tags' in table_df.columns:
                table_df['detected_rule_tags'] = table_df['detected_rule_tags'].apply(
                    lambda x: ", ".join(x) if isinstance(x, list) else (x if x is not None else "N/A")
                )

            st.dataframe(table_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("Selected Transaction Rationale Inspector")
            selected_transaction_id = st.selectbox(
                "Select Transaction ID for in-depth explanation:",
                options=fraud_df['TransactionID'].unique()
            )

            if selected_transaction_id:
                selected_transaction = fraud_df[fraud_df['TransactionID'] == selected_transaction_id].iloc[0]
                st.json(selected_transaction.to_dict())

                st.markdown(f"### Audit Report for {selected_transaction_id}")
                st.markdown(f"**Assessed Risk Level:** `{selected_transaction.get('detected_risk_level', 'N/A')}`")
                st.markdown(f"**Activated Rule Tags:** `{', '.join(selected_transaction.get('detected_rule_tags', ['N/A'])) if isinstance(selected_transaction.get('detected_rule_tags'), list) else selected_transaction.get('detected_rule_tags', 'N/A')}`")

                with st.expander("View RAG Context & Matched Rules"):
                    st.write(selected_transaction.get('fraud_context', 'No context available.'))

                with st.expander("View Natural Language LLM Explanation"):
                    st.write(selected_transaction.get('explanation', 'No explanation available.'))

# TAB 3: CUSTOMER INSIGHTS
with tab3:
    st.header("👥 Customer Insights & Behavioral Segmentation")
    st.markdown("Examine demographic distribution, occupational tendencies, and spending behavior across customer segments.")

    if df_filtered.empty:
        st.info("No data available for customer insights with current filters.")
    else:
        col_customer1, col_customer2 = st.columns(2)
        with col_customer1:
            if 'CustomerAge' in df_filtered.columns:
                fig_age = px.histogram(
                    df_filtered, x='CustomerAge', title='Customer Age Distribution',
                    labels={'CustomerAge': 'Customer Age'}, color_discrete_sequence=['purple'], marginal='box'
                )
                st.plotly_chart(fig_age, use_container_width=True)

        with col_customer2:
            if 'CustomerOccupation' in df_filtered.columns:
                occupation_counts = df_filtered['CustomerOccupation'].value_counts().reset_index()
                occupation_counts.columns = ['Occupation', 'Count']
                fig_occupation = px.bar(
                    occupation_counts, x='Count', y='Occupation', orientation='h',
                    title='Transactions by Customer Occupation',
                    labels={'Count': 'Transaction Count', 'Occupation': 'Occupation'},
                    color='Occupation', color_discrete_sequence=px.colors.qualitative.Prism
                )
                st.plotly_chart(fig_occupation, use_container_width=True)

        if 'CustomerOccupation' in df_filtered.columns and 'TransactionAmount' in df_filtered.columns:
            st.subheader("Average Spending Amount by Occupation")
            avg_amount_by_occ = df_filtered.groupby('CustomerOccupation')['TransactionAmount'].mean().reset_index()
            fig_avg_occ = px.bar(
                avg_amount_by_occ, x='CustomerOccupation', y='TransactionAmount',
                title='Average Transaction Amount by Occupation ($)',
                labels={'CustomerOccupation': 'Occupation', 'TransactionAmount': 'Average Amount ($)'},
                color='CustomerOccupation', color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_avg_occ, use_container_width=True)