import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def calculate_rfm(df, customer_id_col='customer_id', amount_col='transaction_amount', date_col='transaction_datetime', snapshot_date=None):
    df[date_col] = pd.to_datetime(df[date_col])
    if snapshot_date is None:
        snapshot_date = df[date_col].max() + pd.Timedelta(days=1)
    rfm = df.groupby(customer_id_col).agg({
        date_col: lambda x: (snapshot_date - x.max()).days,
        amount_col: ['count', 'sum']
    })
    rfm.columns = ['recency', 'frequency', 'monetary']
    rfm = rfm.reset_index()
    return rfm

def assign_high_risk_label(rfm_df, n_clusters=3, random_state=42):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['recency', 'frequency', 'monetary']])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(rfm_scaled)
    rfm_df['cluster'] = clusters
    # High risk: cluster with highest recency, lowest frequency & monetary
    cluster_stats = rfm_df.groupby('cluster').agg({'recency':'mean', 'frequency':'mean', 'monetary':'mean'})
    high_risk_cluster = cluster_stats.sort_values(['recency', 'frequency', 'monetary'], ascending=[False, True, True]).index[0]
    rfm_df['is_high_risk'] = (rfm_df['cluster'] == high_risk_cluster).astype(int)
    return rfm_df[[rfm_df.columns[0], 'is_high_risk']]

# Example usage
if __name__ == '__main__':
    # df = pd.read_csv('path_to_raw_data.csv')
    # rfm = calculate_rfm(df)
    # high_risk = assign_high_risk_label(rfm)
    # df = df.merge(high_risk, on='customer_id', how='left')
    pass
