import pandas as pd
from src.proxy_target import calculate_rfm, assign_high_risk_label

def test_calculate_rfm():
    data = {
        'customer_id': [1, 1, 2, 2, 3],
        'transaction_amount': [100, 200, 150, 50, 300],
        'transaction_datetime': [
            '2023-01-01', '2023-01-10', '2023-01-05', '2023-01-20', '2023-01-15'
        ]
    }
    df = pd.DataFrame(data)
    rfm = calculate_rfm(df, 'customer_id', 'transaction_amount', 'transaction_datetime', snapshot_date=pd.Timestamp('2023-01-31'))
    assert 'recency' in rfm.columns
    assert 'frequency' in rfm.columns
    assert 'monetary' in rfm.columns
    assert rfm.shape[0] == 3

def test_assign_high_risk_label():
    rfm = pd.DataFrame({
        'customer_id': [1, 2, 3],
        'recency': [30, 20, 10],
        'frequency': [1, 2, 3],
        'monetary': [100, 200, 300]
    })
    result = assign_high_risk_label(rfm, n_clusters=3, random_state=42)
    assert 'is_high_risk' in result.columns
    assert set(result['is_high_risk']).issubset({0, 1})
    assert result.shape[0] == 3
