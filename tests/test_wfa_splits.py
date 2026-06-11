import pytest
import pandas as pd
import numpy as np
from data.wfa_pipeline import WalkForwardPipeline

@pytest.fixture
def mock_market_data(tmp_path):
    """Generates synthetic MetaTrader-formatted datasets."""
    dates = pd.date_range(start='2023-01-01', periods=2000, freq='h')
    
    # Simulate MetaTrader structure
    data_template = {
        '<DATE>': dates.strftime('%Y.%m.%d'),
        '<TIME>': dates.strftime('%H:%M:%S'),
        '<OPEN>': np.random.randn(2000),
        '<HIGH>': np.random.randn(2000),
        '<LOW>': np.random.randn(2000),
        '<CLOSE>': np.random.randn(2000),
        '<TICKVOL>': 1,
        '<VOL>': 0,
        '<SPREAD>': 4
    }
    
    xau = pd.DataFrame(data_template)
    dxy = pd.DataFrame(data_template)

    # Intentionally inject missing data into DXY
    dxy.loc[50:100, '<CLOSE>'] = np.nan

    xau_path = tmp_path / "xau_mock.csv"
    dxy_path = tmp_path / "dxy_mock.csv"

    # Export as tab-delimited exactly like MT5
    xau.to_csv(xau_path, sep='\t', index=False)
    dxy.to_csv(dxy_path, sep='\t', index=False)

    return str(xau_path), str(dxy_path)

def test_missing_data_retention(mock_market_data):
    """Ensures DXY correlation gaps don't empty the entire dataset."""
    xau_path, dxy_path = mock_market_data
    pipeline = WalkForwardPipeline(xau_path, dxy_path)
    df = pipeline.load_and_merge()

    assert len(df) > 1900
    assert not df['close_dxy'].isnull().any()

def test_embargo_leakage_prevention(mock_market_data):
    """Verifies strict separation between train and test sets."""
    xau_path, dxy_path = mock_market_data
    embargo = 100
    pipeline = WalkForwardPipeline(xau_path, dxy_path, embargo_bars=embargo)
    pipeline.load_and_merge()

    splits = pipeline.generate_splits(train_size=500, test_size=200, step_size=200)
    first_split = splits[0]

    train_end_idx = pipeline.master_df.index.get_loc(first_split['train'].index[-1])
    test_start_idx = pipeline.master_df.index.get_loc(first_split['test'].index[0])

    assert (test_start_idx - train_end_idx - 1) == embargo