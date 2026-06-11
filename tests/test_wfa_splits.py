import pytest
import pandas as pd
import numpy as np
from data.wfa_pipeline import WalkForwardPipeline

@pytest.fixture
def mock_market_data(tmp_path):
    """Generates a synthetic dataset to test pipeline logic without needing real CSVs."""
    dates = pd.date_range(start='2023-01-01', periods=2000, freq='h')
    
    xau = pd.DataFrame({'close': np.random.randn(2000)}, index=dates)
    dxy = pd.DataFrame({'close_dxy': np.random.randn(2000)}, index=dates)

    # Intentionally inject a large block of missing DXY data
    dxy.iloc[50:100] = np.nan

    xau_path = tmp_path / "xau_mock.csv"
    dxy_path = tmp_path / "dxy_mock.csv"

    xau.reset_index(names='time').to_csv(xau_path, index=False)
    dxy.reset_index(names='time').to_csv(dxy_path, index=False)

    return str(xau_path), str(dxy_path)

def test_missing_data_retention(mock_market_data):
    """Ensures DXY correlation gaps don't empty the entire dataset."""
    xau_path, dxy_path = mock_market_data
    pipeline = WalkForwardPipeline(xau_path, dxy_path)
    df = pipeline.load_and_merge()

    # The dataset should remain intact (~2000 rows), not collapse to 0.
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

    # Assert the gap in indices matches our strict embargo rule
    assert (test_start_idx - train_end_idx) == embargo