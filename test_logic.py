import pytest
import pandas as pd
import numpy as np

# Assuming bitcoinmoon.py is in the same directory or the path is configured.
from bitcoinmoon import calculate_cmf, calculate_rsi

@pytest.fixture
def sample_data():
    """Loads the test data from a CSV file for testing."""
    df = pd.read_csv('test_data.csv', parse_dates=['Date'], index_col='Date')
    # Ensure dataframe is not empty
    assert not df.empty, "Test data could not be loaded or is empty."
    return df

def test_data_loading(sample_data):
    """Tests that the sample data fixture loads correctly."""
    df = sample_data
    assert isinstance(df, pd.DataFrame)
    expected_cols = ['open', 'high', 'low', 'close', 'volume']
    assert all(col in df.columns for col in expected_cols)
    assert df.index.name == 'Date'

def test_calculate_cmf(sample_data):
    """Tests the CMF calculation logic."""
    df = sample_data
    cmf_series = calculate_cmf(df)
    
    # Check if the result is a pandas Series
    assert isinstance(cmf_series, pd.Series)
    
    # Check that it's not all NaN
    assert not cmf_series.isnull().all(), "CMF calculation resulted in all NaNs."
    
    # Check that the length matches the input dataframe
    assert len(cmf_series) == len(df)
    
    # Optional: Check a specific calculated value for correctness if we have a known benchmark
    # For now, we just check the type and that it's not empty.
    # Example: assert np.isclose(cmf_series.iloc[-1], -0.1, atol=0.05)


def test_calculate_rsi(sample_data):
    """Tests the RSI calculation logic."""
    df = sample_data
    rsi_series = calculate_rsi(df)
    
    # Check if the result is a pandas Series
    assert isinstance(rsi_series, pd.Series)
    
    # The first (period-1) values will be NaN, which is expected.
    # We check that not ALL values are NaN.
    assert not rsi_series.isnull().all(), "RSI calculation resulted in all NaNs."
    
    # Check that the length matches the input dataframe
    assert len(rsi_series) == len(df)
    
    # RSI values should be between 0 and 100
    # We check the non-NaN values
    valid_rsi_values = rsi_series.dropna()
    assert (valid_rsi_values >= 0).all() and (valid_rsi_values <= 100).all()
    
    # Example: assert np.isclose(rsi_series.iloc[-1], 60.0, atol=1.0)