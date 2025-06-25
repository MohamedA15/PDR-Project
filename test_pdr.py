import pytest
import numpy as np
from main import (
    kalman_filter,
    detect_stationary_periods,
    detect_steps,
    calculate_rmse,
    heading_stability_metric,
)

# Test fixtures
@pytest.fixture
def synthetic_accel_data():
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    clean = np.sin(2 * np.pi * 0.5 * t)
    noisy = clean + 0.1 * np.random.randn(len(t))
    return clean, noisy

@pytest.fixture
def synthetic_stationary_data():
    np.random.seed(42)
    # Create synthetic data with known stationary/moving periods
    accel_stationary = np.random.normal(0, 0.05, 100)  # Low variance
    gyro_stationary = np.random.normal(0, 0.02, 100)
    accel_moving = np.random.normal(0, 1.0, 100)      # High variance
    gyro_moving = np.random.normal(0, 0.5, 100)
    return (
        np.concatenate([accel_stationary, accel_moving]),
        np.concatenate([gyro_stationary, gyro_moving])
    )

# Test cases
def test_kalman_filter_variance_reduction(synthetic_accel_data):
    """Test that Kalman filter reduces noise variance"""
    clean, noisy = synthetic_accel_data
    filtered = kalman_filter(noisy)
    
    original_variance = np.var(noisy - clean)
    filtered_variance = np.var(filtered - clean)
    
    assert filtered_variance < original_variance
    assert np.isclose(filtered.mean(), clean.mean(), rtol=0.1)

def test_stationary_detection(synthetic_stationary_data):
    """Test that stationary periods are correctly identified"""
    accel, gyro = synthetic_stationary_data
    stationary = detect_stationary_periods(accel, gyro)
    
    # First half should be stationary (low variance)
    assert np.mean(stationary[:100]) > 0.8
    # Second half should be moving (high variance)
    assert np.mean(stationary[100:]) < 0.2

def test_step_detection():
    """Test step detection in synthetic data"""
    # Create signal with peaks every 50 samples
    signal = np.zeros(500)
    signal[::50] = 2.0 
    signal = np.convolve(signal, np.ones(10)/10, mode='same') 
    
    stationary = np.zeros_like(signal, dtype=bool)
    steps = detect_steps(signal, stationary)
    
    assert len(steps) >= 9  
    assert np.all(np.diff(steps) >= 45)  
def test_rmse_calculation():
    """Test RMSE calculation with known values"""
    true = np.array([[0, 0], [1, 1], [2, 2]])
    pred = np.array([[0, 0], [1, 1.1], [2, 2.2]])
    expected = np.sqrt((0 + 0.01 + 0.04)/3)
    assert np.isclose(calculate_rmse(true, pred), expected)

def test_heading_stability():
    """Test heading stability metric"""
    # Stable heading (small changes)
    stable = np.cumsum(np.random.normal(0, 0.01, 100))
    assert heading_stability_metric(stable) < 0.05
    
    # Changing heading (larger changes)
    changing = np.cumsum(np.random.normal(0, 0.1, 100))
    assert heading_stability_metric(changing) > 0.05