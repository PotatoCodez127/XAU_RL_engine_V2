import pytest
import numpy as np
import pandas as pd
from env.xau_dynamic_env import XAUDynamicEnv

@pytest.fixture
def dummy_env():
    # Provide a minimal DataFrame just to initialize the environment
    dates = pd.date_range('2023-01-01', periods=100, freq='h')
    df = pd.DataFrame({'close': np.random.randn(100)}, index=dates)
    return XAUDynamicEnv(df=df)

def test_action_space_mapping(dummy_env):
    """Ensures SAC continuous actions map correctly to operational multipliers."""
    
    # Test Maximum Action (Full Long, Max TP, Max SL)
    max_action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    dir_max, size_max, tp_max, sl_max = dummy_env._map_action(max_action)
    
    assert dir_max == 1
    assert size_max == 1.0
    assert tp_max == 5.0 # Max boundary
    assert sl_max == 3.0 # Max boundary
    
    # Test Minimum Action (Full Short, Min TP, Min SL)
    min_action = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    dir_min, size_min, tp_min, sl_min = dummy_env._map_action(min_action)
    
    assert dir_min == -1
    assert size_min == 1.0 # Size is absolute magnitude
    assert tp_min == 0.5 # Min boundary
    assert sl_min == 0.5 # Min boundary

def test_environment_step_mechanics(dummy_env):
    """Verifies the environment successfully advances a step and returns correct shapes."""
    dummy_env.reset()
    
    action = dummy_env.action_space.sample()
    obs, reward, terminated, truncated, info = dummy_env.step(action)
    
    assert obs.shape == (dummy_env.obs_dim,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    
    # Verify the info dictionary tracked the dynamic risk parameters
    assert "tp_mult_used" in info
    assert "sl_mult_used" in info
    assert "drawdown" in info