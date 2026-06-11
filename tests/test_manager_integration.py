import pytest
import pandas as pd
import numpy as np
from stable_baselines3 import SAC
from env.xau_dynamic_env import XAUDynamicEnv

@pytest.fixture
def dummy_enriched_data():
    """Simulates the WFA data after it has passed through the Oracle."""
    dates = pd.date_range('2023-01-01', periods=500, freq='h')
    df = pd.DataFrame({
        'close': np.random.randn(500),
        'prob_hold': np.random.uniform(0, 1, 500),
        'prob_long': np.random.uniform(0, 1, 500),
        'prob_short': np.random.uniform(0, 1, 500)
    }, index=dates)
    return df

def test_sac_environment_compatibility(dummy_enriched_data):
    """
    Verifies that the Stable-Baselines3 SAC algorithm can initialize 
    and interact with our custom 3D Action Space environment.
    """
    env = XAUDynamicEnv(df=dummy_enriched_data)
    
    # Initialize a lightweight SAC model purely for testing graph connections
    model = SAC("MlpPolicy", env, policy_kwargs=dict(net_arch=[32, 32]), learning_starts=10)
    
    try:
        # Run a micro-training loop to ensure the step/reset loop doesn't crash
        model.learn(total_timesteps=20)
    except Exception as e:
        pytest.fail(f"SAC integration failed during training step: {e}")
        
    # Verify the model can predict an action
    obs, _ = env.reset()
    action, _states = model.predict(obs, deterministic=True)
    
    # Assert the action matches our 3D space requirement [Direction, TP, SL]
    assert action.shape == (3,)
    assert -1.0 <= action[0] <= 1.0