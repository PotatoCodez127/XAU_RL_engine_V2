import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class XAUDynamicEnv(gym.Env):
    """
    A custom trading environment for XAUUSD that supports a 3D Action Space.
    Allows the RL agent to dynamically set Take Profit and Stop Loss limits.
    """
    metadata = {'render_modes': ['human', 'console']}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0, window_size: int = 30):
        super().__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.window_size = window_size
        
        # Action Space: Continuous [-1, 1] for SAC
        # [0]: Direction & Size (< 0 = Short, > 0 = Long, magnitude = position size)
        # [1]: TP Multiplier (mapped to 0.5x - 5.0x ATR)
        # [2]: SL Multiplier (mapped to 0.5x - 3.0x ATR)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation Space: Assuming the Oracle provides compressed features (e.g., 15 dims)
        # plus portfolio metrics (drawdown, current equity ratio). Total = 17 dims.
        self.obs_dim = 17 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        self.reset()

    def _map_action(self, action: np.ndarray):
        """Translates [-1, 1] neural network outputs to real-world trading parameters."""
        raw_direction = action[0]
        direction = 1 if raw_direction > 0 else -1 if raw_direction < 0 else 0
        size_pct = abs(raw_direction) # How much of the portfolio to risk
        
        # Map TP from [-1, 1] to [0.5, 5.0]
        tp_mult = np.interp(action[1], [-1, 1], [0.5, 5.0])
        
        # Map SL from [-1, 1] to [0.5, 3.0]
        sl_mult = np.interp(action[2], [-1, 1], [0.5, 3.0])
        
        return direction, size_pct, tp_mult, sl_mult

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        # Placeholder for actual Oracle feature integration
        # In the full engine, this will pull the current market state and append
        # the equity metrics: [current_equity / initial_balance, current_drawdown]
        drawdown = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0
        equity_ratio = self.equity / self.initial_balance
        
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[-2:] = [equity_ratio, drawdown]
        return obs

    def step(self, action: np.ndarray):
        direction, size_pct, tp_mult, sl_mult = self._map_action(action)
        
        # --- Simulated Physics for Testing ---
        # In the real loop, this accesses self.df.iloc[self.current_step] 
        # to calculate distance-to-TP vs distance-to-SL using high/low prices.
        
        # Simulate a generic outcome for the sake of the structural test
        simulated_pnl = 0.0
        if direction != 0:
            # Randomly hit TP or SL based on a naive probability for structural testing
            hit_tp = np.random.rand() > 0.6 
            if hit_tp:
                simulated_pnl = (self.balance * size_pct) * (tp_mult * 0.01)
            else:
                simulated_pnl = -(self.balance * size_pct) * (sl_mult * 0.01)

        self.balance += simulated_pnl
        self.equity = self.balance
        
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
            
        drawdown = (self.peak_equity - self.equity) / self.peak_equity
        
        # Risk-Adjusted Reward Shaping (Sortino proxy)
        # We severely penalize large drawdowns to force the SAC to prefer stable growth
        reward = simulated_pnl - (drawdown * self.initial_balance * 0.5)
        
        self.current_step += 1
        
        # Termination conditions
        terminated = self.equity <= self.initial_balance * 0.1 # Blown account
        truncated = self.current_step >= len(self.df) - 1
        
        info = {
            "balance": self.balance,
            "drawdown": drawdown,
            "tp_mult_used": tp_mult,
            "sl_mult_used": sl_mult
        }
        
        return self._get_observation(), float(reward), terminated, truncated, info