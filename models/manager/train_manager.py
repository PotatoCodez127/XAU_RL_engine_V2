import os
import torch
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from sklearn.preprocessing import StandardScaler

from env.xau_dynamic_env import XAUDynamicEnv
from models.oracle.attention_net import TemporalAttentionOracle

class ManagerPipeline:
    def __init__(self, features_path: str, dxy_path: str = "", oracle_weights_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Sniff the feature dimensions from the master dataset
        temp_df = pd.read_csv(features_path, nrows=1)
        exclude_cols = ['target', 'time', 'datetime', 'date']
        self.feature_cols = [c for c in temp_df.columns if c not in exclude_cols and not c.startswith('env_')]
        input_dim = len(self.feature_cols)

        # Initialize Oracle with the correct dimensions
        self.oracle = TemporalAttentionOracle(input_dim=input_dim, seq_len=30).to(self.device)
        if oracle_weights_path and os.path.exists(oracle_weights_path):
            self.oracle.load_state_dict(torch.load(oracle_weights_path, map_location=self.device))
        self.oracle.eval() # Freeze Oracle during Manager training

    def _precompute_oracle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Pre-computing Oracle pattern recognition...")
        
        scaler = StandardScaler()
        if len(df) > 0 and len(self.feature_cols) > 0:
            raw_features = scaler.fit_transform(df[self.feature_cols].values)
        else:
            raise ValueError("Dataframe is missing the required feature columns.")
        
        probs_list = np.zeros((len(df), 3))
        
        with torch.no_grad():
            for i in range(30, len(df)):
                window = raw_features[i-30:i]
                window_tensor = torch.FloatTensor(window).unsqueeze(0).to(self.device)
                logits = self.oracle(window_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                probs_list[i] = probs
                
        df['prob_hold'] = probs_list[:, 0]
        df['prob_long'] = probs_list[:, 1]
        df['prob_short'] = probs_list[:, 2]
        
        return df

    def train_wfa_split(self, split_idx: int, train_df: pd.DataFrame, val_df: pd.DataFrame):
        # 1. Enqueue data through the Oracle
        enriched_train = self._precompute_oracle_features(train_df.copy())
        enriched_val = self._precompute_oracle_features(val_df.copy())
        
        # 2. Initialize Environments
        train_env = Monitor(XAUDynamicEnv(df=enriched_train))
        val_env = Monitor(XAUDynamicEnv(df=enriched_val))
        
        policy_kwargs = dict(net_arch=[128, 128])
        
        # 3. Configure SAC Agent (With Static Entropy Firewall)
        model = SAC(
            "MlpPolicy", 
            train_env, 
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            buffer_size=50000,
            batch_size=256,
            ent_coef=0.05, 
            target_update_interval=2, 
            tensorboard_log=f"./logs/wfa_split_{split_idx}/"
        )
        
        os.makedirs(f"./models/manager/saved/wfa_{split_idx}/", exist_ok=True)
        
        eval_callback = EvalCallback(
            val_env, 
            best_model_save_path=f"./models/manager/saved/wfa_{split_idx}/",
            log_path=f"./logs/wfa_split_{split_idx}/",
            eval_freq=5000,
            deterministic=True, 
            render=False
        )
        
        print(f"--- Starting Manager Training for Split {split_idx} ---")
        model.learn(total_timesteps=50000, callback=eval_callback)
        return model