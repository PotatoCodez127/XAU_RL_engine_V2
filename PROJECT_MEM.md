# XAUUSD Hybrid RL Engine - Project Memory

## 1. Architecture Overview (V2)
* **Target Asset:** XAUUSD (Gold) paired with DXY (US Dollar Index) correlation.
* **Data Pipeline:** Raw MetaTrader 5 M1 data, structurally engineered into a 15-minute operational base enriched with 30m/4H higher-timeframe wick zones.
* **Engine Type:** Hybrid Pipeline.
    * **Phase A (The Oracle):** Supervised Temporal Attention Neural Network. Analyzes 30-period rolling windows of 15m structural features to output directional probabilities `[Hold, Long, Short]`.
    * **Phase B (The Manager):** Soft Actor-Critic (SAC) Reinforcement Learning agent. Ingests Oracle probabilities and current account metrics to output continuous physical actions `[Direction, Position Size %, TP Multiplier, SL Multiplier]`.

## 2. Validation & Environment Physics
* **Walk-Forward Analysis (WFA):** Dataset sliced into 56 rolling windows (10,000 train bars / 2,500 test bars) with a strict 50-bar embargo to prevent data leakage.
* **Environment Constraints (XAUDynamicEnv):** * `0.1` drawdown penalty multiplier to encourage calculated risk.
    * `-0.05` inactivity penalty for the `Hold` action to enforce a "ticking clock" opportunity cost.
    * **The Billion Dollar Ceiling:** Internal balance strictly clamped at $1e9, and observation `equity_ratio` clamped at 10.0 to prevent PyTorch `float32` NaN explosion during exponential compounding.
* **Evaluation (Backtester):** Uses a flat-risk normalization ($100 per trade) to accurately evaluate R:R and Winrate without exponential distortion.

## 3. Current Performance Benchmark (Split 55 Out-of-Sample)
* **Trade Frequency:** ~2-3 trades per week (146 trades over 1.3 years of OOS data).
* **Behavior:** Evolved into an Intraday-to-Short-Swing hybrid. Enters with tight day-trader Stop Losses (~0.50x ATR) and holds for massive swings (~5.00x ATR), exploiting a 10:1 Reward-to-Risk ratio.

## 4. Current Phase: The "Master Brain" Deployment
* **Goal:** Train a single, unified SAC model that retains memory of all market regimes from 2020 to 2026.
* **Execution Strategy:** Utilizing continuous WFA memory transfer. The orchestrator (`main.py`) passes the `best_model.zip` from the previous split into the next. 
* **Infrastructure Handling:** Training is distributed across multiple Colab instances using `START_SPLIT`, `END_SPLIT`, and `RESUME_SAC_PATH` to bypass 12-hour compute timeouts while preserving sequential memory.