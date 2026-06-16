# XAUUSD Hybrid RL Engine - Project Memory

## 1. Architecture Overview (V2)
* **Target Asset:** XAUUSD (Gold) with DXY (US Dollar Index) correlation.
* **Timeframe:** Raw MetaTrader 5 M1 data, structurally engineered into a 15-minute operational base with 30m/4H higher-timeframe wick zones.
* **Engine Type:** Hybrid Pipeline.
    * **Phase A (The Oracle):** A Supervised Temporal Attention Neural Network. It analyzes 30-period rolling windows of the 15m structural features to output probabilities for `[Hold, Long, Short]`.
    * **Phase B (The Manager):** A Soft Actor-Critic (SAC) Reinforcement Learning agent. It takes the Oracle's probabilities and current account metrics to output continuous physical actions `[Direction, Position Size %, TP Multiplier, SL Multiplier]`.

## 2. Validation Framework
* **Walk-Forward Analysis (WFA):** The pipeline slices the dataset into 56 rolling windows (10,000 train bars / 2,500 test bars) with a strict 50-bar embargo to prevent data leakage.
* **Stationarity:** All price data is converted to percentage returns or structural distances (`dist_res_zone_top`, etc.) to prevent the neural networks from curve-fitting to absolute prices.

## 3. Current Performance Benchmark (Split 55 - Strict Reward Function)
* **Winrate:** 75.00%
* **Reward-to-Risk (R:R):** 2.34
* **Max Drawdown:** 1.46%
* **Trade Frequency:** Extremely low (4 trades in out-of-sample). 
* **Behavioral Analysis:** The initial environment penalized drawdowns too heavily (`0.5` multiplier), creating an institutional-grade, hyper-conservative "sniper" that only traded when mathematical certainty was absolute.

## 4. Current Objective & Active Tuning
* **Goal:** Increase trade frequency to 2-3 trades per week while maintaining high R:R and low drawdown.
* **Modifications:**
    1.  Reduced the dynamic drawdown penalty in `xau_dynamic_env.py` from `0.5` to `0.1`.
    2.  Introduced a `-0.05` inactivity penalty for the `Hold` action to create a mathematical opportunity cost.
* **Testing Strategy:** Fast-forwarding the `main.py` orchestrator to `START_SPLIT = 53` to rapidly train and backtest the new reward mechanics on the final 3 WFA windows before committing to a full 56-split re-run.