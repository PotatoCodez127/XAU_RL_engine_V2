# Project Memory: XAU RL Engine V2 (Continuous State Summary)

## 1. Architectural Architecture
* **Oracle Phase A:** PyTorch Temporal Attention Oracle predicting directional logits `[Hold, Long, Short]` from a 30-period rolling feature window.
* **Manager Phase B:** Stable-Baselines3 Soft Actor-Critic (SAC) continuous manager controlling action space mapping to `[Direction/Size, TP Multiplier, SL Multiplier]`.
* **Validation Pipeline:** Walk-Forward Analysis (WFA) split generator with a 50-bar embargo window to mitigate data leakage.

## 2. Current Project State & Milestone Achieved
* **Holdout Firewall:** Implemented a strict 20% holdout slice (`holdout_fraction=0.2`) on the master dataset (`labeled_features_15m.csv`). This isolates pure out-of-sample data for validation.
* **WFA Completion:** The "Master Brain" sequential training pipeline ran successfully through all windows inside the 80% boundary, finishing at `wfa_43` (`models/manager/saved/wfa_43/best_model.zip`).

## 3. Discovered Behavioral Bottleneck
* **EV Farming / Hyperactivity:** In an unconstrained backtest environment, the agent executes approximately 93 trades per day, trading on roughly 97% of all available 15-minute bars.
* **Backtest Performance Snapshot:**
  * Total Trades: 29,384
  * Winrate: 34.48%
  * Avg TP Multiplier: 4.97x
  * Avg SL Multiplier: 1.05x
  * Max Drawdown: 4.36%
* **Root Cause:** Zero entry friction (no spread/commission modeling) combined with a holding penalty incentivized the SAC network to exploit its structural risk-to-reward ratio through hyper-frequency order placement.

## 4. Next Phase Roadmap: Gating & Friction Implementation
To reduce execution frequency to a realistic retail range (1–2 trades/day) and improve edge selection, three structural layers are slated for integration:
1. **Transaction Friction:** Introduction of a dynamic or flat transaction fee per entry (simulating a 1-pip spread/commission) to eliminate micro-scalping profitability.
2. **Algorithmic Cooldown:** A mandatory execution lockout period (e.g., 24 steps / 6 hours) inside the environment tracking immediately following a trade closure.
3. **Deterministic Oracle Gating:** Restricting the SAC Manager from executing positions unless the underlying Phase A Attention network's categorical certainty exceeds a critical threshold (e.g., > 85%).