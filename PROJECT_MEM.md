# Project Memory & Architecture Log

## Architecture V1 (Legacy - Deprecated)
**The Steps We've Tried:**
* **Data:** Synthetic 4-year M1 data generation for XAUUSD.
* **Features:** 15m structural zones, 30m/4H multi-timeframe mapping, DXY correlation.
* **Oracle:** LSTM network trained with standard loss functions to predict directional probabilities. Achieved ~69% accuracy on validation.
* **Manager:** PPO and SAC agents acting on Oracle confidence.
* **Result:** Forward journal testing showed initial success (+27% peak equity) followed by severe out-of-sample degradation (29% drawdown). Identified as Market Regime Shift / Concept Drift due to static training sets.

## Architecture V2 (Current)
**Core Objectives:**
1.  **Data Integrity:** Transition to real XAUUSD/DXY data. Implement Walk-Forward Analysis (WFA) with strict data embargoing to prevent time-series leakage.
2.  **Oracle Upgrade:** Replace LSTM with Temporal Attention layers to mitigate recency bias. Implement Focal Loss to prioritize difficult inflection points over standard noise.
3.  **Manager Upgrade:** Expand SAC action space to 3D `[Direction/Size, TP_Mult, SL_Mult]` allowing dynamic stop-losses. Shift reward function from absolute PnL to Risk-Adjusted metrics (Sortino Ratio).