Title: Implement Anti-Hyperactivity Mechanisms in 'XAU RL Engine V2' Gold Trading System

Context:
You are an expert quantitative developer specializing in Deep Reinforcement Learning, PyTorch, and Stable-Baselines3. You are being handed the codebase for 'XAU RL Engine V2', a hybrid algorithmic trading engine for Gold (XAUUSD). The system consists of a Supervised Temporal Attention Oracle (Phase A) which feeds directional probabilities into a Soft Actor-Critic (SAC) Manager Environment (Phase B).

The system has completed Walk-Forward Analysis (WFA) training through split 43, honoring a strict 20% data holdout firewall. However, out-of-sample backtests reveal extreme "Reward Hacking" / EV Farming: the agent executes ~93 trades per day, taking entries on nearly 97% of all available 15-minute candles. It exploits a high Take Profit multiplier (~5x) against a tight Stop Loss (~1x) to farm reward, which is completely un-tradable in production due to broker spread and execution slippage.

Your Mission:
You must modify the codebase to throttle trade frequency down to a realistic retail distribution (target: 1-2 trades per day max) and increase the baseline winrate by forcing the system to isolate high-probability setups. 

CRITICAL MANDATE: Before providing ANY code changes, you must first explain your structural plan and architectural rationale. Do not rewrite any files until your conceptual path is confirmed.

Files to Analyze:
1. `env/xau_dynamic_env.py` - Holds the custom Gym/Gymnasium environment physics, action-handling, and reward metrics.
2. `backtest.py` - Orchestrates out-of-sample simulations and logs trade performance metrics.
3. `main.py` - The master continuous training loop orchestrating the WFA pipeline.

Architectural Targets to Implement:

1. Friction Injection (Spread/Commission Model):
   Add entry friction inside `XAUDynamicEnv`. Every time a position is initialized (Long or Short), deduct a transaction penalty from the balance/reward space equivalent to a standard retail spread (e.g., $10 flat or a fraction of the ATR). This forces the SAC network to learn that entering trades has a non-zero physical cost.

2. Algorithmic Cooldown (Time Lockout):
   Introduce a tracking attribute (`self.cooldown_timer`) inside the environment's `reset()` and `step()` functions. Once a position is closed, the environment must enter a mandatory lockout period of 24 steps (6 hours on the 15m base). During this window, any directional attempt by the SAC agent is forced to resolve as a 'Hold', preventing consecutive bar spamming.

3. Confidence Gatekeeping (Oracle Filtering):
   Introduce a strict probability gate inside the execution phase. The environment must evaluate the precomputed Oracle vectors (`prob_long`, `prob_short`). If the Oracle's confidence score for the selected direction does not exceed an explicit certainty threshold (e.g., 0.85), the action must be overridden to a 'Hold' state before processing portfolio metrics.

## 5. Architectural Fixes Implemented (Anti-Hyperactivity)
To resolve EV farming and micro-scalping, three structural layers were integrated into `XAUDynamicEnv` and aligned in `backtest.py`:

1. **Transaction Friction:** A fixed $10 transaction penalty is applied to the PnL of every executed trade to simulate broker spread and eliminate high-frequency, low-margin arbitrage.

2. **Algorithmic Cooldown:** A 24-step (6-hour) mandatory lockout period (`self.cooldown_timer`) is triggered post-execution, physically capping maximum trading frequency to 1-2 times per day.

3. **Oracle Gatekeeping:** An explicit certainty threshold (`self.oracle_threshold = 0.85`) restricts the SAC network from entering positions unless the Phase A Attention network provides high-confidence directional vectors. Logging has been expanded to surface these probabilities natively in the out-of-sample WFA reports.