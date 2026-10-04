AI Trading Engine - Market Understanding

## Getting started

    pip install -r requirements.txt
    streamlit run app.py

The app runs standalone out of the box on a built-in synthetic multi-session
OHLCV generator (regime-switching volatility, realistic overnight/weekend/
holiday gaps) — no external data feed or API key required. Swap in a real
feed any time from Settings → Data source (CSV upload), or point
`core/data.py`'s `load_csv` at a live pipeline; every feature, model, and
backtest below consumes whatever `core/data.py` returns, unchanged.

Suggested first run: open **Training**, click **Train Now** (defaults take
roughly 30–90s on CPU), then visit **AI / Model** and **Backtesting** to see
the model's output and a held-out-window simulation.

---

Build a functional Python UI with:
• Dashboard
• AI / Model
• Training
• Backtesting
• Performance
• Trading
• Settings
• Support & Resistance
• Bollinger Bands
• Volatility
• Trend Momentum
• Multi-Timeframe Gap Analysis
• Logs / System Status

1. Multi-Timeframe Market Analysis
The AI engine must analyze:
5m / 15m / 30m / 1H / 2H / 4H
Timeframes must be configurable from Settings.
2. Bollinger Bands
Calculate for each selected timeframe:
• Middle Band
• Upper Band
• Lower Band
• BB Position
• Distance to Upper Band
• Distance to Lower Band
• BB Width %
• BB Width change
• BB Slope
• Band expansion/contraction
• Bollinger breakout status
• Mean-reversion conditions
Configurable parameters:
• BB Period
• Standard deviation multiplier
• Timeframes
The AI must learn whether BB conditions historically favor continuation, reversal or mean
reversion.
3. Volatility
Keep volatility measurements focused on:
• ATR
• Rolling/Realized Volatility
• Volatility Percentile
• Volatility Expansion / Contraction
Identify:
• Low
• Normal
• High
• Extreme
Volatility is an AI feature, not a fixed trading signal.
4. Trend & Momentum
Analyze:
15m / 30m / 1H / 2H / 4H
Features:
• Trend direction
• Trend strength
• Recent returns
• Momentum
• Rate of change
• Higher-timeframe trend alignment
• Trend acceleration/deceleration
The AI must learn the significance of these features rather than use fixed rules.
5. Support & Resistance
The existing trading robot will provide the Support & Resistance levels.
The AI engine must accept these levels as external inputs.
Do not recreate the proprietary S/R calculation inside the AI engine.
Features:
• Support price
• Resistance price
• Distance from current price
• Strength
• Timeframe
• Age
• Touch count
• Breakout status
• Rejection status
6. Multi-Timeframe Gap Analysis
Analyze:
5m / 15m / 30m / 1H / 2H / 4H
plus:
Overnight / Out-of-Hours / Weekend / Holiday / Session-Open gaps
All timeframes and minimum gap thresholds must be configurable.
Calculate:
• Gap size %
• Gap direction
• Gap count
• Cumulative gap
• Weighted cumulative gap
• Directional consistency
• Gap clustering
• Gap age
• Gap persistence
• Maximum favorable movement
• Maximum adverse movement
• Gap closure %
• Time to closure
• Historical closure probability
The system must analyze multiple small gaps as a combined pattern, not only as
independent gaps.
7. Out-of-Hours Gap Analysis
Explicitly detect:
• Previous session close → next session open
• Overnight gaps
• Friday → Monday gaps
• Holiday gaps
• Session-boundary gaps
Separate these from normal intraday gaps.
For each out-of-hours gap track:
• Gap size
• Direction
• Previous close
• Opening price
• % closed
• Time to closure
• Maximum extension
• Same-session closure
• Sessions remaining open
• Historical closure probability
The AI must learn whether specific out-of-hours gap configurations tend to close, continue,
reverse or expand.
Do not assume that overnight gaps automatically close.
8. Gap + Bollinger Band Interaction
The AI must be able to combine gap features with:
• BB Position
• BB Width %
• BB Width change
• BB Slope
• BB expansion/contraction
• BB breakout
• Mean-reversion conditions
Examples include:
• Gap near lower BB
• Gap near upper BB
• Gap outside BB
• Gap + expanding BB Width
• Gap + contracting BB Width
• Gap after BB breakout
• Gap against higher-timeframe trend
• Gap with higher-timeframe trend
The AI must learn the historical outcome of these combinations.
9. Market Regime
Create market-regime features using:
• Trend
• Volatility
• Bollinger Band behavior
• Momentum
• Recent returns
• Gap behavior
• Time/session
The AI should learn how the same setup behaves under different market regimes.
10. Time & Session Features
Include:
• Time of day
• Day of week
• Market open
• Market close
• Opening period
• Closing period
• Overnight period
• Weekend
• Holiday
• Time since market open
• Time until market close
11. AI Feature Engine
Create one standardized feature dataset containing:
Bollinger Bands
Volatility
Trend / Momentum
Support & Resistance
Intraday Gaps
Out-of-Hours Gaps
Gap Closure
Market Regime
Time / Session
The same feature pipeline must be used for:
• Training
• Validation
• Backtesting
• Live prediction
Strictly prevent look-ahead bias and data leakage.
12. AI Models
Use:
XGBoost
For structured/tabular feature learning.
Temporal Transformer
For learning chronological relationships between historical bars and multi-timeframe
features.
Both models must use the same standardized feature-engineering pipeline.
13. AI Output
The AI should produce probabilities and expected outcomes:
• Continuation Probability
• Reversal Probability
• Gap Closure Probability
• Expected Return
• Confidence
The AI must not use hard-coded assumptions such as:
Multiple negative gaps = Buy
or:
Overnight gap = automatic gap-closure trade
The model must learn from historical data whether a configuration is associated with:
• Trend continuation
• Mean reversion
• Reversal
• Gap closure
• Further gap expansion
• Volatility expansion
• No meaningful movement
14. Architecture
Market Data
↓
Data Processing
↓
Multi-Timeframe Feature Engine
├── Bollinger Bands
├── Volatility
├── Trend / Momentum
├── External Support & Resistance
├── Intraday Gap Analysis
├── Out-of-Hours Gap Analysis
├── Gap Closure
├── Market Regime
└── Time / Session
↓
Unified Feature Dataset
↓
XGBoost + Temporal Transformer
↓
Probabilities / Expected Return
↓
Risk Management

 



