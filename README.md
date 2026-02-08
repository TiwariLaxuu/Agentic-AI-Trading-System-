# Agentic-AI-Trading-System-

## Goal
Build an agentic AI trading system that:

Uses only OHLCV

Detects market regimes

Selects or switches strategies

Uses RL for execution

Manages risk and capital

Learns continuously


## Agentic AI Trading System (OHLCV-Only)

```mermaid

flowchart TB
    center["LLM Agentic Core"]

    A["Market Data (OHLCV)"]
    B["Feature Engine"]
    C["Market Regime Detector"]
    D["Strategy Selector"]
    E["RL Execution Agent"]
    F["Risk Manager"]
    G["Portfolio Monitor"]
    H["Learning & Memory"]

    A --> center
    B --> center
    C --> center
    D --> center
    E --> center
    F --> center
    G --> center
    H --> center

    center --> D
    center --> E
    center --> F
    center --> H
