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

Market Data (OHLCV)
        
        ↓
        
Feature Engine
        
        ↓
        
Market Regime Detector
        
        ↓
        
Strategy Selector (Agentic Planner)
        
        ↓
        
RL Execution Agent
       
        ↓
        
Risk Manager & Portfolio Monitor
       
        ↓
        
Learning & Memory Module



flowchart TD
    A[Market Data (OHLCV)]
    B[Feature Engine]
    C[Market Regime Detector]
    D[Strategy Selector (Agentic Planner)]
    E[RL Execution Agent]
    F[Risk Manager & Portfolio Monitor]
    G[Learning & Memory Module]

    A --> B --> C --> D --> E --> F --> G
