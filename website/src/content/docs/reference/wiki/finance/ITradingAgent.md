---
title: "ITradingAgent<T>"
description: "Interface for financial trading agents using reinforcement learning."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Finance.Interfaces`

Interface for financial trading agents using reinforcement learning.

## For Beginners

Trading agents learn to make buy/sell decisions:

**The Key Insight:**
Financial markets are complex environments where RL agents can learn optimal
trading strategies through trial and error. Trading agents observe market state
(prices, volumes, indicators) and learn to take actions (buy, sell, hold) that
maximize risk-adjusted returns over time.

**Key Components:**

- State: Market observations (prices, volumes, technical indicators)
- Actions: Trading decisions (buy, sell, hold, position sizes)
- Rewards: Returns, risk-adjusted metrics (Sharpe ratio)
- Environment: Market simulator or live trading interface

**Financial Metrics:**
Trading agents track specialized metrics not found in standard RL:

- Sharpe Ratio: Risk-adjusted return measure
- Maximum Drawdown: Largest peak-to-trough decline
- Win Rate: Percentage of profitable trades
- Cumulative Return: Total portfolio growth

## How It Works

ITradingAgent extends standard RL agent capabilities with financial-specific
functionality for trading, portfolio management, and risk assessment.

## Properties

| Property | Summary |
|:-----|:--------|
| `CumulativeReturn` | Gets the cumulative return since trading began. |
| `InitialCapital` | Gets the initial capital when trading started. |
| `MaxDrawdown` | Gets the maximum drawdown experienced during trading. |
| `PortfolioValue` | Gets the current portfolio value. |
| `SharpeRatio` | Gets the current Sharpe ratio of the trading strategy. |
| `TotalTrades` | Gets the total number of trades executed. |
| `WinRate` | Gets the win rate (percentage of profitable trades). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExecuteTradeWithRiskManagement(Vector<>,)` | Executes a trade with risk management constraints. |
| `GetModelMetadata` | Gets the agent's model metadata with financial context. |
| `GetTradingMetrics` | Gets comprehensive financial metrics for the trading strategy. |
| `ResetTrading()` | Resets the trading episode and portfolio state. |
| `SelectTradingAction(Vector<>,Boolean)` | Selects a trading action given the current market state. |
| `StoreTradingExperience(Vector<>,Vector<>,,Vector<>,Boolean,)` | Stores a trading experience for learning. |
| `TrainOnExperiences` | Performs one training step using stored experiences. |

