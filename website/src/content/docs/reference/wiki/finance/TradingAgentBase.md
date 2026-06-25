---
title: "TradingAgentBase<T>"
description: "Base class for financial trading agents using reinforcement learning."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Finance.Trading.Agents`

Base class for financial trading agents using reinforcement learning.

## For Beginners

This is the foundation for AI trading agents:

**The Key Insight:**
Trading is a sequential decision-making problem perfectly suited for RL.
The agent observes market conditions, makes trading decisions, and learns from
the resulting profits or losses. Over many training episodes, it learns strategies
that maximize risk-adjusted returns.

**Architecture:**

- Inherits from ReinforcementLearningAgentBase (standard RL functionality)
- Implements ITradingAgent (financial-specific methods)
- Adds portfolio tracking (values, returns, drawdown)
- Adds financial metrics (Sharpe ratio, win rate)
- Integrates risk management (position limits, VaR constraints)

**How Trading Agents Learn:**

1. Observe market state (prices, volumes, indicators)
2. Take action (buy, sell, adjust positions)
3. Receive reward (returns, risk-adjusted returns)
4. Store experience in replay buffer
5. Sample experiences and update neural network
6. Repeat thousands of times until convergence

## How It Works

TradingAgentBase extends the standard RL agent infrastructure with financial-specific
functionality. It provides common implementation for portfolio tracking, risk management,
and financial metrics that all trading agents share.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TradingAgentBase(TradingAgentOptions<>)` | Initializes a new instance of the TradingAgentBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CumulativeReturn` |  |
| `InitialCapital` |  |
| `MaxDrawdown` |  |
| `OptionPricer` | Option-pricing strategy for options-aware agents (valuation / Greeks / implied vol). |
| `PortfolioValue` |  |
| `PositionSizer` | Position-sizing strategy used to translate a signal/edge into a capital fraction. |
| `SharpeRatio` |  |
| `TotalTrades` |  |
| `WinRate` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyRiskConstraints(Vector<>,)` | Applies risk constraints to a trading action. |
| `CalculateCumulativeReturn` | Calculates the cumulative return since trading began. |
| `CalculateMaxDrawdown` | Calculates the maximum drawdown from portfolio history. |
| `CalculateReward()` | Calculates reward with optional risk adjustment. |
| `CalculateSharpeRatio` | Calculates the Sharpe ratio from portfolio history. |
| `CalculateWinRate` | Calculates the win rate (percentage of profitable trades). |
| `CreateBaseOptions(TradingAgentOptions<>)` | Creates base RL options from trading options. |
| `EnsureDefaultLayers(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32)` | Ensures a trading network architecture has valid input/output sizes and default layers. |
| `ExecuteTradeWithRiskManagement(Vector<>,)` |  |
| `GetMetrics` |  |
| `GetTradingMetrics` |  |
| `ResetEpisode` |  |
| `ResetTrading()` |  |
| `SelectTradingAction(Vector<>,Boolean)` |  |
| `StoreTradingExperience(Vector<>,Vector<>,,Vector<>,Boolean,)` |  |
| `TrainOnExperiences` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `PortfolioHistory` | History of portfolio values over time. |
| `TradeHistory` | History of individual trade P&L values. |
| `TradingOptions` | Options specific to financial trading. |
| `_initialCapital` | Initial capital at the start of trading. |
| `_peakValue` | Tracks peak portfolio value for drawdown calculation. |
| `_portfolioValue` | Current portfolio value. |
| `_totalTrades` | Total number of trades executed. |
| `_winningTrades` | Count of profitable trades. |

