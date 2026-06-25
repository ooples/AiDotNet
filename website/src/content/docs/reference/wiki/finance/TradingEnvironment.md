---
title: "TradingEnvironment<T>"
description: "Base environment for financial trading simulations."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Finance.Trading.Environments`

Base environment for financial trading simulations.

## For Beginners

This is the "market simulator" shared by all trading
environments. It feeds price data to agents, executes trades, and computes
rewards based on portfolio changes.

## How It Works

TradingEnvironment implements common portfolio bookkeeping for RL trading:
positions, cash balance, portfolio value, and windowed market observations.
Derived environments specialize how actions are interpreted.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TradingEnvironment(Tensor<>,Int32,,Double,Boolean,Boolean,Int32,Nullable<Int32>)` | Initializes a new trading environment. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionSpaceSize` |  |
| `IsContinuousActionSpace` |  |
| `ObservationSpaceDimension` |  |
| `Random` | Gets the environment random number generator. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAction(Vector<>,Vector<>)` | Applies the trading action to update positions and cash. |
| `BuildObservation(Int32)` | Builds an observation vector for a specific time step. |
| `Close` |  |
| `ComputeReward(,)` | Computes the reward from portfolio value changes. |
| `ExecuteTrade(Int32,,)` | Executes a trade for a specific asset. |
| `GetPricesAt(Int32)` | Gets current prices at the specified time step. |
| `Reset` |  |
| `Seed(Int32)` |  |
| `Step(Vector<>)` |  |
| `UpdatePortfolioValue(Vector<>)` | Updates the portfolio value based on current prices. |
| `ValidateMarketData` | Validates that market data is usable for trading. |

