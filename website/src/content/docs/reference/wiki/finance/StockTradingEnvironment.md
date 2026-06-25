---
title: "StockTradingEnvironment<T>"
description: "Simple single-asset trading environment with buy/hold/sell actions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Trading.Environments`

Simple single-asset trading environment with buy/hold/sell actions.

## For Beginners

This environment is the "hello world" of trading RL:
the agent decides whether to buy one unit, sell one unit, or do nothing.

## How It Works

StockTradingEnvironment models a single asset with discrete actions:
hold, buy, or sell. It uses the base TradingEnvironment for portfolio
bookkeeping and reward calculation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StockTradingEnvironment(Tensor<>,Int32,,,Double,Boolean,Boolean,Int32,Nullable<Int32>)` | Creates a stock trading environment for a single asset. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionSpaceSize` |  |
| `IsContinuousActionSpace` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAction(Vector<>,Vector<>)` | Applies the discrete buy/hold/sell action. |
| `GetActionIndex(Vector<>)` | Converts a Vector action into a discrete action index. |

