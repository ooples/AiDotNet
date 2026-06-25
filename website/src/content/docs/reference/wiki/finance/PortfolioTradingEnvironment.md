---
title: "PortfolioTradingEnvironment<T>"
description: "Multi-asset portfolio trading environment with continuous weight actions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Trading.Environments`

Multi-asset portfolio trading environment with continuous weight actions.

## For Beginners

Instead of buy/hold/sell, the agent says "I want
40% in asset A and 60% in asset B" every step. The environment rebalances
the portfolio to match those weights.

## How It Works

PortfolioTradingEnvironment interprets the action vector as target
portfolio weights for each asset. It rebalances positions accordingly.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PortfolioTradingEnvironment(Tensor<>,Int32,,Double,Boolean,Boolean,Int32,Nullable<Int32>)` | Creates a portfolio trading environment. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionSpaceSize` |  |
| `IsContinuousActionSpace` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAction(Vector<>,Vector<>)` | Applies target weight actions by rebalancing the portfolio. |
| `NormalizeWeights(Vector<>)` | Normalizes the action vector into valid weights. |
| `ScaleBuysToCash([],Vector<>)` | Scales buy orders if they exceed available cash. |

