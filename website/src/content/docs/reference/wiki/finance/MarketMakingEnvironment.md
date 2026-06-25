---
title: "MarketMakingEnvironment<T>"
description: "Market making environment that simulates bid/ask quoting and inventory risk."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Trading.Environments`

Market making environment that simulates bid/ask quoting and inventory risk.

## For Beginners

A market maker tries to profit from the bid/ask spread while
keeping inventory small. This environment teaches an agent to balance profit and risk.

## How It Works

MarketMakingEnvironment models a single asset where the agent sets bid/ask offsets.
Orders arrive stochastically based on the spread, and the agent earns the spread
but accumulates inventory risk.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MarketMakingEnvironment(Tensor<>,Int32,,,Double,Double,Int32,Double,Double,Boolean,Boolean,Int32,Nullable<Int32>)` | Creates a market making environment for a single asset. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionSpaceSize` |  |
| `IsContinuousActionSpace` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAction(Vector<>,Vector<>)` | Applies bid/ask offset actions and simulates order fills. |
| `CanChangeInventory()` | Checks if inventory can move without breaching limits. |
| `ComputeReward(,)` | Computes reward with inventory penalty. |

