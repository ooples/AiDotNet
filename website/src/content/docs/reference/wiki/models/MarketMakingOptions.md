---
title: "MarketMakingOptions<T>"
description: "Configuration options for the MarketMakingAgent."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the MarketMakingAgent.

## For Beginners

Market making is a specific trading strategy where an
agent provides liquidity to the market by quoting both a buy and a sell price.
These options control how the agent manages its inventory and sets its spreads.

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseSpread` | The base spread around the mid-price. |
| `InventoryPenalty` | Penalty coefficient for holding inventory (encourages neutral position). |
| `MaxInventory` | The maximum inventory size the agent can hold. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the market making options. |

