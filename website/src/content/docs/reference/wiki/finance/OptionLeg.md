---
title: "OptionLeg"
description: "One option leg of a strategy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Options`

One option leg of a strategy. Quantity is in CONTRACTS (each = `ContractMultiplier` shares).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OptionLeg(OptionRight,OptionTradeAction,Double,Double,Int32)` | One option leg of a strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SignedContracts` | Signed share-equivalent exposure sign: long call / short put are bullish (+), etc. |

