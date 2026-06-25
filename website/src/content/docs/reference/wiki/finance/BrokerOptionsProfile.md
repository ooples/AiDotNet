---
title: "BrokerOptionsProfile"
description: "A broker's mapping from the stable `OptionStrategyClass` archetypes to its own approval `OptionsApprovalLevel` numbers, plus the level this account is authorized for."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Options`

A broker's mapping from the stable `OptionStrategyClass` archetypes to its own approval
`OptionsApprovalLevel` numbers, plus the level this account is authorized for. Brokers
differ, so this is configurable; `Default` is the common scheme (covered/secured = L1,
long premium = L2, defined-risk spreads = L3, naked equity = L4, naked index = L5).

## Properties

| Property | Summary |
|:-----|:--------|
| `Default` | A fully-authorized default-scheme profile (Level5) — convenient for tests/research. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Deny(OptionStrategy)` | Null if permitted, else a human-readable denial reason. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultMap` | The common five-tier scheme. |

