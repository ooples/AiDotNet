---
title: "OptionStrategyClass"
description: "Broker-INDEPENDENT risk archetype of an option strategy — what the position actually IS, regardless of how any particular broker numbers its approval tiers."
section: "API Reference"
---

`Enums` · `AiDotNet.Finance.Options`

Broker-INDEPENDENT risk archetype of an option strategy — what the position actually IS, regardless of
how any particular broker numbers its approval tiers. Brokers map these to their own level numbers
(and disagree with each other — e.g. cash-secured puts are L1 at some brokers, L2 at others), so the
platform classifies into these stable archetypes and lets a `BrokerOptionsProfile` map them
to broker levels.

## Fields

| Field | Summary |
|:-----|:--------|
| `CoveredOrSecured` | Covered by owned stock or fully cash-secured: covered call, protective put, cash-secured put. |
| `DefinedRiskSpread` | Defined-risk multi-leg spread: every short leg hedged by a long (verticals, condors, butterflies). |
| `LongPremium` | Long options only — risk capped at premium: long call/put, long straddle/strangle. |
| `NakedEquity` | Naked/uncovered short EQUITY options — large/undefined risk. |
| `NakedIndex` | Naked/uncovered short INDEX options — the highest-risk tier (cash-settled, gap risk). |
| `None` | Not an options strategy (stock only / empty). |

