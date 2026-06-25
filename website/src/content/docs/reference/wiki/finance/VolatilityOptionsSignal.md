---
title: "VolatilityOptionsSignal"
description: "Turns a realized-volatility FORECAST (the one signal that is actually predictable — see the platform's vol research) into an options stance and a concrete DEFINED-RISK structure, by comparing it to the option market's implied vol:  - foreca…"
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Finance.Options`

Turns a realized-volatility FORECAST (the one signal that is actually predictable — see the platform's
vol research) into an options stance and a concrete DEFINED-RISK structure, by comparing it to the
option market's implied vol:

- forecast realized vol ≪ implied → vol is overpriced → **sell vol** via an iron condor (collect

premium, defined risk) with short strikes ~1 implied-σ out.

- forecast realized vol ≫ implied → vol is cheap → **buy vol** via a long straddle at the money.

This is the monetization bridge for the vol edge. It needs an IMPLIED vol (from an option chain) as
input — wire an options-chain feed to supply it. The DECISION is pure + testable here.

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(Double,Double,Double,Double)` | Compute the vol edge + stance. |
| `Recommend(VolEdge,Double,Double,BrokerOptionsProfile,Double,Double)` | Recommend a concrete defined-risk structure for the edge, with strikes placed off the implied 1-σ move over the option's life (σ·spot·√T). |

