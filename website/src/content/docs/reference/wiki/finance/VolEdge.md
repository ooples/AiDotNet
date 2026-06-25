---
title: "VolEdge"
description: "The volatility edge: our FORECAST realized vol vs the option market's IMPLIED vol, and the resulting stance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Options`

The volatility edge: our FORECAST realized vol vs the option market's IMPLIED vol, and the resulting
stance. Edge = (implied − forecast) / forecast: when implied richly exceeds our forecast, vol is
overpriced (sell it); when our forecast exceeds implied, vol is cheap (buy it).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VolEdge(Double,Double,Double,VolStance)` | The volatility edge: our FORECAST realized vol vs the option market's IMPLIED vol, and the resulting stance. |

