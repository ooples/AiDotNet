---
title: "X11AlgorithmType"
description: "Represents different variants of the X-11 seasonal adjustment algorithm used in time series analysis."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums.AlgorithmTypes`

Represents different variants of the X-11 seasonal adjustment algorithm used in time series analysis.

## For Beginners

X-11 is a statistical method used to remove seasonal patterns from time series data.

Imagine you run an ice cream shop and want to understand your true business growth. Your sales naturally 
spike in summer and drop in winter due to seasonal effects. The X-11 algorithm helps separate:

1. The seasonal component (predictable patterns that repeat, like summer sales spikes)
2. The trend component (your long-term growth or decline)
3. The irregular component (random fluctuations that don't follow patterns)

By removing seasonal effects, you can see if your business is truly growing year-over-year, 
regardless of these predictable seasonal patterns.

X-11 was developed by the U.S. Census Bureau and is widely used by government agencies and 
businesses worldwide to produce "seasonally adjusted" economic indicators like unemployment rates, 
retail sales, and GDP figures that you might hear about in the news.

## Fields

| Field | Summary |
|:-----|:--------|
| `LogAdditiveAdjustment` | A variant of X-11 that applies additive adjustments after logarithmic transformation of the data. |
| `MultiplicativeAdjustment` | A variant of X-11 that uses multiplicative adjustments for seasonal patterns. |
| `Standard` | The standard implementation of the X-11 seasonal adjustment algorithm. |

