---
title: "EnvelopeType"
description: "Specifies whether to use an upper or lower envelope in signal processing and data analysis operations."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies whether to use an upper or lower envelope in signal processing and data analysis operations.

## For Beginners

An envelope in data analysis is like drawing a line that follows the peaks or valleys of your data.

Think of it like tracing the outline of a mountain range:

- The upper envelope follows the tops of the mountains (the highest points)
- The lower envelope follows the bottoms of the valleys (the lowest points)

Envelopes are useful for:

- Identifying trends in noisy data
- Finding the boundaries of oscillating signals
- Detecting peaks and valleys in time series data
- Creating confidence intervals around predictions

For example, if you have stock price data that goes up and down, the upper envelope would connect all the highest prices,
while the lower envelope would connect all the lowest prices.

## Fields

| Field | Summary |
|:-----|:--------|
| `Lower` | Represents the lower envelope that follows the minimum values or valleys in the data. |
| `Upper` | Represents the upper envelope that follows the maximum values or peaks in the data. |

