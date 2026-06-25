---
title: "BeveridgeNelsonAlgorithmType"
description: "Represents different algorithm types for Beveridge-Nelson decomposition of time series data."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums.AlgorithmTypes`

Represents different algorithm types for Beveridge-Nelson decomposition of time series data.

## For Beginners

Beveridge-Nelson decomposition is a technique used in economics and finance to separate 
time series data (like stock prices or GDP over time) into two main components:

1. Permanent Component (Trend) - The long-lasting changes that persist indefinitely
2. Temporary Component (Cycle) - The short-term fluctuations that eventually fade away

Unlike other decomposition methods that might look at regular patterns (like seasonality), 
Beveridge-Nelson focuses on distinguishing between changes that will have lasting effects 
versus those that will eventually reverse.

For example, when analyzing a company's stock price:

- A permanent component might be fundamental improvements in the company's business model
- A temporary component might be short-term market excitement that will eventually subside

This enum lists different algorithmic approaches to performing this type of decomposition.

## Fields

| Field | Summary |
|:-----|:--------|
| `ARIMA` | Uses ARIMA (AutoRegressive Integrated Moving Average) models to perform Beveridge-Nelson decomposition. |
| `Multivariate` | Extends the Beveridge-Nelson decomposition to handle multiple related time series simultaneously. |
| `Standard` | The standard implementation of the Beveridge-Nelson decomposition algorithm. |

