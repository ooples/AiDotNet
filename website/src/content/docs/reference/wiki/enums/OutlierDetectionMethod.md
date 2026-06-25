---
title: "OutlierDetectionMethod"
description: "Defines different methods for detecting outliers in datasets."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines different methods for detecting outliers in datasets.

## For Beginners

Outliers are data points that differ significantly from other observations in your dataset.
Think of them as unusual values that stand out from the pattern - like if most people in a classroom are 
between 5'0" and 6'0" tall, someone who is 7'5" would be an outlier. Detecting outliers is important because 
they can skew your analysis or cause your AI model to learn incorrect patterns. These methods help you 
identify which data points might be outliers so you can decide whether to investigate them further or 
remove them from your analysis.

## Fields

| Field | Summary |
|:-----|:--------|
| `Combined` | Uses both Z-Score and IQR methods together to identify outliers. |
| `IQR` | Detects outliers using the Interquartile Range method, which is resistant to extreme values. |
| `ZScore` | Detects outliers based on how many standard deviations a value is from the mean. |

