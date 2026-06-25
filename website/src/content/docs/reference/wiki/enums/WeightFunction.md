---
title: "WeightFunction"
description: "Defines different weight functions used in robust statistical methods and machine learning algorithms."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines different weight functions used in robust statistical methods and machine learning algorithms.

## How It Works

**For Beginners:** Weight functions are special mathematical formulas that help AI models handle unusual 
or extreme data points (outliers).

In regular statistics, outliers can significantly throw off your results. For example, if you're 
calculating the average income in a neighborhood and one billionaire lives there, the average 
would be misleadingly high.

Weight functions solve this problem by automatically giving less importance (lower weight) to data 
points that seem unusual or extreme. This makes your AI models more robust and reliable when 
dealing with real-world data that might contain errors or unusual values.

Each weight function has different characteristics that make it suitable for different situations:

- Some are gentler with outliers (like Huber)
- Others are more aggressive in downweighting extreme values (like Bisquare)

Choosing the right weight function depends on how much you expect your data to contain outliers 
and how you want to handle them.

## Fields

| Field | Summary |
|:-----|:--------|
| `Andrews` | The Andrews weight function, which uses a sine wave to handle outliers. |
| `Bisquare` | The Bisquare (also known as Tukey's biweight) weight function, which completely downweights extreme outliers. |
| `Huber` | The Huber weight function, which provides a balance between efficiency and robustness. |

