---
title: "ImportanceThresholdStrategy"
description: "Defines strategies for setting the importance threshold in feature selection."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines strategies for setting the importance threshold in feature selection.

## For Beginners

When selecting features based on importance scores from a model,
you need to decide which features are "important enough" to keep. This enum provides
different strategies for making that decision.

## How It Works

Think of it like deciding which students make the honor roll:

- Mean: Keep students who score above the class average
- Median: Keep the top 50% of students

Note: You can also use a custom threshold by calling the SelectFromModel constructor
that accepts a specific threshold value instead of a strategy.

## Fields

| Field | Summary |
|:-----|:--------|
| `Mean` | Keep features with importance greater than or equal to the mean importance. |
| `Median` | Keep features with importance greater than or equal to the median importance. |

