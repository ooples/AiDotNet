---
title: "SequentialFeatureSelectionDirection"
description: "Defines the direction of sequential feature selection."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the direction of sequential feature selection.

## For Beginners

Sequential feature selection can work in two directions:
starting with no features and adding them, or starting with all features and removing them.

## How It Works

Think of it like packing for a trip:

- Forward Selection: Start with an empty suitcase and add items one by one, choosing the most

important item each time until you have enough.

- Backward Elimination: Start with a full suitcase and remove items one by one, removing the

least important item each time until you reach your desired size.

## Fields

| Field | Summary |
|:-----|:--------|
| `Backward` | Backward elimination starts with all features and iteratively removes the feature whose removal least degrades performance. |
| `Forward` | Forward selection starts with zero features and incrementally adds the feature that most improves performance. |

