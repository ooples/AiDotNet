---
title: "DiversityExampleSelector<T>"
description: "Selects diverse examples to maximize coverage of different patterns."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.FewShot`

Selects diverse examples to maximize coverage of different patterns.

## For Beginners

Picks examples that are different from each other.

Instead of picking similar examples, this ensures variety:

- If you have 100 sentiment examples
- Random might give 3 positive examples
- Diversity gives 1 positive, 1 negative, 1 neutral

Example:

Use this when:

- You want broad coverage of the example space
- Your examples cluster into natural groups
- Model needs to understand the full range of possibilities

## How It Works

This selector picks examples that are different from each other to provide broad coverage.
It uses a greedy algorithm to iteratively select examples that are most different from
already-selected examples.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiversityExampleSelector(Func<String,Vector<>>)` | Initializes a new instance of the DiversityExampleSelector class. |
| `DiversityExampleSelector(Func<String,Vector<>>,)` | Initializes a new instance of the DiversityExampleSelector class. |
| `DiversityExampleSelector(IEmbeddingModel<>)` | Initializes a new instance of the DiversityExampleSelector class. |
| `DiversityExampleSelector(IEmbeddingModel<>,)` | Initializes a new instance of the DiversityExampleSelector class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DiversityThreshold` | Gets the diversity threshold used for selection. |

## Methods

| Method | Summary |
|:-----|:--------|
| `OnExampleAdded(FewShotExample)` | Called when an example is added. |
| `OnExampleRemoved(FewShotExample)` | Called when an example is removed. |
| `SelectExamplesCore(String,Int32)` | Selects diverse examples using a greedy algorithm. |

