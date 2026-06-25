---
title: "RandomSwap<T>"
description: "Randomly swaps the positions of words in text."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Text`

Randomly swaps the positions of words in text.

## For Beginners

Random swap changes word order, like swapping
"the big dog" to "big the dog". This helps models become less sensitive
to exact word ordering.

## How It Works

**When to use:**

- Sentiment analysis (where word presence matters more than order)
- Topic classification
- Tasks where word order is somewhat flexible

**When NOT to use:**

- Tasks where word order is critical (machine translation, grammar checking)
- Named entity recognition

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RandomSwap(Int32,Double,String)` | Creates a new random swap augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumSwaps` | Gets the number of word swaps to perform. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(String[],AugmentationContext<>)` |  |
| `GetParameters` |  |

