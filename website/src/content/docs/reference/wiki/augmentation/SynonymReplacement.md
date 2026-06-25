---
title: "SynonymReplacement<T>"
description: "Replaces random words with their synonyms."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Text`

Replaces random words with their synonyms.

## For Beginners

Synonym replacement swaps words with similar-meaning words.
For example, "happy" might become "joyful" or "glad". This helps models understand
that different words can have the same meaning.

## How It Works

**When to use:**

- Text classification tasks
- Sentiment analysis
- When training data has limited vocabulary variation

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SynonymReplacement(Double,Double,String)` | Creates a new synonym replacement augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CustomSynonyms` | Gets or sets the custom synonym dictionary. |
| `ReplacementFraction` | Gets the fraction of words to replace. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(String[],AugmentationContext<>)` |  |
| `GetParameters` |  |

