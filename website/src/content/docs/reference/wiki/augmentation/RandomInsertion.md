---
title: "RandomInsertion<T>"
description: "Randomly inserts synonyms of existing words into the text."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Text`

Randomly inserts synonyms of existing words into the text.

## For Beginners

Random insertion adds new words to the text based on
synonyms of existing words. For example, "I love programming" might become
"I love really programming" if "really" is considered related to context.

## How It Works

**When to use:**

- Text classification tasks
- Training data expansion
- Making models robust to longer text variations

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RandomInsertion(Int32,Double,String)` | Creates a new random insertion augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CustomInsertionWords` | Gets or sets custom words for insertion. |
| `NumInsertions` | Gets the number of insertions to perform. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(String[],AugmentationContext<>)` |  |
| `GetParameters` |  |

