---
title: "RandomDeletion<T>"
description: "Randomly deletes words from text."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Text`

Randomly deletes words from text.

## For Beginners

Random deletion removes some words from text,
simulating how people often skip words when speaking quickly or how text
might have missing words in noisy transcriptions.

## How It Works

**When to use:**

- Text classification where exact wording isn't critical
- Training robust models for noisy/incomplete text
- Simulating transcription errors

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RandomDeletion(Double,Int32,Double,String)` | Creates a new random deletion augmentation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DeletionProbability` | Gets the probability of deleting each word. |
| `MinWordsToKeep` | Gets the minimum number of words to keep. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(String[],AugmentationContext<>)` |  |
| `GetParameters` |  |

