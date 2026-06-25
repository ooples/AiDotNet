---
title: "ITextRecognizer<T>"
description: "Interface for text recognition models that read text from cropped image regions."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Document.Interfaces`

Interface for text recognition models that read text from cropped image regions.

## For Beginners

Text recognition is the second step in reading text from images.
Given a small image containing only text (like a single word or line), the recognizer
outputs the actual characters. This is like reading what's written in a highlighted region.

Example usage:

## How It Works

Text recognition models convert cropped images of text into character sequences.
They work on pre-detected text regions (from a text detector).

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxSequenceLength` | Gets the maximum sequence length this recognizer can output. |
| `SupportedCharacters` | Gets the supported character set (alphabet) for this recognizer. |
| `SupportsAttentionVisualization` | Gets whether this recognizer supports attention visualization. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetAttentionWeights` | Gets the attention weights for visualization (if supported). |
| `GetCharacterProbabilities` | Gets the character-level probabilities for the last recognition. |
| `RecognizeText(Tensor<>)` | Recognizes text from a cropped image region. |
| `RecognizeTextBatch(IEnumerable<Tensor<>>)` | Recognizes text from multiple cropped image regions (batch processing). |

