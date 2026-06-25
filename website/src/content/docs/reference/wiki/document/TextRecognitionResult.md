---
title: "TextRecognitionResult<T>"
description: "Represents the result of text recognition from a cropped image."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document`

Represents the result of text recognition from a cropped image.

## For Beginners

Text recognition reads the actual characters from an image
of text. This result contains the recognized text string along with confidence
scores for each character.

## Properties

| Property | Summary |
|:-----|:--------|
| `Alternatives` | Gets alternative recognition hypotheses with their scores. |
| `AttentionWeights` | Gets the attention weights for visualization (if available). |
| `CharacterProbabilities` | Gets the character-level probability distribution (shape: [seq_len, vocab_size]). |
| `Characters` | Gets the per-character confidence scores. |
| `Confidence` | Gets the overall confidence score (0-1). |
| `ConfidenceValue` | Gets the confidence as a double value. |
| `ProcessingTimeMs` | Gets the processing time in milliseconds. |
| `Text` | Gets the recognized text string. |

