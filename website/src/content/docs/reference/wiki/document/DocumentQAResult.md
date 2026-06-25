---
title: "DocumentQAResult<T>"
description: "Represents the result of document question answering."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document`

Represents the result of document question answering.

## For Beginners

When you ask a question about a document, the model returns
an answer along with confidence information and optionally the evidence (where in
the document the answer came from).

## Properties

| Property | Summary |
|:-----|:--------|
| `AlternativeAnswers` | Gets alternative answers if multiple were considered. |
| `Answer` | Gets the answer to the question. |
| `Confidence` | Gets the confidence score for the answer (0-1). |
| `ConfidenceLevel` | Gets the classification of confidence level. |
| `ConfidenceValue` | Gets the confidence as a double value for comparison. |
| `Evidence` | Gets the evidence regions that support the answer. |
| `HasAnswer` | Gets whether the model was able to find an answer. |
| `NoAnswerSentinel` | Gets the sentinel string used when a model has no answer. |
| `ProcessingTimeMs` | Gets processing time in milliseconds. |
| `Question` | Gets the original question that was asked. |

