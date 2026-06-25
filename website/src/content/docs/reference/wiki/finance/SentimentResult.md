---
title: "SentimentResult<T>"
description: "Result of sentiment analysis on a single text."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.Interfaces`

Result of sentiment analysis on a single text.

## For Beginners

This class holds the sentiment prediction for a piece of text,
including the predicted class (positive/negative/neutral), the confidence score,
and the probability distribution over all classes.

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassProbabilities` | Gets or sets the probability distribution over all sentiment classes. |
| `Confidence` | Gets or sets the confidence score (probability of predicted class). |
| `OriginalText` | Gets or sets the original text that was analyzed. |
| `PredictedClass` | Gets or sets the predicted sentiment class. |

