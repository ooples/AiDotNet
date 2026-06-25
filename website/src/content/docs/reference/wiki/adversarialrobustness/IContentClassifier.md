---
title: "IContentClassifier<T>"
description: "Defines the interface for ML-based content classification."
section: "API Reference"
---

`Interfaces` · `AiDotNet.AdversarialRobustness.Safety`

Defines the interface for ML-based content classification.

## For Beginners

Think of this as an AI-powered content moderator that
can understand the meaning of text, not just look for specific keywords. It can detect
subtle forms of harmful content that simple pattern matching would miss.

## How It Works

Content classifiers provide machine learning-based analysis of content to detect
harmful, toxic, or inappropriate material. Unlike regex-based pattern matching,
ML classifiers can understand semantic meaning and context.

## Methods

| Method | Summary |
|:-----|:--------|
| `Classify(Vector<>)` | Classifies content and returns the classification result. |
| `ClassifyBatch(Matrix<>)` | Classifies a batch of content items. |
| `ClassifyText(String)` | Classifies content provided as text. |
| `GetSupportedCategories` | Gets the list of content categories this classifier can detect. |
| `IsReady` | Checks if the classifier is ready to make predictions. |

