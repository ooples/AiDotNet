---
title: "RuleBasedContentClassifier<T>"
description: "A rule-based content classifier that uses pattern matching for classification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AdversarialRobustness.Safety`

A rule-based content classifier that uses pattern matching for classification.

## For Beginners

This is a simple classifier that looks for specific words
and patterns in text. While it's less sophisticated than ML-based classifiers, it's
fast, interpretable, and doesn't require training data.

## How It Works

This classifier serves as a baseline or fallback when ML models are not available.
It uses configurable regex patterns to detect various categories of harmful content.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RuleBasedContentClassifier(Dictionary<String,List<String>>,Double)` | Initializes a new rule-based content classifier with custom patterns. |
| `RuleBasedContentClassifier(Double)` | Initializes a new rule-based content classifier with default patterns. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddPattern(String,String)` | Adds a detection pattern for a category. |
| `Classify(Vector<>)` |  |
| `ClassifyText(String)` |  |
| `ClearCategory(String)` | Removes all patterns for a category. |
| `Deserialize(Byte[])` |  |
| `IsReady` |  |
| `LoadModel(String)` |  |
| `SaveModel(String)` |  |
| `Serialize` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `RegexTimeout` | Timeout for regex operations to prevent ReDoS attacks. |
| `_categoryPatterns` | Pattern rules for each category. |
| `_isReady` | Whether the classifier is initialized and ready. |

