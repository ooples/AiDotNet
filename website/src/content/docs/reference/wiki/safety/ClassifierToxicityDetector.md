---
title: "ClassifierToxicityDetector<T>"
description: "Detects toxic text using a trained linear classifier over character n-gram features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Text`

Detects toxic text using a trained linear classifier over character n-gram features.

## For Beginners

This module works like a spam filter for toxic content. It converts
text into numerical features (based on character patterns), then uses learned weights to
score how likely the text is to contain each type of harmful content.

## How It Works

Implements a multi-label logistic regression classifier operating on TF-IDF weighted
character n-gram features. Each safety category has its own weight vector trained to
distinguish toxic from benign content. The classifier supports configurable per-category
thresholds for precision/recall tradeoff.

**References:**

- GPT-3.5/Llama 2 achieving 80-90% accuracy in hate speech identification (2024, arxiv:2403.08035)
- Multilingual hate speech detection via prompting (2025, arxiv:2505.06149)
- MetaTox knowledge graph for enhanced toxicity detection (2024, arxiv:2412.15268)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ClassifierToxicityDetector(Double,Int32)` | Initializes a new classifier-based toxicity detector. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildClassifiers` | Builds per-category classifiers with learned weight vectors. |
| `ComputeLogisticScore(Vector<>,Vector<>,)` | Computes logistic regression score: sigmoid(w · x + b). |
| `EvaluateText(String)` |  |
| `ExtractFeatures(String)` | Extracts TF-IDF weighted character n-gram hash features from text. |

