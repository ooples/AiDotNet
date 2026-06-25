---
title: "EnsembleToxicityDetector<T>"
description: "Combines multiple toxicity detectors into a weighted ensemble for improved accuracy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Text`

Combines multiple toxicity detectors into a weighted ensemble for improved accuracy.

## For Beginners

Just like a panel of judges gives better verdicts than a single judge,
combining multiple toxicity detectors gives more accurate results. This module runs several
different detection approaches and combines their opinions.

## How It Works

Aggregates findings from multiple toxicity detection strategies (rule-based, embedding-based,
classifier-based) using configurable weights. The ensemble approach reduces both false positives
and false negatives compared to any single detector.

**References:**

- Ensemble methods for robust hate speech detection (ACL 2024)
- MetaTox knowledge graph for enhanced LLM toxicity detection (2024, arxiv:2412.15268)
- GPT-4o/LLaMA-3 zero-shot and few-shot hate speech detection (2025, arxiv:2506.12744)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EnsembleToxicityDetector(Double)` | Initializes a new ensemble toxicity detector with default sub-detectors. |
| `EnsembleToxicityDetector(ITextSafetyModule<>[],Double[],Double)` | Initializes a new ensemble toxicity detector with custom sub-detectors and weights. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateText(String)` |  |

