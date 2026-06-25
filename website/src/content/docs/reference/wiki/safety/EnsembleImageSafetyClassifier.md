---
title: "EnsembleImageSafetyClassifier<T>"
description: "Combines multiple image safety classifiers into a weighted ensemble for robust detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Image`

Combines multiple image safety classifiers into a weighted ensemble for robust detection.

## For Beginners

Just like getting multiple doctors' opinions leads to a better
diagnosis, running multiple safety classifiers and combining their results gives more
reliable detection. If two out of three classifiers flag an image as unsafe, we can be
more confident in the detection.

## How It Works

Runs multiple image classifiers (CLIP, ViT, SceneGraph) and aggregates their findings
using weighted voting. When multiple classifiers agree on a finding, the confidence is
boosted. This provides defense-in-depth: each classifier catches different types of
unsafe content, and the ensemble's combined coverage is greater than any single classifier.

**References:**

- UnsafeBench: Ensemble approaches improve F1 (2024, arxiv:2405.03486)
- Multi-model safety evaluation for VLMs (2025, arxiv:2512.06589)
- Ensemble methods for robust content moderation (Survey, 2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EnsembleImageSafetyClassifier(Double)` | Initializes a new ensemble image safety classifier with default sub-classifiers. |
| `EnsembleImageSafetyClassifier(IImageSafetyModule<>[],Double[],Double)` | Initializes a new ensemble image safety classifier with custom sub-classifiers. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ModuleName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateImage(Tensor<>)` |  |

