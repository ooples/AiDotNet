---
title: "AVICIAlgorithm<T>"
description: "AVICI — Amortized Variational Inference for Causal Discovery."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.DeepLearning`

AVICI — Amortized Variational Inference for Causal Discovery.

## For Beginners

AVICI is like a "universal causal discovery engine" powered by a
transformer (similar to ChatGPT's architecture). It processes data statistics through
attention mechanisms to recognize causal patterns.

## How It Works

AVICI uses a transformer-inspired architecture to perform causal discovery. The model
processes data via self-attention over variable pairs to produce edge probabilities.
It computes scaled dot-product attention between variable-pair representations derived
from sufficient statistics, then refines edge logits through multiple attention layers.

**Algorithm:**

- Compute pairwise features from data: covariance, correlation, variances
- Project features to query/key/value via learned matrices
- Apply multi-head self-attention over variable pairs
- Output layer produces edge logits from attended representations
- Apply NOTEARS acyclicity constraint via augmented Lagrangian
- Threshold edge probabilities for final DAG

Reference: Lorch et al. (2023), "Amortized Inference for Causal Structure Learning", NeurIPS.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

