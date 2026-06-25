---
title: "FederatedDriftMethod"
description: "Specifies the drift detection method used in federated learning."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the drift detection method used in federated learning.

## For Beginners

These methods detect when client data distributions change over time.
Each has different strengths — some catch sudden changes quickly (PageHinkley, DDM), while
others are better at gradual drift (ADWIN). Model-based methods (GradientDivergence, WeightDivergence)
detect drift through changes in training behavior rather than raw statistics.

## Fields

| Field | Summary |
|:-----|:--------|
| `ADWIN` | ADWIN (Adaptive Windowing): maintains a variable-length window and detects drift when two sub-windows have significantly different means. |
| `DDM` | DDM (Drift Detection Method): monitors error rate and standard deviation. |
| `GradientDivergence` | Gradient divergence: detects drift by comparing gradient directions between rounds. |
| `PageHinkley` | Page-Hinkley test: sequential analysis for detecting change in the mean of a process. |
| `WeightDivergence` | Weight divergence: detects drift by measuring how much model weights change relative to historical patterns. |

