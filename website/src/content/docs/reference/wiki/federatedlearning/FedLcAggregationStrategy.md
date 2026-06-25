---
title: "FedLcAggregationStrategy<T>"
description: "Implements FedLC (Logit Calibration) aggregation strategy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Aggregators`

Implements FedLC (Logit Calibration) aggregation strategy.

## For Beginners

When different clients have different class distributions
(e.g., Hospital A sees mostly flu, Hospital B sees mostly cold), local models develop
biased predictions. FedLC fixes this by adjusting each client's logits based on its
local class frequency before aggregation.

## How It Works

During local training, logits are calibrated:

where p_local[c] is the local class frequency for class c. This counteracts the bias
introduced by the imbalanced local data distribution.

Reference: Zhang, J., et al. (2022). "Federated Learning with Label Distribution Skew
via Logits Calibration." ICML 2022.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FedLcAggregationStrategy(Double)` | Initializes a new instance of the `FedLcAggregationStrategy` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CalibrationTemperature` | Gets the logit calibration temperature (tau). |
| `RegisteredClientCount` | Gets the number of clients with registered distributions. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Aggregate(Dictionary<Int32,Dictionary<String,[]>>,Dictionary<Int32,Double>)` |  |
| `CalibrateBatch(Matrix<>,Int32)` | Calibrates a batch of logits for a client. |
| `CalibrateLogits(Vector<>,Int32)` | Calibrates raw logits using the client's local class distribution. |
| `ClearAllDistributions` | Clears all registered class distributions. |
| `ComputeClassDistribution(Int32[],Int32)` | Computes class distribution from sample labels (convenience helper). |
| `GetStrategyName` |  |
| `RegisterClassDistribution(Int32,Vector<>)` | Registers a client's local class distribution for use in logit calibration. |
| `UnregisterClassDistribution(Int32)` | Removes a client's registered class distribution. |

