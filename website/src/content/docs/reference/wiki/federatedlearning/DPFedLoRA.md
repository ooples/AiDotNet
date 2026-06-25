---
title: "DPFedLoRA<T>"
description: "Implements DP-FedLoRA — Differentially Private Federated LoRA with per-layer noise calibration."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Adapters`

Implements DP-FedLoRA — Differentially Private Federated LoRA with per-layer noise calibration.

## For Beginners

Standard DP-SGD adds the same amount of noise to all parameters,
which wastes privacy budget because some layers are more sensitive than others. DP-FedLoRA
calibrates the DP noise per-layer: layers with higher sensitivity get more noise, while
less sensitive layers keep their updates cleaner. This gives better privacy-utility tradeoffs
specifically designed for LoRA adapter aggregation in federated settings.

## How It Works

Per-layer noise:

Reference: DP-FedLoRA: Differentially Private Federated LoRA Fine-Tuning (2024).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DPFedLoRA(Int32,Int32,Double,Double,Double,Int32,Int32,Int32,Int32)` | Creates a new DP-FedLoRA strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdapterParameterCount` |  |
| `ClipNorm` | Gets the per-sample clip norm. |
| `CompressionRatio` |  |
| `CumulativeRdpEpsilon` | Gets the current cumulative RDP epsilon (before conversion to (eps,delta)-DP). |
| `NoiseMultiplier` | Gets the DP noise multiplier. |
| `Rank` | Gets the LoRA rank. |
| `RoundsCompleted` | Gets the number of aggregation rounds completed so far. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AccumulatePrivacyCost(Int32)` | Accumulates the privacy cost of one round using Renyi Differential Privacy accounting. |
| `AggregateAdapters(Dictionary<Int32,Vector<>>,Dictionary<Int32,Double>)` |  |
| `ComputePrivacySpent(Double)` | Computes the cumulative (epsilon, delta)-DP guarantee after all rounds so far. |
| `EstimateMaxRounds(Double,Double,Int32)` | Estimates the maximum number of rounds that can be performed while staying within a target epsilon budget. |
| `ExtractAdapterParameters(Vector<>)` |  |
| `MergeAdapterParameters(Vector<>,Vector<>)` |  |

