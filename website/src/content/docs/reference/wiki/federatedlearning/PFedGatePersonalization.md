---
title: "PFedGatePersonalization<T>"
description: "Implements pFedGate — gated layer-wise mixture of local and global parameters."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Personalization`

Implements pFedGate — gated layer-wise mixture of local and global parameters.

## For Beginners

pFedGate learns a small "gate" per layer that decides how much
to use the global model vs. the client's local model for that layer. Gates are numbers
between 0 and 1: gate=0 means "use fully global" and gate=1 means "use fully local."
Each client learns different gate values based on their data distribution. The gates
are lightweight (one scalar per layer) and personalized (not aggregated).

## How It Works

Per-layer mixing:

Reference: Chen, S., et al. (2023). "pFedGate: Data-Driven Expert Gating for
Personalized Federated Learning." NeurIPS 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PFedGatePersonalization(Double,Double)` | Creates a new pFedGate personalization strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GateInitValue` | Gets the initial gate value. |
| `GateLearningRate` | Gets the gate learning rate. |
| `Gates` | Gets the gate values for all layers. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGates(Dictionary<String,[]>,Dictionary<String,[]>)` | Applies gate mixing to produce effective parameters per layer. |
| `ComputeGateRegularizationLoss(Double)` | Computes the total gate regularization loss (L2 on gates to prevent extreme values). |
| `InitializeGates(Dictionary<String,[]>)` | Initializes gates for a model structure. |
| `UpdateGate(String,Double)` | Updates gate values based on validation performance. |
| `UpdateGatesFromLosses(Dictionary<String,Double>,Dictionary<String,Double>)` | Updates all gates simultaneously by comparing validation loss with global-only vs local-only parameters. |

