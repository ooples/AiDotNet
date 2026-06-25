---
title: "FedCILContinualLearning<T>"
description: "Implements FedCIL — Federated Class-Incremental Learning with prototype consolidation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.ContinualLearning`

Implements FedCIL — Federated Class-Incremental Learning with prototype consolidation.

## For Beginners

In class-incremental learning, new classes appear over time
(e.g., a spam filter encountering new spam categories). FedCIL handles this in FL by:
(1) maintaining class prototypes (average feature vectors per class) that are shared instead
of raw data, (2) using these prototypes to generate synthetic features for old classes
during training, preventing forgetting. This is especially important when different clients
see different new classes at different times.

## How It Works

Algorithm:

Reference: Qi, D., et al. (2023). "Better Generative Replay for Continual Federated
Learning." CVPR 2023.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FedCILContinualLearning(Double,Int32)` | Creates a new FedCIL strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `KnownClasses` | Gets the known classes as a snapshot (not a live view). |
| `PrototypeDecay` | Gets the prototype decay rate. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateImportance(Dictionary<Int32,Vector<>>,Dictionary<Int32,Double>)` |  |
| `ComputeImportance(Vector<>,Matrix<>)` |  |
| `ComputeRegularizationPenalty(Vector<>,Vector<>,Vector<>,Double)` |  |
| `GenerateSyntheticFeatures(Int32,Int32,Double)` | Generates synthetic features for a class from its prototype (with noise). |
| `ProjectGradient(Vector<>,Vector<>)` |  |
| `UpdatePrototype(Int32,[])` | Updates the global prototype for a class. |

