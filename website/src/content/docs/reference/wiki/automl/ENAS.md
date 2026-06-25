---
title: "ENAS<T>"
description: "Efficient Neural Architecture Search via Parameter Sharing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AutoML.NAS`

Efficient Neural Architecture Search via Parameter Sharing.
ENAS uses a controller RNN to sample architectures and shares weights among child models,
achieving 1000x speedup over standard NAS.

Reference: "Efficient Neural Architecture Search via Parameter Sharing" (ICML 2018)

## For Beginners

ENAS makes architecture search 1000x faster by sharing
weights between candidate architectures. Instead of training each design from scratch,
all designs share the same trained parameters. A controller network learns to pick
good architectures, like a manager who assigns existing workers to different teams
rather than hiring new ones each time.

## Methods

| Method | Summary |
|:-----|:--------|
| `AddOperationToArchitecture(Architecture<>,Int32,Int32,Int32)` | Adds an operation to the architecture if valid |
| `ComputeEntropy(List<>)` | Computes entropy of a probability distribution |
| `ComputeProbabilities(Vector<>,Int32,Int32)` | Computes probability distribution over choices using controller |
| `GetBaseline` | Gets current baseline value |
| `GetControllerGradients` | Gets controller gradients |
| `GetControllerParameters` | Gets controller parameters for optimization |
| `GetSharedGradients` | Gets shared weight gradients |
| `GetSharedWeights` | Gets shared weights for all operations |
| `GetSharedWeights(String)` | Gets shared weights for a specific operation |
| `SampleAndUpdateState(Vector<>,Int32,Int32,,)` | Samples a choice, updates log probability, entropy, and hidden state |
| `SampleArchitecture` | Samples an architecture using the controller policy |
| `SampleFromDistribution(List<>)` | Samples from a probability distribution |
| `UpdateController(,,)` | Updates controller using REINFORCE policy gradient |
| `UpdateHiddenState(Vector<>,Int32)` | Updates controller hidden state (simplified LSTM cell) |

