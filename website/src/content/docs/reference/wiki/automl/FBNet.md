---
title: "FBNet<T>"
description: "FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AutoML.NAS`

FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search.
Uses Gumbel-Softmax with hardware latency constraints to find efficient architectures
optimized for specific target devices.

Reference: "FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable NAS" (CVPR 2019)

## For Beginners

FBNet (Facebook Net) finds architectures that are both
accurate and fast on specific hardware. It uses differentiable search with real
hardware latency measurements as constraints. Think of it as an architect who designs
buildings that are both beautiful and structurally efficient for a specific site,
rather than designing in the abstract.

## Methods

| Method | Summary |
|:-----|:--------|
| `AnnealTemperature(Int32,Int32)` | Anneals the temperature during training |
| `ComputeExpectedLatency` | Computes the expected latency cost for the entire architecture |
| `ComputeTotalLoss()` | Computes the total loss with latency regularization Loss = Cross-Entropy + λ * log(Latency) Using log(latency) makes the loss more sensitive to changes when latency is small |
| `DeriveArchitecture` | Derives the discrete architecture by selecting the operation with highest probability |
| `GetArchitectureCost` | Gets the final architecture's hardware cost breakdown |
| `GetArchitectureGradients` | Gets architecture gradients |
| `GetArchitectureParameters` | Gets architecture parameters for optimization |
| `GetTemperature` | Gets current temperature |
| `GumbelSoftmax(Vector<>,Boolean)` | Applies Gumbel-Softmax to layer-wise architecture parameters |
| `MeetsConstraints` | Checks if the derived architecture meets hardware constraints |
| `SetConstraints(HardwareConstraints<>)` | Sets hardware constraints for the search |

