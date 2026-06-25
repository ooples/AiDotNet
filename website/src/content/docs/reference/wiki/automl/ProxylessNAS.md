---
title: "ProxylessNAS<T>"
description: "ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AutoML.NAS`

ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware.
Uses path binarization and latency-aware loss to search directly on the target device
without requiring a proxy task or separate hardware lookup tables.

Reference: "ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware" (ICLR 2019)

## For Beginners

ProxylessNAS searches for architectures directly on the
target task and hardware, eliminating the need for proxy tasks. Most NAS methods
search on small datasets then transfer, but ProxylessNAS searches on the full task.
Think of it as test-driving cars on the actual roads you will drive rather than a
parking lot. It uses path binarization to keep memory usage manageable.

## Methods

| Method | Summary |
|:-----|:--------|
| `BinarizePaths(Matrix<>)` | Applies path binarization for memory-efficient single-path sampling. |
| `ComputeExpectedLatency(Int32,Int32)` | Computes the expected latency cost of the architecture |
| `ComputeTotalLoss(,Int32,Int32)` | Computes the total loss including task loss and latency regularization |
| `DeriveArchitecture` | Derives the final discrete architecture by selecting operations with highest weights |
| `EstimateArchitectureCost(Int32,Int32)` | Estimates the final architecture's hardware cost |
| `GetArchitectureGradients` | Gets architecture gradients |
| `GetArchitectureParameters` | Gets architecture parameters for optimization |
| `SetBinarizationTemperature(Double)` | Sets the binarization temperature |

