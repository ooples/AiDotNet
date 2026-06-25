---
title: "AdvancedCompressionStrategy"
description: "Advanced gradient compression strategies for federated learning communication efficiency."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Advanced gradient compression strategies for federated learning communication efficiency.

## For Beginners

These strategies go beyond basic top-k or quantization to achieve
100-1000x compression while maintaining model quality. They are designed for production FL
deployments with bandwidth constraints (mobile, satellite, edge devices).

## Fields

| Field | Summary |
|:-----|:--------|
| `Adaptive` | Adaptive compression: dynamically adjusts compression ratio per client based on estimated bandwidth, gradient importance, and staleness. |
| `FedDT` | FedDT: Decision-tree-based compression for heterogeneous architectures. |
| `FedKD` | FedKD: Knowledge-distillation-based compression. |
| `FetchSGD` | FetchSGD: Count-sketch + top-k hybrid compression for massive models. |
| `GradientSketch` | Gradient sketching: Count Sketch-based compression that maps gradients into a compact sketch using hash functions. |
| `OneBitSGD` | 1-bit SGD: extreme compression where each gradient component is encoded as a single bit (sign only). |
| `PowerSGD` | PowerSGD: low-rank gradient approximation using SVD-like factorization. |
| `SignSGD` | SignSGD: transmit only the sign of each gradient element (1-bit per parameter). |

