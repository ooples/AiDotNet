---
title: "BigNAS<T>"
description: "BigNAS: Scaling Up Neural Architecture Search with Big Single-Stage Models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AutoML.NAS`

BigNAS: Scaling Up Neural Architecture Search with Big Single-Stage Models.
Combines sandwich sampling with in-place knowledge distillation to train
very large super-networks that can adapt to various deployment scenarios.

Reference: "BigNAS: Scaling Up Neural Architecture Search with Big Single-Stage Models"

## For Beginners

BigNAS trains one giant "super-network" that contains
many smaller networks inside it. After training, you can extract a network of any
size for your deployment needs without retraining. Think of it like buying one
adjustable tool instead of many fixed-size tools - the super-network adapts to
fit phones, tablets, or servers.

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDistillationLoss(Vector<>,Vector<>,)` | Computes knowledge distillation loss between teacher and student networks |
| `EvaluateConfig(BigNASConfig,HardwareConstraints<>,Int32,Int32)` | Evaluates a configuration |
| `EvolutionarySearch(HardwareConstraints<>,Int32,Int32,Int32,Int32,DateTime,CancellationToken)` | Evolutionary search for finding optimal sub-network |
| `GenerateRandomConfig` | Generates a random sub-network configuration |
| `MultiObjectiveSearch(List<ValueTuple<String,HardwareConstraints<>>>,Int32,Int32,Int32,Int32)` | Searches for optimal sub-networks for multiple hardware constraints simultaneously |
| `SandwichSample` | Sandwich sampling: samples smallest, largest, and random sub-networks together This improves training efficiency and performance of all sub-networks |

