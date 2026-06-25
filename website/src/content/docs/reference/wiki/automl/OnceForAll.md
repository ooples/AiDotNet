---
title: "OnceForAll<T>"
description: "Once-for-All (OFA) Networks: Train Once, Specialize for Anything."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AutoML.NAS`

Once-for-All (OFA) Networks: Train Once, Specialize for Anything.
Trains a single large network that supports diverse architectural configurations,
enabling instant specialization to different hardware platforms without retraining.

Reference: "Once-for-All: Train One Network and Specialize it for Efficient Deployment" (ICLR 2020)

## For Beginners

Once-for-All trains a single super-network that can be
instantly adapted to any device without retraining. After one training session, you
can extract optimized networks for phones, tablets, or servers. Think of it like a
universal clothing pattern that can be tailored to any size without starting over.
This saves enormous compute compared to training separate models per device.

## Methods

| Method | Summary |
|:-----|:--------|
| `ConfigToArchitecture(SubNetworkConfig)` | Converts a configuration to an architecture |
| `Crossover(SubNetworkConfig,SubNetworkConfig)` | Crossover operation for evolutionary search |
| `EvaluateConfig(SubNetworkConfig,HardwareConstraints<>,Int32,Int32)` | Evaluates a configuration based on accuracy and hardware constraints |
| `EvaluateSubNetworkOnValidation(SubNetworkConfig,Tensor<>,Tensor<>)` | Evaluates a sub-network configuration on validation data. |
| `GenerateRandomConfig` | Generates a random sub-network configuration |
| `GetSharedWeights(String,Int32,Int32)` | Gets shared weights for a specific layer configuration |
| `Mutate(SubNetworkConfig)` | Mutation operation for evolutionary search |
| `SampleSubNetwork` | Samples a sub-network configuration based on current training stage |
| `SearchArchitecture(Tensor<>,Tensor<>,Tensor<>,Tensor<>,TimeSpan,CancellationToken)` | Searches for the best sub-network architecture from the OFA supernet. |
| `SetTrainingStage(Int32)` | Progressive shrinking: trains the OFA network in stages Stage 1: Train largest kernel sizes Stage 2: Add elastic depth Stage 3: Add elastic expansion ratios Stage 4: Add elastic width |
| `SpecializeForHardware(HardwareConstraints<>,Int32,Int32,Int32,Int32)` | Specializes the OFA network to meet specific hardware constraints Uses evolutionary search to find the best sub-network configuration |

