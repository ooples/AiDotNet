---
title: "FederatedContinualLearningOptions"
description: "Configuration options for federated continual learning (preventing catastrophic forgetting in FL)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for federated continual learning (preventing catastrophic forgetting in FL).

## For Beginners

When federated models learn new tasks over time, they can forget what they
learned before (catastrophic forgetting). These strategies identify which model parameters are important
for previous tasks and protect them during future training rounds. Each client computes importance locally,
and the server aggregates these importance estimates across all clients.

## Properties

| Property | Summary |
|:-----|:--------|
| `DataFreeFCLDistillationTemperature` | Gets or sets the distillation temperature for data-free continual learning. |
| `DataFreeFCLDistillationWeight` | Gets or sets the distillation loss weight for data-free FCL. |
| `ExperienceReplayBufferCapacity` | Gets or sets the maximum buffer capacity for experience replay per client. |
| `ExperienceReplayRatio` | Gets or sets the ratio of replay samples to new samples during training. |
| `FedAGCCorrectionStrength` | Gets or sets the correction strength for adaptive gradient correction. |
| `FedCILPrototypeDecay` | Gets or sets the EMA decay rate for prototype consolidation in FedCIL. |
| `FisherSamples` | Gets or sets the number of data samples for Fisher information estimation. |
| `ProjectionThreshold` | Gets or sets the projection threshold for orthogonal projection. |
| `RegularizationStrength` | Gets or sets the regularization strength for EWC penalty. |
| `Strategy` | Gets or sets the strategy. |

