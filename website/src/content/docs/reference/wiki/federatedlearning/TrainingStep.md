---
title: "TrainingStep"
description: "Represents a single training step in the integrity log."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Verification`

Represents a single training step in the integrity log.

## Properties

| Property | Summary |
|:-----|:--------|
| `Epoch` | Gets or sets the epoch number. |
| `Loss` | Gets or sets the loss after this step. |
| `ModelStateHash` | Gets or sets the hash of the model state at this step. |
| `StepHash` | Gets or sets the hash of this step (chained with previous). |

