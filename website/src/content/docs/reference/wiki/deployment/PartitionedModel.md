---
title: "PartitionedModel<T, TInput, TOutput>"
description: "Represents a model partitioned for cloud+edge deployment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Edge`

Represents a model partitioned for cloud+edge deployment.

## For Beginners

PartitionedModel provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `CloudModel` | Gets or sets the model part for cloud execution. |
| `EdgeModel` | Gets or sets the model part for edge execution. |
| `IntermediateShape` | Gets or sets the intermediate tensor shape between edge and cloud. |
| `OriginalModel` | Gets or sets the original model. |
| `PartitionStrategy` | Gets or sets the partition strategy used. |

