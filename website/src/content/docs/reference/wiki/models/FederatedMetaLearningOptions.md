---
title: "FederatedMetaLearningOptions"
description: "Configuration options for federated meta-learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for federated meta-learning.

## How It Works

**For Beginners:** Meta-learning in federated settings aims to learn a "good starting point" (initial model)
that can adapt quickly to each client's local data with a small amount of fine-tuning.

In this library, federated meta-learning is implemented as an alternative server update rule that uses
client adaptation results (post-local training) to update the global initialization.

## Properties

| Property | Summary |
|:-----|:--------|
| `Enabled` | Gets or sets whether federated meta-learning is enabled. |
| `InnerEpochs` | Gets or sets the number of local adaptation epochs used for the inner loop. |
| `MetaLearningRate` | Gets or sets the server meta learning rate applied to the average adaptation delta. |
| `Strategy` | Gets or sets the federated meta-learning strategy. |

