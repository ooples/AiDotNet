---
title: "ACLOptions<T, TInput, TOutput>"
description: "Configuration options for the ACL (Adaptive Continual Learning) algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for the ACL (Adaptive Continual Learning) algorithm.

## How It Works

ACL learns task-specific parameter importance masks that protect critical weights
from catastrophic forgetting. The importance is estimated from gradient magnitudes
accumulated via exponential moving average across adaptation steps.

## Properties

| Property | Summary |
|:-----|:--------|
| `ElasticRegWeight` | Regularization weight pulling adapted parameters toward the initial (pre-task) values for protected parameters. |
| `ImportanceDecay` | EMA decay rate for accumulating parameter importance across tasks. |
| `MaskSparsityPenalty` | L1 penalty coefficient on the parameter importance masks to encourage sparsity. |
| `ProtectionStrength` | How much to scale down the learning rate for parameters deemed important. |

