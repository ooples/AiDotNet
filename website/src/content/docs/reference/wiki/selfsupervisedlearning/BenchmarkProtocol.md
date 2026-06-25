---
title: "BenchmarkProtocol"
description: "Supported evaluation protocols for transfer learning benchmarks."
section: "API Reference"
---

`Enums` · `AiDotNet.SelfSupervisedLearning.Evaluation`

Supported evaluation protocols for transfer learning benchmarks.

## Fields

| Field | Summary |
|:-----|:--------|
| `FewShot` | Few-shot learning with limited labeled data. |
| `FineTuning` | Fine-tune entire network with lower learning rate. |
| `LinearProbing` | Freeze encoder, train linear classifier only. |

