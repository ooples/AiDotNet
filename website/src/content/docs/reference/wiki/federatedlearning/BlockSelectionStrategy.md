---
title: "BlockSelectionStrategy"
description: "Strategy for selecting which block to synchronize each round."
section: "API Reference"
---

`Enums` · `AiDotNet.FederatedLearning.Decentralized`

Strategy for selecting which block to synchronize each round.

## Fields

| Field | Summary |
|:-----|:--------|
| `Cyclic` | Rotate through blocks in order (round-robin). |
| `ImportanceBased` | Select blocks based on gradient importance (highest-change block first). |
| `Random` | Select blocks randomly each round. |

