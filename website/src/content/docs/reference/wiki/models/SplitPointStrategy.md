---
title: "SplitPointStrategy"
description: "Specifies how to choose the split point in a split neural network for vertical FL."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies how to choose the split point in a split neural network for vertical FL.

## For Beginners

In vertical FL, the neural network is "split" into two parts:
a bottom model (runs locally at each party) and a top model (runs at the coordinator).
The split point determines where the network is divided. Choosing the right split point
affects both privacy (deeper splits leak less information) and efficiency (deeper splits
require more local computation but less communication).

## Fields

| Field | Summary |
|:-----|:--------|
| `AutoOptimal` | Automatically selects the split point that minimizes information leakage while maintaining model accuracy. |
| `BalancedCompute` | Selects the split point that balances computational load across parties. |
| `Manual` | The user specifies exactly which layer to split at. |

