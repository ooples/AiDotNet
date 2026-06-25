---
title: "HybridSchedulePattern"
description: "Defines the scheduling pattern for hybrid SSM/attention architectures."
section: "API Reference"
---

`Enums` · `AiDotNet.NeuralNetworks.Layers.SSM`

Defines the scheduling pattern for hybrid SSM/attention architectures.

## For Beginners

Hybrid architectures mix SSM blocks (like Mamba) with attention blocks.
The schedule pattern determines which layers are SSM and which are attention. Different patterns
have been shown to work well for different use cases.

## Fields

| Field | Summary |
|:-----|:--------|
| `Custom` | User provides an explicit sequence of layer types. |
| `JambaStyle` | Jamba-style: every Nth layer is attention, rest are SSM. |
| `SambaStyle` | Samba-style: alternating Mamba and sliding window attention blocks. |
| `ZambaStyle` | Zamba-style: shared attention layers interleaved with Mamba blocks. |

