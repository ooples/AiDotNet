---
title: "CheckpointingExtensions"
description: "Provides extension methods for gradient checkpointing on computation nodes."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Autodiff`

Provides extension methods for gradient checkpointing on computation nodes.

## Methods

| Method | Summary |
|:-----|:--------|
| `WithCheckpoint(ComputationNode<>,Func<ComputationNode<>,ComputationNode<>>)` | Wraps a computation with gradient checkpointing. |
| `WithSequentialCheckpoint(ComputationNode<>,IReadOnlyList<Func<ComputationNode<>,ComputationNode<>>>,Int32)` | Applies a sequence of functions with gradient checkpointing. |

