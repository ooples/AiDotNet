---
title: "FoundationModelSize"
description: "Defines the size variants available for foundation models."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the size variants available for foundation models.

## For Beginners

Larger models generally produce better predictions but require
more memory and computation time. Choose a size based on your resources:

- **Tiny/Mini:** Fast experiments, limited hardware, edge deployment
- **Small/Base:** Good balance of quality and speed for most use cases
- **Large/XLarge:** Best accuracy when compute resources are available

## How It Works

Foundation models typically come in multiple sizes, trading off between accuracy and
computational cost. This enum replaces error-prone string-based size selection
(e.g., `"base"`, `"large"`) with compile-time type safety.

## Fields

| Field | Summary |
|:-----|:--------|
| `Base` | Base variant (~50-200M parameters). |
| `Large` | Large variant (~200-700M parameters). |
| `Mini` | Mini variant (~5-20M parameters). |
| `Small` | Small variant (~14-50M parameters). |
| `Tiny` | Tiny variant with minimal parameters (~1-5M). |
| `XLarge` | Extra-large variant (700M+ parameters). |

