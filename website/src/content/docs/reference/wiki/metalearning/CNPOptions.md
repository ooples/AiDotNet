---
title: "CNPOptions<T, TInput, TOutput>"
description: "Configuration options for Conditional Neural Process (CNP) (Garnelo et al., ICML 2018)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Conditional Neural Process (CNP) (Garnelo et al., ICML 2018).

## For Beginners

CNP is the simplest neural process:

1. Look at each example individually and summarize it
2. Average all summaries into one global summary
3. Use the global summary to predict at new points

Fast but produces independent predictions (no coherent function samples).

## How It Works

CNP encodes each context pair (x,y) independently, aggregates via mean pooling,
and decodes to make predictions at target points. It provides a simple, fast approach
to function approximation from context sets.

## Properties

| Property | Summary |
|:-----|:--------|
| `MetaModel` | Gets or sets the encoder/decoder model. |
| `RepresentationDim` | Gets or sets the representation dimensionality. |

