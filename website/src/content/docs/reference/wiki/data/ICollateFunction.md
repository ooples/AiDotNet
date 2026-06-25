---
title: "ICollateFunction<TSample, TBatch>"
description: "Defines how individual samples are assembled into a batch."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Data.Collation`

Defines how individual samples are assembled into a batch.

## How It Works

Collate functions control how variable-length or heterogeneous samples are combined
into a single batch tensor. This is critical for NLP and sequence models where
inputs have different lengths.

## Methods

| Method | Summary |
|:-----|:--------|
| `Collate(IReadOnlyList<>)` | Assembles a collection of individual samples into a batch. |

