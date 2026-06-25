---
title: "SequencePackingCollateFunction<T>"
description: "Packs multiple variable-length sequences into fixed-length blocks for efficient LLM training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Text`

Packs multiple variable-length sequences into fixed-length blocks for efficient LLM training.

## How It Works

Sequence packing concatenates multiple short sequences into a single block,
reducing padding waste. Each block has a corresponding attention mask that
prevents cross-sequence attention.

This is the technique used by GPT-3 and LLaMA training to maximize GPU utilization.
Instead of padding every sequence to max_length, shorter sequences are concatenated
until the block is full, with separator tokens between them.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SequencePackingCollateFunction(Int32,Int32)` | Initializes a new sequence packing collate function. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Collate(IReadOnlyList<Tensor<>>)` | Packs multiple token sequences into fixed-length blocks. |

