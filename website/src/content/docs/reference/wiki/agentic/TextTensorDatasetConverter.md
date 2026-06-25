---
title: "TextTensorDatasetConverter"
description: "Tokenizes a text `FineTuningDataset` into tensor supervised-fine-tuning data so a tensor model (e.g."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Agentic.SelfImproving`

Tokenizes a text `FineTuningDataset` into tensor supervised-fine-tuning data so a tensor model
(e.g. `MambaLanguageModel`) can be fine-tuned directly. This is the "tokenizer in front of a tensor
model" step the string-only `FineTuningDataConverter` leaves to the model pipeline.

## For Beginners

Turns "good run" text into the number grids a neural language model learns from:
for every position it records "given the words so far, the next word should be this one."

## How It Works

Each example's "prompt + completion" text is tokenized to a fixed-length window. The input is the one-hot
encoding of tokens `[0..L-1]` and the target is the one-hot of the next tokens `[1..L]` — standard
next-token (causal LM) supervision. Both are shape `[1, L, vocab]`, so a language model's per-position
logits and the targets flatten to equal-length vectors for the cross-entropy loss. Each example's reward is
carried as its sample weight.

Unlike a `string`-typed fine-tune (where the SFT loss step cannot turn a string into a numeric vector),
tensor inputs/outputs convert cleanly, so this is the path that actually trains a tensor model end-to-end.

## Methods

| Method | Summary |
|:-----|:--------|
| `ToTensorData(FineTuningDataset,IGenerationTokenizer,Int32,Int32)` | Converts a reward-filtered dataset to tensor next-token supervised data. |

