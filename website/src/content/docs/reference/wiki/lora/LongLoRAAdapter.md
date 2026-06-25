---
title: "LongLoRAAdapter<T>"
description: "LongLoRA adapter that efficiently extends LoRA to handle longer context lengths using shifted sparse attention."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LoRA.Adapters`

LongLoRA adapter that efficiently extends LoRA to handle longer context lengths using shifted sparse attention.

## For Beginners

LongLoRA makes it affordable to train models on longer sequences.

The Problem:

- Standard LoRA works great for adapting models, but extending context length is expensive
- Full dense attention on long sequences requires O(n²) computation
- Training on 32k tokens instead of 2k tokens would be 256x slower!

LongLoRA's Solution:

- Uses a clever "shifted sparse attention" trick during training
- Divides the sequence into groups and shifts them to maintain information flow
- Much cheaper to train: O(n * k) where k is group size (typically 2048)
- At inference, uses full dense attention to maintain quality

Key Parameters:

- OriginalContextLength: The base model's context window (e.g., 2048)
- ExtendedContextLength: The target longer context (e.g., 8192 or 32768)
- UseShiftedAttention: Enable shifted sparse attention (training only)
- AttentionShiftSize: How many positions to shift attention groups (usually half the group size)

Example Use Case:
You have a model trained on 2k token contexts but need to process 16k token documents.
LongLoRA lets you extend the context efficiently:

- Training: Use shifted sparse attention (much faster)
- Inference: Use full dense attention (full quality)

Comparison to Standard LoRA:

- Standard LoRA: Efficient parameter adaptation, same context length
- LongLoRA: Efficient parameter adaptation + context length extension
- Adds minimal overhead (just the attention shift mechanism)

Research Background:
LongLoRA has been successfully used to extend:

- LLaMA 2 7B from 4k to 32k context (8x extension)
- LLaMA 2 13B from 4k to 64k context (16x extension)
- With only ~10% of the training cost compared to full fine-tuning

Reference: LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models (2023)
https://arxiv.org/abs/2309.12307

## How It Works

LongLoRA (2023) addresses the challenge of adapting large language models to longer context windows
in a parameter-efficient manner. While standard LoRA works well for same-length fine-tuning,
extending context windows naively would require substantial computational resources.

LongLoRA introduces two key innovations:

1. Shifted Sparse Attention (S²-Attn): During training only, uses shifted group attention patterns

that are more efficient while maintaining effectiveness for long contexts

2. Dense Attention at Inference: At inference time, switches back to standard dense attention

for full context utilization without the training overhead

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LongLoRAAdapter(ILayer<>,Int32,Int32,Int32,Double,Int32,Boolean)` | Initializes a new LongLoRA adapter for efficient context length extension. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AttentionShiftSize` | Gets the attention shift size used in shifted sparse attention. |
| `ExtendedContextLength` | Gets the extended context length this adapter targets. |
| `IsTraining` | Gets or sets whether the adapter is in training mode. |
| `OriginalContextLength` | Gets the original context length of the base model. |
| `UseShiftedAttention` | Gets or sets whether to use shifted sparse attention during forward/backward passes. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyShiftedAttention(Tensor<>)` | Applies shifted sparse attention pattern to the input tensor. |
| `Forward(Tensor<>)` | Performs the forward pass with optional shifted sparse attention. |
| `MergeToOriginalLayer` | Merges the LongLoRA adaptation into the base layer and returns the merged layer. |
| `ResetState` | Resets the internal state of the adapter. |
| `ReverseShiftedAttention(Tensor<>)` | Reverses the shifted sparse attention pattern to restore original positions. |
| `ShiftGroup(Tensor<>,Int32,Int32,Int32)` | Shifts elements within a group by the specified amount. |
| `UpdateParameterGradientsFromLayers` | Updates the parameter gradients vector from the layer gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_attentionShiftSize` | The shift size for shifted sparse attention (typically half the group size). |
| `_extendedContextLength` | The extended context length that this adapter targets. |
| `_isTraining` | Whether the model is currently in training mode. |
| `_originalContextLength` | The original context length that the base model was trained on. |
| `_useShiftedAttention` | Whether to use shifted sparse attention during training (disabled at inference). |

