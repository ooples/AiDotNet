---
title: "LongLoRAAdapter"
description: "LongLoRA adapter that efficiently extends LoRA to handle longer context lengths using shifted sparse attention."
section: "Reference"
---

_LoRA / PEFT Adapters_

LongLoRA adapter that efficiently extends LoRA to handle longer context lengths using shifted sparse attention.

## For Beginners

LongLoRA makes it affordable to train models on longer sequences. The Problem: - Standard LoRA works great for adapting models, but extending context length is expensive - Full dense attention on long sequences requires O(n²) computation - Training on 32k tokens instead of 2k tokens would be 256x slower! LongLoRA's Solution: - Uses a clever "shifted sparse attention" trick during training - Divides the sequence into groups and shifts them to maintain information flow - Much cheaper to train: O(n * k) where k is group size (typically 2048) - At inference, uses full dense attention to maintain quality Key Parameters: - OriginalContextLength: The base model's context window (e.g., 2048) - ExtendedContextLength: The target longer context (e.g., 8192 or 32768) - UseShiftedAttention: Enable shifted sparse attention (training only) - AttentionShiftSize: How many positions to shift attention groups (usually half the group size) Example Use Case: You have a model trained on 2k token contexts but need to process 16k token documents. LongLoRA lets you extend the context efficiently: - Training: Use shifted sparse attention (much faster) - Inference: Use full dense attention (full quality) Comparison to Standard LoRA: - Standard LoRA: Efficient parameter adaptation, same context length - LongLoRA: Efficient parameter adaptation + context length extension - Adds minimal overhead (just the attention shift mechanism) Research Background: LongLoRA has been successfully used to extend: - LLaMA 2 7B from 4k to 32k context (8x extension) - LLaMA 2 13B from 4k to 64k context (16x extension) - With only ~10% of the training cost compared to full fine-tuning Reference: LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models (2023) https://arxiv.org/abs/2309.12307

## How It Works

LongLoRA (2023) addresses the challenge of adapting large language models to longer context windows in a parameter-efficient manner. While standard LoRA works well for same-length fine-tuning, extending context windows naively would require substantial computational resources. 

LongLoRA introduces two key innovations: 1. Shifted Sparse Attention (S²-Attn): During training only, uses shifted group attention patterns that are more efficient while maintaining effectiveness for long contexts 2. Dense Attention at Inference: At inference time, switches back to standard dense attention for full context utilization without the training overhead

