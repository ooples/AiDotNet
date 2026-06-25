---
title: "AdaLoRAAdapter"
description: "Adaptive Low-Rank Adaptation (AdaLoRA) adapter that dynamically allocates parameter budgets among weight matrices."
section: "Reference"
---

_LoRA / PEFT Adapters_

Adaptive Low-Rank Adaptation (AdaLoRA) adapter that dynamically allocates parameter budgets among weight matrices.

## For Beginners

AdaLoRA is like smart LoRA that learns which parts of the adaptation matter most.

Think of standard LoRA as giving every layer the same budget (rank=8 everywhere).
AdaLoRA is smarter:

- Some layers get more budget (rank=16) because they're important for the task
- Other layers get less budget (rank=2) because small changes are enough
- The model learns this automatically during training

How it works:

1. Start with a large rank (e.g., maxRank=32)
2. During training, track how important each component is
3. Prune components with low importance scores
4. Focus parameters on what actually helps

Benefits:

- More parameter-efficient than fixed-rank LoRA
- Better performance with same parameter budget
- Automatically finds optimal rank per layer

Reference: "Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning" (ICLR 2023)
https://arxiv.org/abs/2303.10512

## How It Works

AdaLoRA improves upon standard LoRA by dynamically adjusting the rank allocation based on importance scores.
Instead of using a fixed rank for all weight matrices, AdaLoRA:

- Starts with a maximum rank and adaptively reduces it during training
- Computes importance scores for each singular value component
- Prunes less important components to focus parameter budget on critical adaptations
- Allows different layers to have different effective ranks

This leads to more efficient parameter usage compared to fixed-rank LoRA, especially for large models
where some layers need more adaptation capacity than others.

