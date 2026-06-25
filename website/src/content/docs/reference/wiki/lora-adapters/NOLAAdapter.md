---
title: "NOLAAdapter"
description: "Implements NOLA (Compressing LoRA using Linear Combination of Random Basis) adapter for extreme parameter efficiency."
section: "Reference"
---

_LoRA / PEFT Adapters_

Implements NOLA (Compressing LoRA using Linear Combination of Random Basis) adapter for extreme parameter efficiency.

## For Beginners

NOLA is an extreme compression technique for LoRA that makes fine-tuning
even more efficient. Instead of storing and training two low-rank matrices (A and B), NOLA:

- Generates random "template" matrices on-the-fly (same random numbers every time due to fixed seed)
- Only trains small coefficients that control how much of each template to use
- Achieves 2-3x fewer parameters than LoRA while maintaining performance

Think of it like this:

- Traditional LoRA: You have 100 adjustable knobs (parameters)
- NOLA: You have 5 master controls that blend pre-defined settings

Key innovations:

1. **Memory efficiency:** Random basis matrices are discarded after use and regenerated when needed
2. **Parameter efficiency:** Only coefficients are trained, not full matrices
3. **Performance:** Achieves similar or better results than LoRA with far fewer parameters

Example compression (1000x1000 layer, rank=8):

- LoRA: 16,000 parameters (1000×8 + 8×1000)
- NOLA with 100 basis: 200 parameters (100 coefficients for A + 100 for B) - 80x reduction!

On LLaMA-2 70B, NOLA achieves 20x compression over LoRA with no accuracy loss.

## How It Works

NOLA overcomes the rank-one lower bound in traditional LoRA by re-parameterizing the low-rank matrices
using linear combinations of randomly generated basis matrices. Instead of optimizing the full low-rank
matrices A and B, NOLA:

1. Generates fixed random basis matrices using a deterministic seed
2. Optimizes only scalar coefficients that linearly combine these basis matrices
3. Regenerates basis matrices during forward/backward passes to minimize memory usage

This decouples the number of trainable parameters from both the choice of rank and the network architecture,
achieving compression ratios of 20x over standard LoRA without accuracy degradation.

**Reference:** NOLA: Compressing LoRA using Linear Combination of Random Basis
(Koohpayegani et al., ICLR 2024) - https://arxiv.org/abs/2310.02556

