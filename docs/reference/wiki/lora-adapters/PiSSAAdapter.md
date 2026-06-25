---
title: "PiSSAAdapter"
description: "Principal Singular Values and Singular Vectors Adaptation (PiSSA) adapter for parameter-efficient fine-tuning."
section: "Reference"
---

_LoRA / PEFT Adapters_

Principal Singular Values and Singular Vectors Adaptation (PiSSA) adapter for parameter-efficient fine-tuning.

## For Beginners

Think of PiSSA as "smart LoRA initialization". Standard LoRA starts from random: - Random A matrix (like throwing darts blindfolded) - Zero B matrix (starts with no effect) - Learns everything from scratch PiSSA starts from the most important parts of pretrained weights: - A and B capture the top-r "principal directions" of the pretrained model - Starts closer to the optimal solution - Like starting a puzzle with the border pieces already connected Example: If you have a pretrained language model with a 4096x4096 weight matrix, PiSSA with rank=8 will: 1. Find the top 8 most important patterns in those weights via SVD 2. Put those patterns into A and B (making them trainable) 3. Freeze the remaining "less important" patterns 4. Train only the top 8 patterns to adapt to your task This is much more efficient than starting from random and achieves better results!

## How It Works

PiSSA (NeurIPS 2024 Spotlight) improves upon standard LoRA by initializing adapter matrices with principal components from Singular Value Decomposition (SVD) of pretrained weights, rather than random initialization. This results in more effective use of the rank budget and faster convergence. 

**Key Differences from Standard LoRA:** - Standard LoRA: A initialized randomly, B initialized to zero - PiSSA: A and B initialized from top-r singular vectors of pretrained weights - Standard LoRA: All weights trainable - PiSSA: Residual weights frozen, only top-r components trainable 

**How PiSSA Works:** 1. Perform SVD on pretrained weights: W = U Σ V^T 2. Initialize adapter matrices from top-r components: - A = V_r (top-r right singular vectors, dimensions: inputSize × rank) - B = Σ_r * U_r^T (top-r left singular vectors scaled by singular values, dimensions: rank × outputSize) 3. Freeze residual matrix: W_residual = W - (A*B)^T 4. During training: output = W_residual * input + LoRA(input) 5. Only B and A are updated; W_residual stays frozen 

**Performance Benefits:** PiSSA achieves superior performance compared to standard LoRA: - GSM8K benchmark: 72.86% (PiSSA) vs 67.7% (LoRA) - Better initialization captures important pretrained knowledge - More effective gradient updates from the start - Faster convergence with fewer training steps 

**References:** - Paper: "PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models" - Venue: NeurIPS 2024 (Spotlight) - Key Insight: SVD-based initialization > random initialization for low-rank adaptation

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new PiSSAAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured PiSSAAdapter (rank {config.Rank}).");
```

