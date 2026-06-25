---
title: "LoHaAdapter"
description: "LoHa (Low-Rank Hadamard Product Adaptation) adapter for parameter-efficient fine-tuning."
section: "Reference"
---

_LoRA / PEFT Adapters_

LoHa (Low-Rank Hadamard Product Adaptation) adapter for parameter-efficient fine-tuning.

## For Beginners

LoHa is a variant of LoRA that uses element-wise multiplication instead of matrix multiplication. Think of it this way: - Standard LoRA: Learns "row and column patterns" that combine via matrix multiply - LoHa: Learns "pixel-by-pixel patterns" that combine via element-wise multiply LoHa is especially good when: 1. You need to capture local, element-wise patterns (like in images) 2. The weight matrix has spatial structure (like convolutional filters) 3. You want each weight to be adjusted somewhat independently Trade-offs compared to LoRA: - More parameters: Both A and B must be full-sized (input×output) per rank dimension - Different expressiveness: Better for element-wise patterns, different from matrix patterns - Better for CNNs: The element-wise nature matches convolutional structure better Example: A 100×100 weight matrix with rank=8 - Standard LoRA: 8×100 + 100×8 = 1,600 parameters - LoHa: 2 × 8 × 100 × 100 = 160,000 parameters (each rank has 2 full-sized matrices) LoHa uses MORE parameters than LoRA but models element-wise weight interactions via Hadamard products.

## How It Works

LoHa uses element-wise Hadamard products (⊙) instead of matrix multiplication for adaptation. Instead of computing ΔW = B * A like standard LoRA, LoHa computes: ΔW = sum over rank of (A[i] ⊙ B[i]) This formulation can capture element-wise patterns that matrix multiplication may miss, making it particularly effective for: - Convolutional layers (local spatial patterns) - Element-wise transformations - Fine-grained weight adjustments 

**Mathematical Formulation:** Standard LoRA: ΔW = B * A where B is rank×output, A is input×rank LoHa: ΔW = Σ(A[i] ⊙ B[i]) where A[i] and B[i] are both input×output The Hadamard product (⊙) performs element-wise multiplication, allowing each element of the weight matrix to be adjusted independently across the rank dimensions.

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new LoHaAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured LoHaAdapter (rank {config.Rank}).");
```

