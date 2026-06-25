---
title: "LoRAPlusAdapter"
description: "LoRA+ adapter that uses optimized learning rates for faster convergence and better performance."
section: "Reference"
---

_LoRA / PEFT Adapters_

LoRA+ adapter that uses optimized learning rates for faster convergence and better performance.

## For Beginners

LoRA+ is an enhanced version of LoRA that trains faster and better. In standard LoRA: - Both matrix A and B are updated with the same learning rate - Matrix B starts at zero, so it needs time to "catch up" - Matrix A starts random, so it's already contributing from the start LoRA+ recognizes this asymmetry: - Matrix A is updated with a base learning rate (e.g., 0.0001) - Matrix B is updated with a higher learning rate (e.g., 0.0016 = 16x higher) - This accelerates learning without instability Key parameters: - BaseLearningRate: Learning rate for matrix A (the "slow" matrix) - LearningRateRatio: Multiplier for matrix B (typically 16.0) - ScaledLearningRate: Computed as BaseLearningRate * LearningRateRatio Research shows LoRA+ typically achieves: - 2x faster convergence - Better final performance - No additional parameters compared to standard LoRA Example: If base learning rate is 0.0001 and ratio is 16.0: - Matrix A updates with learning rate 0.0001 - Matrix B updates with learning rate 0.0016 Reference: LoRA+: Efficient Low Rank Adaptation of Large Models (February 2024)

## How It Works

LoRA+ (February 2024) improves upon standard LoRA by using different learning rates for the A and B matrices. The key insight is that matrix B (which starts at zero) needs faster updates than matrix A (which starts random). This simple modification leads to significantly faster convergence and improved final performance.

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new LoRAPlusAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured LoRAPlusAdapter (rank {config.Rank}).");
```

