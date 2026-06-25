---
title: "ReLoRAAdapter"
description: "Restart LoRA (ReLoRA) adapter that periodically merges and restarts LoRA training for continual learning."
section: "Reference"
---

_LoRA / PEFT Adapters_

Restart LoRA (ReLoRA) adapter that periodically merges and restarts LoRA training for continual learning.

## For Beginners

ReLoRA is like having multiple rounds of LoRA training. Imagine you're fine-tuning a model on data that keeps changing: - Round 1: Train LoRA on dataset A for 1000 steps - Merge: Add the learned changes into the base model - Restart: Reset LoRA matrices and train on dataset B for 1000 steps - Merge: Add these new changes to the (already updated) base model - Repeat... Benefits: - Continual learning: Can keep learning from new data indefinitely - No catastrophic forgetting: Old knowledge is preserved in the base layer - Parameter efficient: LoRA matrices stay small even after many restarts - Flexible: Can adapt to distribution shifts and new tasks How it works: 1. Train normally with LoRA for N steps (restart interval) 2. At step N: Merge LoRA weights → AccumulatedWeight += LoRA 3. Reset LoRA matrices to zero (fresh start) 4. Continue training for another N steps 5. Repeat indefinitely Use cases: - Training on streaming data (news articles, user behavior, etc.) - Adapting to distribution shifts over time - Long-running training sessions that need checkpoints - Multi-task learning with periodic task switches Reference: "ReLoRA: High-Rank Training Through Low-Rank Updates" (2023) https://arxiv.org/abs/2307.05695

## How It Works

ReLoRA addresses the challenge of continual learning and long-running training by periodically: 1. Merging the LoRA weights into the base layer (accumulating the adaptation) 2. Resetting the LoRA matrices to restart training fresh 3. Continuing training with a clean slate while preserving previous learning 

This approach: - Prevents catastrophic forgetting by accumulating adaptations into the base layer - Allows continuous adaptation to new data without losing old knowledge - Maintains parameter efficiency by resetting LoRA to small matrices - Enables training on continuously evolving data streams

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new ReLoRAAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured ReLoRAAdapter (rank {config.Rank}).");
```

