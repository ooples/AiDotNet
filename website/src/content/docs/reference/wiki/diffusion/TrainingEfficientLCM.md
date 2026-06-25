---
title: "TrainingEfficientLCM<T>"
description: "Training-Efficient Latent Consistency Model for resource-constrained LCM distillation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.FastGeneration`

Training-Efficient Latent Consistency Model for resource-constrained LCM distillation.

## For Beginners

Training LCM requires expensive GPU compute — typically 32 A100
hours. This variant uses LoRA (lightweight adapters) and other efficiency tricks to
reduce that to about 3 A100 hours, making it feasible for researchers and hobbyists
to create their own fast-generation models from any Stable Diffusion checkpoint.

## How It Works

Reduces the computational cost of LCM distillation by using LoRA-based fine-tuning
instead of full model training, memory-efficient gradient checkpointing, and a
simplified consistency loss. Achieves comparable quality to full LCM with 10x less
training compute.

Reference: Based on LCM-LoRA (Luo et al., 2023) with additional training optimizations

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TrainingEfficientLCM(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,Int32,Nullable<Int32>)` | Initializes a new Training-Efficient LCM. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `LoRArank` | Gets the LoRA rank used for efficient fine-tuning. |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `DeepCopy` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `SetParameters(Vector<>)` |  |

