---
title: "ILoRAAdapter<T>"
description: "Interface for LoRA (Low-Rank Adaptation) adapters that wrap existing layers with parameter-efficient adaptations."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for LoRA (Low-Rank Adaptation) adapters that wrap existing layers with parameter-efficient adaptations.

## For Beginners

A LoRA adapter wraps an existing layer (like a dense or convolutional layer)
and adds a small "correction layer" that learns what adjustments are needed. This is much more
memory-efficient than retraining all the weights in a large model.

Think of it like:

- The base layer has the original knowledge (frozen or trainable)
- The LoRA layer learns a small correction
- The final output combines both: original + correction

This allows you to adapt large pre-trained models with 100x fewer trainable parameters!

## How It Works

LoRA adapters enable efficient fine-tuning of neural networks by learning low-rank decompositions
of weight updates instead of modifying all weights directly. This interface defines the contract
for all LoRA adapter implementations across different layer types.

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the scaling factor (alpha) for the LoRA adaptation. |
| `BaseLayer` | Gets the base layer being adapted with LoRA. |
| `IsBaseLayerFrozen` | Gets whether the base layer's parameters are frozen during training. |
| `LoRALayer` | Gets the LoRA layer providing the low-rank adaptation. |
| `Rank` | Gets the rank of the low-rank decomposition. |

## Methods

| Method | Summary |
|:-----|:--------|
| `MergeToOriginalLayer` | Merges the LoRA weights back into the original layer for deployment. |

