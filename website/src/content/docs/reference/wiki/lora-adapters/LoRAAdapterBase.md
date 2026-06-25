---
title: "LoRAAdapterBase"
description: "Abstract base class for LoRA (Low-Rank Adaptation) adapters that wrap existing layers."
section: "Reference"
---

_LoRA / PEFT Adapters_

Abstract base class for LoRA (Low-Rank Adaptation) adapters that wrap existing layers.

## For Beginners

This is the foundation for all LoRA adapters in the library.

A LoRA adapter wraps an existing layer (like a dense or convolutional layer) and adds
a small "correction layer" that learns what adjustments are needed. This base class:

- Manages both the original layer and the LoRA correction layer
- Handles parameter synchronization between them
- Provides common forward/backward pass logic (original + correction)
- Lets specialized adapters handle layer-specific details

This design allows you to create LoRA adapters for any layer type by:

1. Inheriting from this base class
2. Implementing layer-specific validation
3. Implementing how to merge the LoRA weights back into the original layer

The result is parameter-efficient fine-tuning that works across different layer architectures!

## How It Works

This base class provides common functionality for all LoRA adapter implementations.
It manages the base layer, LoRA layer, and parameter synchronization, while allowing
derived classes to implement layer-type-specific logic such as merging and validation.

