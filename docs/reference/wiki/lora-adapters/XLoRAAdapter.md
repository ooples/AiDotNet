---
title: "XLoRAAdapter"
description: "X-LoRA (Mixture of LoRA Experts) adapter that uses multiple LoRA experts with learned routing."
section: "Reference"
---

_LoRA / PEFT Adapters_

X-LoRA (Mixture of LoRA Experts) adapter that uses multiple LoRA experts with learned routing.

## For Beginners

X-LoRA is like having multiple specialists instead of one generalist. Think of it like this: - Standard LoRA: One adapter tries to handle all tasks - X-LoRA: Multiple expert adapters, each specializing in different patterns - A "gating network" decides which experts to use for each input Real-world analogy: Instead of one doctor handling all patients, you have: - Expert 1: Specializes in one type of pattern (e.g., cat images) - Expert 2: Specializes in another pattern (e.g., dog images) - Expert 3: Handles other cases - Gating network: Looks at each input and decides which expert(s) to consult Benefits: - More capacity: Multiple experts can learn different aspects - Better specialization: Each expert focuses on what it's good at - Dynamic routing: Different inputs activate different experts - Efficient: Only computes what's needed for each input Example: For a 1000x1000 layer with 4 experts at rank=4 each: - Total LoRA parameters: 4 * (4 * 1000 + 4 * 1000) = 32,000 parameters - Gating network: ~1000 parameters - Total: ~33,000 parameters (still 96.7% reduction from 1M!) - But with more capacity than single rank=16 LoRA (32,000 params) Trade-offs: + More flexible: Experts specialize in different patterns + Better performance: Often outperforms single LoRA at same parameter count + Dynamic routing: Adapts to different inputs - More complex: Requires training gating network - Slightly slower: Must compute multiple experts and gating weights Reference: "Mixture of LoRA Experts" (X-LoRA) https://arxiv.org/abs/2402.07148

## How It Works

X-LoRA extends standard LoRA by using a mixture of experts approach: - Multiple LoRA adapters ("experts") are applied to the same layer - A gating network learns to weight each expert's contribution based on the input - Different inputs may activate different experts, allowing for more flexible adaptation - This provides greater capacity than a single LoRA adapter with the same total rank 

The forward pass computes: - base_output = base_layer(input) - For each expert i: expert_output[i] = lora_expert[i](input) - gating_weights = softmax(gating_network(input)) - final_lora_output = sum(gating_weights[i] * expert_output[i]) - output = base_output + final_lora_output

