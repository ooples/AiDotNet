---
title: "XLoRAAdapter<T>"
description: "X-LoRA (Mixture of LoRA Experts) adapter that uses multiple LoRA experts with learned routing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LoRA.Adapters`

X-LoRA (Mixture of LoRA Experts) adapter that uses multiple LoRA experts with learned routing.

## For Beginners

X-LoRA is like having multiple specialists instead of one generalist.

Think of it like this:

- Standard LoRA: One adapter tries to handle all tasks
- X-LoRA: Multiple expert adapters, each specializing in different patterns
- A "gating network" decides which experts to use for each input

Real-world analogy: Instead of one doctor handling all patients, you have:

- Expert 1: Specializes in one type of pattern (e.g., cat images)
- Expert 2: Specializes in another pattern (e.g., dog images)
- Expert 3: Handles other cases
- Gating network: Looks at each input and decides which expert(s) to consult

Benefits:

- More capacity: Multiple experts can learn different aspects
- Better specialization: Each expert focuses on what it's good at
- Dynamic routing: Different inputs activate different experts
- Efficient: Only computes what's needed for each input

Example: For a 1000x1000 layer with 4 experts at rank=4 each:

- Total LoRA parameters: 4 * (4 * 1000 + 4 * 1000) = 32,000 parameters
- Gating network: ~1000 parameters
- Total: ~33,000 parameters (still 96.7% reduction from 1M!)
- But with more capacity than single rank=16 LoRA (32,000 params)

Trade-offs:
+ More flexible: Experts specialize in different patterns
+ Better performance: Often outperforms single LoRA at same parameter count
+ Dynamic routing: Adapts to different inputs

- More complex: Requires training gating network
- Slightly slower: Must compute multiple experts and gating weights

Reference: "Mixture of LoRA Experts" (X-LoRA)
https://arxiv.org/abs/2402.07148

## How It Works

X-LoRA extends standard LoRA by using a mixture of experts approach:

- Multiple LoRA adapters ("experts") are applied to the same layer
- A gating network learns to weight each expert's contribution based on the input
- Different inputs may activate different experts, allowing for more flexible adaptation
- This provides greater capacity than a single LoRA adapter with the same total rank

The forward pass computes:

- base_output = base_layer(input)
- For each expert i: expert_output[i] = lora_expert[i](input)
- gating_weights = softmax(gating_network(input))
- final_lora_output = sum(gating_weights[i] * expert_output[i])
- output = base_output + final_lora_output

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `XLoRAAdapter(ILayer<>,Int32,Int32,Double,Boolean)` | Initializes a new X-LoRA adapter with the specified parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Experts` | Gets the array of LoRA expert layers. |
| `GatingNetwork` | Gets the gating network used for routing. |
| `NumberOfExpertss` | Gets the number of LoRA experts in this adapter. |
| `ParameterCount` | Gets the total number of trainable parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass using mixture of LoRA experts. |
| `GetLastGatingWeights` | Gets the gating weights from the last forward pass. |
| `GetParameters` | Gets the current parameters as a vector. |
| `MergeToOriginalLayer` | Merges all LoRA expert adaptations into the base layer and returns the merged layer. |
| `ResetState` | Resets the internal state of the base layer, all experts, and the gating network. |
| `SetParameters(Vector<>)` | Sets the layer parameters from a vector. |
| `UpdateLayersFromParameters` | Updates the layers from the parameter vector. |
| `UpdateParameterGradientsFromLayers` | Updates the parameter gradients vector from the layer gradients. |
| `UpdateParameters()` | Updates parameters using the specified learning rate. |
| `UpdateParametersFromLayers` | Updates the parameter vector from the current layer states. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_experts` | Array of LoRA expert layers. |
| `_gatingNetwork` | Gating network that computes expert weights for each input. |
| `_lastExpertOutputs` | Temporary storage for expert outputs during forward pass (needed for backward pass). |
| `_lastGatingWeights` | Temporary storage for gating weights during forward pass (needed for backward pass). |
| `_lastInput` | Temporary storage for the last input during forward pass (needed for backward pass). |

