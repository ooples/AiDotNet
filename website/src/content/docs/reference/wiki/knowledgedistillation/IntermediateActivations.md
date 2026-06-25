---
title: "IntermediateActivations<T>"
description: "Stores intermediate layer activations collected during a forward pass."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation`

Stores intermediate layer activations collected during a forward pass.

## For Beginners

During neural network training, we often want to inspect or use
the outputs from internal layers (not just the final output). These internal outputs are called
"intermediate activations". This class stores them in a dictionary keyed by layer name.

## How It Works

**Use Cases:**

- Feature-based distillation: Match intermediate layer outputs between teacher and student
- Neuron selectivity: Analyze how individual neurons respond across a batch
- Attention transfer: Transfer attention patterns from teacher to student
- Debugging: Inspect what each layer is learning

**Example Usage:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AllActivations` | Gets all stored activations as a read-only dictionary. |
| `LayerCount` | Gets the number of layers with stored activations. |
| `LayerNames` | Gets the names of all layers with stored activations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(String,Matrix<>)` | Adds intermediate activations for a specific layer. |
| `Contains(String)` | Checks if activations exist for a specific layer. |
| `Get(String)` | Retrieves intermediate activations for a specific layer. |

