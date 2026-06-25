---
title: "GraphConvolutionalLoRAAdapter"
description: "LoRA adapter for Graph Convolutional layers, enabling parameter-efficient fine-tuning of GNN models."
section: "Reference"
---

_LoRA / PEFT Adapters_

LoRA adapter for Graph Convolutional layers, enabling parameter-efficient fine-tuning of GNN models.

## For Beginners

LoRA for GNNs allows you to fine-tune large pre-trained
graph neural networks with a fraction of the trainable parameters.

**Why LoRA for GNNs?**

- Pre-trained GNN models can be huge (millions of parameters)
- Fine-tuning all parameters requires lots of memory
- LoRA learns small "correction" matrices instead
- Result: 10-100x fewer trainable parameters

**How it works:**

- Original GNN layer stays frozen (no updates)
- LoRA adds two small matrices (A and B) that learn adaptations
- Output = original_output + LoRA_correction
- Only A and B are trained, saving memory and time

**Example - Fine-tuning a GNN for drug discovery:**
```cs
// Wrap existing GAT layer with LoRA
var gatLayer = new GraphAttentionLayer<double>(128, 64, numHeads: 8);
var loraGat = new GraphConvolutionalLoRAAdapter<double>(
gatLayer, rank: 8, alpha: 16);

// Now train only the LoRA parameters
loraGat.UpdateParameters(learningRate);

// After training, merge LoRA into original layer
var mergedLayer = loraGat.MergeToOriginalLayer();
```

**Supported base layers:**

- GraphConvolutionalLayer (GCN)
- GraphAttentionLayer (GAT)
- GraphSAGELayer
- GraphIsomorphismLayer (GIN)
- Any layer implementing IGraphConvolutionLayer

## How It Works

This adapter enables LoRA (Low-Rank Adaptation) for graph neural network layers.
It wraps a graph convolutional layer (GCN, GAT, GraphSAGE, GIN) and adds a low-rank
adaptation that can be efficiently trained while keeping the base layer frozen.

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new GraphConvolutionalLoRAAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured GraphConvolutionalLoRAAdapter (rank {config.Rank}).");
```

