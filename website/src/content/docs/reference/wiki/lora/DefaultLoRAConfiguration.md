---
title: "DefaultLoRAConfiguration<T>"
description: "Default LoRA configuration that applies LoRA to all layers with trainable weight matrices."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.LoRA`

Default LoRA configuration that applies LoRA to all layers with trainable weight matrices.

## For Beginners

This is a ready-to-use LoRA configuration for most common scenarios.

When you apply this configuration to a model:

- All Dense layers get wrapped with LoRA adapters
- All FullyConnected layers get wrapped with LoRA adapters
- All other layers (convolutional, pooling, etc.) pass through unchanged

This is perfect for:

- Fine-tuning pre-trained models on new tasks
- Adapting large language models with limited resources
- Training multiple task-specific adapters for the same base model

Example usage:
```cs
// Create a configuration with rank=8, alpha=8, and frozen base layers
var loraConfig = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, freezeBaseLayer: true);

// Apply to all layers in your model
var adaptedLayers = model.Layers.Select(layer => loraConfig.ApplyLoRA(layer)).ToList();
```

The configuration respects these parameters:

- Rank: Controls compression (fewer parameters = lower rank)
- Alpha: Controls adaptation strength (typically same as rank)
- FreezeBaseLayer: Whether to freeze original weights (true for efficiency)

## How It Works

This configuration implements an intelligent strategy: wrap all layers that have trainable
weight matrices with StandardLoRAAdapter, and leave utility layers (activation, pooling, etc.)
unchanged. This maximizes the benefits of LoRA across all applicable layer types.

**Supported Layer Types (30+ layer types):**

- Dense/Linear layers (Dense, FullyConnected, FeedForward)
- Convolutional layers (all Conv variants including depthwise, separable, dilated, etc.)
- Recurrent layers (LSTM, GRU, ConvLSTM, Bidirectional)
- Attention layers (Attention, MultiHeadAttention, SelfAttention)
- Transformer layers (Encoder, Decoder)
- Embedding layers (Embedding, PatchEmbedding)
- Specialized layers (Highway, GatedLinearUnit, SqueezeAndExcitation, Capsule, CRF, etc.)

**Available LoRA Variants:** AiDotNet includes 32 cutting-edge LoRA variants for different use cases:

- StandardLoRAAdapter: Generic LoRA for all layer types
- QLoRAAdapter: 4-bit quantization for 75% memory reduction
- DoRAAdapter: Weight decomposition (+3.7% on LLaMA-7B)
- AdaLoRAAdapter: Adaptive rank allocation
- VeRAAdapter: Shared matrices (10x fewer parameters)
- LoRAPlusAdapter: Dual learning rates (2x faster convergence)
- LoHaAdapter: Hadamard products for CNNs
- LoKrAdapter: Kronecker products (57x compression)
- DyLoRAAdapter: Dynamic rank training
- RoSAAdapter: Robust to distribution shifts
- DVoRAAdapter: DoRA+VeRA hybrid
- LoRAFAAdapter: Frozen A matrix (50% reduction)
- DeltaLoRAAdapter: Delta-based updates with momentum
- LoRADropAdapter: Dropout regularization
- PiSSAAdapter: SVD initialization (NeurIPS 2024)
- GLoRAAdapter: Weight + activation adaptation
- LongLoRAAdapter: Context length extension
- MultiLoRAAdapter: Multi-task learning with routing
- XLoRAAdapter: Mixture of experts
- TiedLoRAAdapter: Weight tying (90% reduction)
- ReLoRAAdapter: Restart mechanism prevents forgetting
- LoftQAdapter: Alternating quantization+LoRA
- QALoRAAdapter: Quantization-aware training
- VBLoRAAdapter: Vector banks (2024)
- SLoRAAdapter: Scalable serving (1000+ adapters)
- MoRAAdapter: High-rank updates for knowledge tasks
- LoRAXSAdapter: Extreme efficiency (100x compression)
- FloraAdapter: Gradient compression view
- ChainLoRAAdapter: Sequential task chaining
- HRAAdapter: Hybrid low-rank + sparse
- LoRETTAAdapter: Tensor-train decomposition
- NOLAAdapter: Random basis (20x compression)

To use a specific variant, pass a factory function to the constructor.
Example: new DefaultLoRAConfiguration<double>(rank: 8, adapterFactory: (layer, r, a, f) => new QLoRAAdapter<double>(layer, r, a, f))

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DefaultLoRAConfiguration(Int32,Double,Boolean,ILoRAAdapter<>)` | Initializes a new DefaultLoRAConfiguration with the specified parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets the scaling factor (alpha) for LoRA adaptations. |
| `FreezeBaseLayer` | Gets whether base layers should be frozen during training. |
| `Rank` | Gets the rank of the low-rank decomposition to use for adapted layers. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdaptBlockSublayer(LayerBase<>,Action<LayerBase<>>)` | Recursively applies LoRA to one sublayer hosted inside a composite transformer block and, when the sublayer was actually wrapped, installs the adapter through the block's replace hook so the block's sublayer registration (tape parameter dis… |
| `ApplyLoRA(ILayer<>)` | Applies LoRA adaptation to layers with trainable weight matrices. |
| `ApplyLoRAToModel(ILayeredModel<>)` | Applies LoRA adapters to all eligible layers in a layered model, returning the list of adapted layers. |
| `CreateAdapter(ILayer<>)` | Creates an instance of the configured LoRA adapter for the given layer. |
| `IsLoRATarget(ILayer<>)` | Non-mutating predicate: returns `true` when this configuration would wrap `layer` with a LoRA adapter (modulo the shape-resolved guard, which is independent of the layer type). |
| `IsLoRATargetType(ILayer<>)` | Whitelist of concrete layer types that `ILayer{` wraps via `ILayer{`. |
| `ShouldApplyLoRA(LayerCategory)` | Determines whether a layer should have LoRA applied based on its `LayerCategory`. |
| `TryDeclareShapeSafely(LayerBase<>)` | Exception-safe wrapper over the `TryDeclareShape` shape oracle (same policy as AiModelBuilder's pre-scan): a throwing oracle means "needs warmup" — treat the layer as not wrappable rather than failing the whole LoRA pass. |

