---
layout: default
title: LoRA Adapters
parent: Reference
nav_order: 5
permalink: /reference/lora-adapters/
---

# LoRA Adapters
{: .no_toc }

Complete reference for all 37+ LoRA adapter variants in AiDotNet.
{: .fs-6 .fw-300 }

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

LoRA (Low-Rank Adaptation) enables efficient fine-tuning of large models by training only small adapter matrices, reducing memory requirements by 90%+ while maintaining performance.

---

## Standard LoRA

### Basic LoRA

```csharp
var loraConfig = new LoRAConfig<float>
{
    Rank = 8,                           // Low-rank dimension
    Alpha = 16,                         // Scaling factor
    TargetModules = ["q_proj", "v_proj"], // Layers to adapt
    Dropout = 0.05f
};

var loraModel = model.ApplyLoRA(loraConfig);
```

### Configuration Options

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `Rank` | int | 8 | Low-rank dimension (r) |
| `Alpha` | int | 16 | Scaling factor (α) |
| `TargetModules` | string[] | ["q_proj", "v_proj"] | Layers to adapt |
| `Dropout` | float | 0.0 | Dropout probability |
| `BiasMode` | BiasMode | None | How to handle biases |

### Memory Comparison

| Model Size | Full Fine-tune | LoRA (r=8) | Savings |
|:-----------|:---------------|:-----------|:--------|
| 1B | 4 GB | 0.4 GB | 90% |
| 7B | 28 GB | 2.8 GB | 90% |
| 13B | 52 GB | 5.2 GB | 90% |
| 70B | 280 GB | 28 GB | 90% |

---

## QLoRA (Quantized LoRA)

Train on 4-bit quantized models for maximum memory efficiency:

```csharp
var quantConfig = new QuantizationConfig<float>
{
    QuantizationType = QuantizationType.NF4,  // 4-bit NormalFloat
    ComputeType = DataType.BFloat16,
    DoubleQuantization = true
};

var quantizedModel = await HuggingFaceHub.LoadModelAsync<float>(
    "meta-llama/Llama-2-7b-hf",
    quantConfig);

var qloraConfig = new QLoRAConfig<float>
{
    Rank = 8,
    Alpha = 16,
    TargetModules = ["q_proj", "k_proj", "v_proj", "o_proj"]
};

var qloraModel = quantizedModel.ApplyQLoRA(qloraConfig);
```

### QLoRA Memory Comparison

| Model | Full FT | LoRA | QLoRA |
|:------|:--------|:-----|:------|
| 7B | 28+ GB | 10 GB | 5 GB |
| 13B | 52+ GB | 18 GB | 8 GB |
| 70B | OOM | 90 GB | 24 GB |

---

## LoRA Variants

### DoRA (Weight-Decomposed LoRA)

Decomposes weights into magnitude and direction for better fine-tuning:

```csharp
var doraConfig = new DoRAConfig<float>
{
    Rank = 8,
    Alpha = 16,
    TargetModules = ["q_proj", "v_proj"],
    DecomposeMagnitude = true
};

var doraModel = model.ApplyDoRA(doraConfig);
```

### AdaLoRA (Adaptive LoRA)

Adaptively allocates rank budget to different layers:

```csharp
var adaLoraConfig = new AdaLoRAConfig<float>
{
    InitialRank = 12,
    TargetRank = 8,
    Alpha = 16,
    BetaStart = 0.85f,
    BetaEnd = 0.95f
};

var adaLoraModel = model.ApplyAdaLoRA(adaLoraConfig);
```

### VeRA (Vector-based LoRA)

Uses shared random matrices with trainable scaling vectors:

```csharp
var veraConfig = new VeRAConfig<float>
{
    Rank = 256,  // Can use higher rank
    Alpha = 16,
    SharedAcrossLayers = true
};

var veraModel = model.ApplyVeRA(veraConfig);
```

### LoKr (Low-Rank Kronecker)

Uses Kronecker product for low-rank decomposition:

```csharp
var lokrConfig = new LoKrConfig<float>
{
    Factor = 16,
    Alpha = 16,
    TargetModules = ["q_proj", "v_proj"]
};

var lokrModel = model.ApplyLoKr(lokrConfig);
```

### LoHa (Low-Rank Hadamard)

Uses Hadamard product for decomposition:

```csharp
var lohaConfig = new LoHaConfig<float>
{
    Rank = 8,
    Alpha = 16,
    TargetModules = ["q_proj", "v_proj"]
};

var lohaModel = model.ApplyLoHa(lohaConfig);
```

### IA³ (Infused Adapter by Inhibiting and Amplifying)

Learns rescaling vectors instead of matrices:

```csharp
var ia3Config = new IA3Config<float>
{
    TargetModules = ["k_proj", "v_proj", "down_proj"],
    FeedforwardModules = ["down_proj"]
};

var ia3Model = model.ApplyIA3(ia3Config);
```

### (IA)³ Memory Usage

| Method | Parameters | Memory |
|:-------|:-----------|:-------|
| LoRA (r=8) | 0.1% | ~10% of full |
| IA³ | 0.01% | ~1% of full |

### Prefix Tuning

Adds trainable prefix tokens to attention:

```csharp
var prefixConfig = new PrefixTuningConfig<float>
{
    NumVirtualTokens = 20,
    ProjectionDim = 256
};

var prefixModel = model.ApplyPrefixTuning(prefixConfig);
```

### Prompt Tuning

Adds learnable soft prompts:

```csharp
var promptConfig = new PromptTuningConfig<float>
{
    NumVirtualTokens = 10,
    InitializerRange = 0.5f
};

var promptModel = model.ApplyPromptTuning(promptConfig);
```

### P-Tuning v2

Deep prompt tuning across all layers:

```csharp
var ptuningConfig = new PTuningV2Config<float>
{
    NumVirtualTokens = 20,
    EncoderHiddenSize = 128,
    EncoderNumLayers = 2
};

var ptuningModel = model.ApplyPTuningV2(ptuningConfig);
```

---

## Adapter Comparison

| Method | Params | Memory | Quality | Speed |
|:-------|:-------|:-------|:--------|:------|
| LoRA | 0.1% | Low | High | Fast |
| QLoRA | 0.1% | Very Low | High | Medium |
| DoRA | 0.1% | Low | Higher | Fast |
| AdaLoRA | 0.1% | Low | High | Medium |
| VeRA | 0.01% | Very Low | Good | Fast |
| LoKr | 0.05% | Low | Good | Fast |
| LoHa | 0.1% | Low | Good | Fast |
| IA³ | 0.01% | Very Low | Good | Very Fast |
| Prefix Tuning | 0.1% | Low | Good | Medium |
| Prompt Tuning | 0.01% | Very Low | Moderate | Fast |

---

## Target Module Selection

### Attention Only (Most Efficient)

```csharp
TargetModules = ["q_proj", "v_proj"]
```

### All Attention Layers

```csharp
TargetModules = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

### Attention + MLP (Most Expressive)

```csharp
TargetModules = ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]
```

---

## Multiple Adapters

Load and switch between multiple adapters:

```csharp
// Load multiple adapters
model.LoadLoRAAdapters("translation-adapter", "translation");
model.LoadLoRAAdapters("summarization-adapter", "summarization");

// Switch adapters
model.SetActiveAdapter("translation");
var translation = model.Generate("Translate to French: Hello");

model.SetActiveAdapter("summarization");
var summary = model.Generate("Summarize: ...");
```

### Merge Multiple Adapters

```csharp
// Merge adapters with weights
model.MergeAdapters(new Dictionary<string, float>
{
    ["translation"] = 0.7f,
    ["grammar"] = 0.3f
});
```

---

## Saving and Loading

### Save Adapters

```csharp
// Save only adapter weights (small file)
await loraModel.SaveAdaptersAsync("my-lora-adapters");
```

### Load Adapters

```csharp
var baseModel = await HuggingFaceHub.LoadModelAsync<float>("microsoft/phi-2");
var loadedModel = baseModel.LoadLoRAAdapters("my-lora-adapters");
```

### Merge and Export

```csharp
// Merge adapters into base model weights
var mergedModel = loraModel.MergeAndUnload();

// Save merged model
await mergedModel.SaveAsync("merged-model");
```

---

## Training Configuration

```csharp
var trainingConfig = new LoRATrainingConfig<float>
{
    LearningRate = 1e-4f,
    BatchSize = 4,
    GradientAccumulationSteps = 4,
    Epochs = 3,
    WarmupSteps = 100,
    WeightDecay = 0.01f,
    MaxGradNorm = 1.0f
};

await loraModel.TrainAsync(trainingData, trainingConfig);
```

---

## Rank Selection Guide

| Task Complexity | Recommended Rank |
|:----------------|:-----------------|
| Simple tasks (sentiment) | 4-8 |
| Medium tasks (translation) | 8-16 |
| Complex tasks (coding) | 16-32 |
| Multi-task | 32-64 |

---

## Best Practices

1. **Start with rank 8**: Good balance of efficiency and quality
2. **Use alpha = 2 × rank**: Common scaling factor
3. **Target q_proj and v_proj first**: Most efficient
4. **Use QLoRA for large models**: Enables training on consumer GPUs
5. **Lower learning rate**: 1e-4 to 5e-5 (lower than full fine-tuning)
6. **Gradient checkpointing**: For memory efficiency
7. **Evaluate on validation set**: Prevent overfitting
