---
layout: default
title: LLM Fine-tuning
parent: Tutorials
nav_order: 8
has_children: true
permalink: /tutorials/llm-fine-tuning/
---

# LLM Fine-tuning Tutorial
{: .no_toc }

Fine-tune large language models efficiently with LoRA and QLoRA.
{: .fs-6 .fw-300 }

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

AiDotNet provides 37+ LoRA adapters for efficient fine-tuning:
- **LoRA**: Low-Rank Adaptation
- **QLoRA**: 4-bit Quantized LoRA
- **DoRA**: Weight-Decomposed LoRA
- **AdaLoRA**: Adaptive LoRA
- And many more!

---

## Why LoRA?

| Method | Memory (7B Model) | Parameters Trained |
|:-------|:------------------|:-------------------|
| Full Fine-tune | 28+ GB | 100% |
| LoRA (r=8) | 8-12 GB | ~0.1% |
| QLoRA (4-bit) | 4-6 GB | ~0.1% |

---

## Basic LoRA

```csharp
using AiDotNet.LoRA;
using AiDotNet.ModelLoading;

// Load base model
var model = await HuggingFaceHub.LoadModelAsync<float>("microsoft/phi-2");

// Configure LoRA
var loraConfig = new LoRAConfig<float>
{
    Rank = 8,                           // Low-rank dimension
    Alpha = 16,                         // Scaling factor
    TargetModules = ["q_proj", "v_proj"], // Layers to adapt
    Dropout = 0.05f
};

// Apply LoRA adapters
var loraModel = model.ApplyLoRA(loraConfig);

Console.WriteLine($"Trainable parameters: {loraModel.GetTrainableParameterCount():N0}");
// ~0.1% of original parameters
```

---

## Training with LoRA

```csharp
// Prepare training data
var trainingData = new[]
{
    new TrainingSample { Input = "Translate to French: Hello", Output = "Bonjour" },
    new TrainingSample { Input = "Translate to French: Goodbye", Output = "Au revoir" }
};

// Configure training
var trainingConfig = new LoRATrainingConfig<float>
{
    LearningRate = 1e-4f,
    BatchSize = 4,
    GradientAccumulationSteps = 4,
    Epochs = 3,
    WarmupSteps = 100,
    WeightDecay = 0.01f
};

// Train
await loraModel.TrainAsync(trainingData, trainingConfig);
```

---

## QLoRA (4-bit Quantized)

Train even larger models with minimal memory:

```csharp
using AiDotNet.LoRA;
using AiDotNet.Quantization;

// Configure 4-bit quantization
var quantConfig = new QuantizationConfig<float>
{
    QuantizationType = QuantizationType.NF4,  // 4-bit NormalFloat
    ComputeType = DataType.BFloat16,
    DoubleQuantization = true  // Quantize the quantization constants
};

// Load with quantization
var quantizedModel = await HuggingFaceHub.LoadModelAsync<float>(
    "meta-llama/Llama-2-7b-hf",
    quantConfig);

// Apply LoRA on quantized model
var qloraConfig = new QLoRAConfig<float>
{
    Rank = 8,
    Alpha = 16,
    TargetModules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    Dropout = 0.05f
};

var qloraModel = quantizedModel.ApplyQLoRA(qloraConfig);

// Train (uses much less memory!)
await qloraModel.TrainAsync(trainingData, trainingConfig);
```

---

## Saving and Loading Adapters

```csharp
// Save only the LoRA adapters (small file)
await loraModel.SaveAdaptersAsync("my-lora-adapters");

// Load adapters onto base model
var baseModel = await HuggingFaceHub.LoadModelAsync<float>("microsoft/phi-2");
var loadedLoraModel = baseModel.LoadLoRAAdapters("my-lora-adapters");

// Merge adapters into base model (for deployment)
var mergedModel = loraModel.MergeAndUnload();
await mergedModel.SaveAsync("merged-model");
```

---

## LoRA Variants

### DoRA (Weight-Decomposed LoRA)

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

### AdaLoRA (Adaptive Rank)

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

```csharp
var veraConfig = new VeRAConfig<float>
{
    Rank = 256,  // Higher rank possible due to shared matrices
    Alpha = 16,
    SharedAcrossLayers = true
};

var veraModel = model.ApplyVeRA(veraConfig);
```

---

## Multiple LoRA Adapters

Switch between different adapters at inference:

```csharp
// Load base model
var model = await HuggingFaceHub.LoadModelAsync<float>("microsoft/phi-2");

// Load multiple adapters
model.LoadLoRAAdapters("translation-adapter", "translation");
model.LoadLoRAAdapters("summarization-adapter", "summarization");

// Use specific adapter
model.SetActiveAdapter("translation");
var translation = model.Generate("Translate to French: Hello");

model.SetActiveAdapter("summarization");
var summary = model.Generate("Summarize: ...");
```

---

## Dataset Preparation

### Instruction Fine-tuning Format

```csharp
var dataset = new[]
{
    new InstructionSample
    {
        Instruction = "Summarize the following text.",
        Input = "AiDotNet is a machine learning framework...",
        Output = "AiDotNet is a .NET ML framework."
    }
};
```

### Chat Format

```csharp
var chatDataset = new[]
{
    new ChatSample
    {
        Messages = new[]
        {
            new Message { Role = "user", Content = "What is AI?" },
            new Message { Role = "assistant", Content = "AI is..." }
        }
    }
};
```

---

## Best Practices

### Rank Selection

| Task | Recommended Rank |
|:-----|:-----------------|
| Simple tasks | 4-8 |
| Complex tasks | 16-32 |
| Multi-task | 32-64 |

### Target Modules

```csharp
// Attention layers only (most efficient)
TargetModules = ["q_proj", "v_proj"]

// All attention layers
TargetModules = ["q_proj", "k_proj", "v_proj", "o_proj"]

// Attention + MLP (most expressive)
TargetModules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### Training Tips

1. **Start small**: Try rank 4-8 first
2. **Learning rate**: Lower than full fine-tuning (1e-4 to 5e-5)
3. **Batch size**: Use gradient accumulation if memory limited
4. **Regularization**: Use dropout (0.05-0.1)
5. **Early stopping**: Monitor validation loss

---

## Memory Comparison

| Model | Full FT | LoRA | QLoRA |
|:------|:--------|:-----|:------|
| 7B | 28+ GB | 10 GB | 5 GB |
| 13B | 52+ GB | 18 GB | 8 GB |
| 70B | OOM | 90 GB | 24 GB |

---

## Next Steps

- [LoRA Sample](/samples/llm-fine-tuning/LoRA/)
- [QLoRA Sample](/samples/llm-fine-tuning/QLoRA/)
- [LoRA API Reference](/api/AiDotNet.LoRA/)
