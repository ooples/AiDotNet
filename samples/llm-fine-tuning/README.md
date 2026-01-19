# LLM Fine-Tuning Samples

This directory contains examples of fine-tuning large language models with AiDotNet.

## Available Samples

| Sample | Description |
|--------|-------------|
| [LoRA](./LoRA/) | Low-Rank Adaptation fine-tuning |
| [QLoRA](./QLoRA/) | Quantized LoRA for memory efficiency |

## Quick Start

```csharp
using AiDotNet;
using AiDotNet.LoRA;
using AiDotNet.ModelLoading;

// Load base model
var model = await HuggingFaceHub.LoadModelAsync<float>("microsoft/phi-2");

// Configure LoRA
var loraConfig = new LoRAConfig<float>
{
    Rank = 8,
    Alpha = 16,
    TargetModules = ["q_proj", "v_proj"],
    Dropout = 0.05f
};

// Apply LoRA
var loraModel = model.ApplyLoRA(loraConfig);

// Fine-tune
await loraModel.TrainAsync(trainingData, epochs: 3);

// Merge and save
var mergedModel = loraModel.MergeAndUnload();
await mergedModel.SaveAsync("fine-tuned-model");
```

## LoRA Variants (37+)

| Variant | Description |
|---------|-------------|
| LoRA | Low-Rank Adaptation |
| QLoRA | Quantized LoRA |
| DoRA | Weight-Decomposed LoRA |
| AdaLoRA | Adaptive LoRA |
| VeRA | Vector-based LoRA |
| LoHa | Low-Rank Hadamard Product |
| LoKr | Low-Rank Kronecker Product |

## Memory Comparison

| Method | 7B Model Memory |
|--------|-----------------|
| Full Fine-tune | 28+ GB |
| LoRA | 8-12 GB |
| QLoRA (4-bit) | 4-6 GB |

## Learn More

- [LoRA Tutorial](/docs/tutorials/lora/)
- [API Reference](/api/AiDotNet.LoRA/)
