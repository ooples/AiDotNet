---
title: "LLM Fine-tuning"
description: "LoRA, QLoRA, and efficient fine-tuning."
order: 9
section: "Tutorials"
---

Fine-tune large language models efficiently with LoRA and QLoRA.

## Overview

AiDotNet provides 34 LoRA adapter implementations for efficient fine-tuning:

| Category | Adapters |
|:---------|:---------|
| Core | `StandardLoRAAdapter`, `QLoRAAdapter`, `DoRAAdapter`, `AdaLoRAAdapter` |
| Low-Parameter | `VeRAAdapter`, `LoRAXSAdapter`, `NOLAAdapter`, `VBLoRAAdapter`, `LoRAFAAdapter` |
| Composition | `LoHaAdapter`, `LoKrAdapter`, `DenseLoRAAdapter`, `GLoRAAdapter`, `MultiLoRAAdapter`, `ChainLoRAAdapter` |
| Efficiency | `DyLoRAAdapter`, `FloraAdapter`, `SLoRAAdapter`, `LoftQAdapter`, `PiSSAAdapter` |
| Advanced | `MoRAAdapter`, `DVoRAAdapter`, `DeltaLoRAAdapter`, `HRAAdapter`, `RoSAAdapter` |
| Specialized | `LongLoRAAdapter`, `GraphConvolutionalLoRAAdapter`, `LoRETTAAdapter`, `ReLoRAAdapter` |
| Scaling | `LoRAPlusAdapter`, `LoRADropAdapter`, `XLoRAAdapter`, `QALoRAAdapter`, `TiedLoRAAdapter` |

## Why LoRA?

| Method | Memory (7B Model) | Parameters Trained |
|:-------|:------------------|:-------------------|
| Full Fine-tune | 28+ GB | 100% |
| LoRA (r=8) | 8-12 GB | ~0.1% |
| QLoRA (4-bit) | 4-6 GB | ~0.1% |

---

## Basic LoRA with AiModelBuilder

All LoRA fine-tuning uses the standard `AiModelBuilder` pattern with `.ConfigureLoRA()`:

```csharp
using AiDotNet;
using AiDotNet.LoRA;

// Configure LoRA via ILoRAConfiguration
var loraConfig = new DefaultLoRAConfiguration<float>(
    rank: 8,
    alpha: 16.0f,
    dropout: 0.05f
);

// Build model with LoRA using AiModelBuilder
var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>();
var result = await builder
    .ConfigureModel(model)
    .ConfigureLoRA(loraConfig)
    .ConfigureOptimizer(new AdamWOptimizer<float>(learningRate: 1e-4f))
    .BuildAsync(trainingData, trainingLabels);

// Make predictions
var predictions = builder.Predict(testData, result);
```

---

## Training Configuration

```csharp
// Prepare training data (use realistic dataset sizes)
var trainingData = LoadTrainingFeatures();   // Tensor<float> of shape [numSamples, inputDim]
var trainingLabels = LoadTrainingLabels();   // Tensor<float> of shape [numSamples, outputDim]

// Configure LoRA
var loraConfig = new DefaultLoRAConfiguration<float>(
    rank: 8,
    alpha: 16.0f,
    dropout: 0.05f
);

// Train with AiModelBuilder
var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>();
var result = await builder
    .ConfigureModel(model)
    .ConfigureLoRA(loraConfig)
    .ConfigureOptimizer(new AdamWOptimizer<float>(
        learningRate: 1e-4f,
        weightDecay: 0.01f))
    .ConfigureLearningRateScheduler(new CosineAnnealingLR(tMax: 100))
    .ConfigurePreprocessing()
    .BuildAsync(trainingData, trainingLabels);

Console.WriteLine($"Training Loss: {result.TrainingLoss:F4}");
Console.WriteLine($"Validation Loss: {result.ValidationLoss:F4}");
```

---

## QLoRA (4-bit Quantized)

QLoRA applies LoRA on top of a quantized model to reduce memory usage:

```csharp
using AiDotNet;
using AiDotNet.LoRA;

// QLoRA uses the QLoRAAdapter directly
var architecture = new NeuralNetworkArchitecture<float>(
    inputType: InputType.OneDimensional,
    taskType: NeuralNetworkTaskType.MultiClassClassification,
    inputSize: inputDim,
    outputSize: numClasses
);

var loraConfig = new DefaultLoRAConfiguration<float>(
    rank: 8,
    alpha: 16.0f,
    dropout: 0.05f
);

var builder = new AiModelBuilder<float, Tensor<float>, Tensor<float>>();
var result = await builder
    .ConfigureModel(model)
    .ConfigureLoRA(loraConfig)
    .ConfigureQuantization(new QuantizationConfig { Bits = 4 })
    .ConfigureOptimizer(new AdamWOptimizer<float>(learningRate: 1e-4f))
    .BuildAsync(trainingData, trainingLabels);
```

---

## Saving and Loading Models

```csharp
// Save trained model in AIMF format (includes LoRA weights)
builder.SaveModel(result, "lora-finetuned.aimf");

// Save with encryption (requires license key)
builder.SaveModel(result, "lora-finetuned-encrypted.aimf", encrypt: true);

// Load and use later
var loadedResult = builder.LoadModel("lora-finetuned.aimf");
var predictions = builder.Predict(testData, loadedResult);
```

---

## LoRA Adapter Variants

AiDotNet provides specialized adapter implementations for different scenarios. Each adapter extends `LoRAAdapterBase<T>` and implements `ILoRAAdapter<T>`.

### DoRA (Weight-Decomposed LoRA)

Decomposes weight updates into magnitude and direction for better convergence:

```csharp
// DoRAAdapter decomposes updates into magnitude and direction
var doraAdapter = new DoRAAdapter<float>(
    inputSize: 256,
    outputSize: 256,
    rank: 8,
    alpha: 16.0f
);
```

### AdaLoRA (Adaptive Rank)

Dynamically adjusts rank per layer based on importance:

```csharp
var adaLoraAdapter = new AdaLoRAAdapter<float>(
    inputSize: 256,
    outputSize: 256,
    initialRank: 12,
    targetRank: 8,
    alpha: 16.0f
);
```

### VeRA (Vector-based LoRA)

Uses shared random matrices with learned scaling vectors for extreme parameter efficiency:

```csharp
var veraAdapter = new VeRAAdapter<float>(
    inputSize: 256,
    outputSize: 256,
    rank: 256,
    alpha: 16.0f
);
```

---

## Best Practices

### Rank Selection

| Task | Recommended Rank |
|:-----|:-----------------|
| Simple tasks | 4-8 |
| Complex tasks | 16-32 |
| Multi-task | 32-64 |

### Training Tips

1. **Start small**: Try rank 4-8 first, increase if underfitting
2. **Learning rate**: Use lower rates than full fine-tuning (1e-4 to 5e-5)
3. **Gradient accumulation**: Use gradient accumulation if batch size is memory-limited
4. **Regularization**: Apply dropout (0.05-0.1) to prevent overfitting
5. **Validation monitoring**: Split data into train/validation sets and track validation loss to detect overfitting early

```csharp
// Example with validation monitoring via AiModelBuilder
var result = await builder
    .ConfigureModel(model)
    .ConfigureLoRA(loraConfig)
    .ConfigureOptimizer(new AdamWOptimizer<float>(learningRate: 1e-4f))
    .ConfigurePreprocessing()
    .BuildAsync(trainingData, trainingLabels);

// Check for overfitting
if (result.ValidationLoss > result.TrainingLoss * 1.5)
{
    Console.WriteLine("Warning: model may be overfitting. Consider increasing dropout or reducing rank.");
}
```

---

## Memory Comparison

| Model | Full FT | LoRA | QLoRA |
|:------|:--------|:-----|:------|
| 7B | 28+ GB | 10 GB | 5 GB |
| 13B | 52+ GB | 18 GB | 8 GB |
| 70B | OOM | 90 GB | 24 GB |

---

## Next Steps

- [LoRA/Fine-tuning Features](../../../features/lora-finetuning/)
- [Optimizers Reference](../../reference/optimizers/)
- [Getting Started Guide](../../getting-started/installation/)
