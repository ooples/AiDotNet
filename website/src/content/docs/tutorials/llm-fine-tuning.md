---
title: "LLM Fine-tuning"
description: "LoRA, QLoRA, and efficient fine-tuning."
order: 9
section: "Tutorials"
---

Fine-tune models efficiently with LoRA and QLoRA through the `AiModelBuilder` facade.

## Why LoRA?

Full fine-tuning updates every weight — expensive in compute and memory. **LoRA** (Low-Rank Adaptation) freezes the base weights and trains only small rank-`r` adapter matrices, so you adapt a model at a tiny fraction of the parameters. **QLoRA** goes further by quantizing the frozen base, cutting memory again.

## LoRA Fine-Tuning

`ConfigureLoRA(...)` attaches the adapters; `BuildAsync()` wraps the dense layers and trains only the low-rank matrices.

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.LoRA;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

var model = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
    inputFeatures: 64, numClasses: 8, complexity: NetworkComplexity.Simple));

// rank = adapter capacity, alpha = scaling, freeze the base weights.
var loraConfig = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 16, freezeBaseLayer: true);

var rng = new Random(42);
var trainX = new Tensor<double>(new[] { 64, 64 });
var trainY = new Tensor<double>(new[] { 64, 8 });
for (int i = 0; i < 64; i++)
{
    for (int j = 0; j < 64; j++) trainX[new[] { i, j }] = rng.NextDouble();
    trainY[new[] { i, i % 8 }] = 1.0;
}

var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(model)
    .ConfigureLoRA(loraConfig)
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine($"Fine-tuned with LoRA (rank {loraConfig.Rank}, alpha {loraConfig.Alpha}).");
```

## QLoRA (Quantized Base + LoRA)

Add `ConfigureQuantization(...)` to quantize the frozen base before attaching adapters.

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Deployment.Configuration;
using AiDotNet.Enums;
using AiDotNet.LoRA;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

var model = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
    inputFeatures: 64, numClasses: 8, complexity: NetworkComplexity.Simple));

var loraConfig = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 16, freezeBaseLayer: true);
var quantConfig = new QuantizationConfig { Mode = QuantizationMode.Int8 };

var rng = new Random(7);
var trainX = new Tensor<double>(new[] { 64, 64 });
var trainY = new Tensor<double>(new[] { 64, 8 });
for (int i = 0; i < 64; i++)
{
    for (int j = 0; j < 64; j++) trainX[new[] { i, j }] = rng.NextDouble();
    trainY[new[] { i, i % 8 }] = 1.0;
}

var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(model)
    .ConfigureQuantization(quantConfig)
    .ConfigureLoRA(loraConfig)
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Fine-tuned with QLoRA (Int8 base + LoRA adapters).");
```

## Saving the Fine-Tuned Model

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.LoRA;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

var model = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
    inputFeatures: 16, numClasses: 4, complexity: NetworkComplexity.Simple));
var loraConfig = new DefaultLoRAConfiguration<double>(rank: 4, alpha: 8, freezeBaseLayer: true);

var trainX = new Tensor<double>(new[] { 32, 16 });
var trainY = new Tensor<double>(new[] { 32, 4 });
for (int i = 0; i < 32; i++) { trainX[new[] { i, 0 }] = i / 32.0; trainY[new[] { i, i % 4 }] = 1.0; }

var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(model)
    .ConfigureLoRA(loraConfig)
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

result.SaveModel("fine-tuned.aimodel");
Console.WriteLine("Saved the fine-tuned model.");
```

## Best Practices

1. **Freeze the base**: keep `freezeBaseLayer: true` — that is what makes LoRA efficient.
2. **Start with `rank: 8`**: raise it only if the model underfits.
3. **Scale alpha with rank**: `alpha` ≈ 2 × `rank` is a good default.
4. **Use QLoRA for large models**: quantize the base with `ConfigureQuantization`.
5. **Watch overfitting**: compare `result.Evaluation.TrainingSet` vs `result.Evaluation.ValidationSet`.

## Notes

The facade exposes **LoRA** and **QLoRA**. Other PEFT methods — DoRA, LoKr/LoHa, VeRA, AdaLoRA, prefix/prompt tuning — and multi-adapter management are not part of the facade today.

## Next Steps

- [LoRA Adapters Reference](/docs/reference/lora-adapters/)
- [Distributed Training Tutorial](/docs/tutorials/distributed-training/)
