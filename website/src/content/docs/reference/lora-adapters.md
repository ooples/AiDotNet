---
title: "LoRA Adapters"
description: "Parameter-efficient fine-tuning with LoRA."
order: 5
section: "Reference"
---


LoRA (Low-Rank Adaptation) fine-tunes a model by training small rank-`r` matrices on top of frozen base weights, so you adapt large models at a fraction of the parameters and memory. AiDotNet exposes LoRA through the `AiModelBuilder` facade with `ConfigureLoRA(...)`, and QLoRA (quantized base + LoRA) by pairing it with `ConfigureQuantization(...)`.

---

## LoRA Fine-Tuning

Build the base model, attach a LoRA configuration, and `BuildAsync()` wraps the dense layers with adapters and trains only the low-rank matrices.

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.LoRA;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

var model = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
    inputFeatures: 32, numClasses: 4, complexity: NetworkComplexity.Simple));

// rank = adapter capacity; alpha = scaling; freeze the base weights.
var loraConfig = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 16, freezeBaseLayer: true);

var rng = new Random(42);
var trainX = new Tensor<double>(new[] { 32, 32 });
var trainY = new Tensor<double>(new[] { 32, 4 });
for (int i = 0; i < 32; i++)
{
    for (int j = 0; j < 32; j++) trainX[new[] { i, j }] = rng.NextDouble();
    trainY[new[] { i, i % 4 }] = 1.0;
}

var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(model)
    .ConfigureLoRA(loraConfig)
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine($"Fine-tuned with LoRA (rank {loraConfig.Rank}, alpha {loraConfig.Alpha}).");
```

### Configuration

| Parameter | Meaning |
|:----------|:--------|
| `rank` | Rank `r` of the adapter matrices — higher = more capacity, more parameters |
| `alpha` | Scaling factor applied to the adapter output |
| `freezeBaseLayer` | Freeze the base weights and train only the adapters (the LoRA default) |

Typical settings: `rank` 4–16, `alpha` = 2 × `rank`.

---

## QLoRA (Quantized Base + LoRA)

QLoRA shrinks memory by quantizing the frozen base weights, then trains LoRA adapters on top. Add `ConfigureQuantization(...)` alongside `ConfigureLoRA(...)`.

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Deployment.Configuration;
using AiDotNet.Enums;
using AiDotNet.LoRA;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

var model = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
    inputFeatures: 32, numClasses: 4, complexity: NetworkComplexity.Simple));

var loraConfig = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 16, freezeBaseLayer: true);
var quantConfig = new QuantizationConfig { Mode = QuantizationMode.Int8 };

var rng = new Random(7);
var trainX = new Tensor<double>(new[] { 32, 32 });
var trainY = new Tensor<double>(new[] { 32, 4 });
for (int i = 0; i < 32; i++)
{
    for (int j = 0; j < 32; j++) trainX[new[] { i, j }] = rng.NextDouble();
    trainY[new[] { i, i % 4 }] = 1.0;
}

var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(model)
    .ConfigureQuantization(quantConfig)   // quantize the frozen base
    .ConfigureLoRA(loraConfig)            // train the adapters
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Fine-tuned with QLoRA (Int8 base + LoRA adapters).");
```

---

## Best Practices

1. **Freeze the base**: keep `freezeBaseLayer: true` — that is what makes LoRA parameter-efficient.
2. **Start small**: `rank: 8` is a strong default; raise it only if the model underfits.
3. **Scale alpha with rank**: `alpha` ≈ 2 × `rank` is a common starting point.
4. **Use QLoRA for large models**: quantize the base with `ConfigureQuantization` to cut memory.
5. **Validate**: compare `result.Evaluation.TrainingSet` vs `result.Evaluation.ValidationSet` for overfitting.

---

## PEFT Variants (DoRA, LoKr, VeRA, …)

`DefaultLoRAConfiguration` takes an optional `loraAdapter` — pass a specific adapter (DoRA, LoKr, LoHa, VeRA, AdaLoRA, QLoRA, MoRA, PiSSA, …) and `ConfigureLoRA` applies that variant instead of plain LoRA.

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

var model = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
    inputFeatures: 32, numClasses: 4, complexity: NetworkComplexity.Simple));

// DoRA here — swap for LoKrAdapter, LoHaAdapter, VeRAAdapter, AdaLoRAAdapter, etc.
var adapter = new DoRAAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var loraConfig = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);

var trainX = new Tensor<double>(new[] { 32, 32 });
var trainY = new Tensor<double>(new[] { 32, 4 });
for (int i = 0; i < 32; i++) { trainX[new[] { i, 0 }] = i / 32.0; trainY[new[] { i, i % 4 }] = 1.0; }

var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(model)
    .ConfigureLoRA(loraConfig)
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Fine-tuned with a DoRA adapter through ConfigureLoRA.");
```

AiDotNet ships ~25 adapter variants in `AiDotNet.LoRA.Adapters` (DoRA, LoKr, LoHa, VeRA, AdaLoRA, QLoRA, MoRA, PiSSA, DyLoRA, LongLoRA, ReLoRA, GLoRA, and more), plus `ChainLoRAAdapter` / `MultiLoRAAdapter` for composing several.

## Notes

The facade exposes LoRA and QLoRA directly (`ConfigureLoRA`, `ConfigureQuantization`) and every adapter variant via `DefaultLoRAConfiguration`'s `loraAdapter` parameter. Prefix/prompt tuning and runtime adapter hot-swapping are the remaining methods configured outside the single facade call.
