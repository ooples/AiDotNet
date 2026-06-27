---
title: "Deployment"
description: "Quantize, save, and load models for production."
order: 11
section: "Tutorials"
---

Prepare trained models for production: shrink them with quantization, then save and reload them through the `AiModelBuilder` facade and `AiModelResult`.

## Quantization

`ConfigureQuantization(...)` quantizes the model during the build to cut its memory footprint — pass a `QuantizationConfig` with the precision mode.

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Deployment.Configuration;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

var rng = new Random(42);
var trainX = new Tensor<double>(new[] { 64, 32 });
var trainY = new Tensor<double>(new[] { 64, 4 });
for (int i = 0; i < 64; i++)
{
    for (int j = 0; j < 32; j++) trainX[new[] { i, j }] = rng.NextDouble();
    trainY[new[] { i, i % 4 }] = 1.0;
}

var model = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
    inputFeatures: 32, numClasses: 4, complexity: NetworkComplexity.Simple));

var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(model)
    .ConfigureQuantization(new QuantizationConfig { Mode = QuantizationMode.Int8 })
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine("Trained an Int8-quantized model ready for deployment.");
```

## Saving and Loading

Persist a trained model and reload it later, supplying a factory that recreates the model type.

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

double[][] features =
{
    new[] { 1.0, 2.0, 3.0 }, new[] { 4.0, 5.0, 6.0 },
    new[] { 7.0, 8.0, 9.0 }, new[] { 10.0, 11.0, 12.0 }
};
double[] targets = { 10.0, 20.0, 30.0, 40.0 };

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new MultipleRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

// Save for production.
result.SaveModel("model.aimodel");

// Reload it on the serving side.
var loaded = AiModelResult<double, Matrix<double>, Vector<double>>.LoadModel(
    "model.aimodel",
    metadata => new MultipleRegression<double>());

var input = new Matrix<double>(1, 3);
input[0, 0] = 1.0; input[0, 1] = 2.0; input[0, 2] = 3.0;
Console.WriteLine($"Reloaded model prediction: {loaded.Predict(input)[0]:F2}");
```

## Serving Predictions

A reloaded `AiModelResult` predicts exactly like a freshly trained one — wire `result.Predict(...)` behind your API or batch job.

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

double[][] features = { new[] { 1.0, 2.0 }, new[] { 3.0, 4.0 }, new[] { 5.0, 6.0 } };
double[] targets = { 3.0, 7.0, 11.0 };

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new MultipleRegression<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

// Batch-score requests by stacking them into one matrix.
var batch = new Matrix<double>(2, 2);
batch[0, 0] = 2.0; batch[0, 1] = 3.0;
batch[1, 0] = 4.0; batch[1, 1] = 5.0;

var predictions = result.Predict(batch);
for (int i = 0; i < predictions.Length; i++)
    Console.WriteLine($"Request {i}: {predictions[i]:F2}");
```

## Best Practices

1. **Quantize for size**: `ConfigureQuantization` with `QuantizationMode.Int8` shrinks the model.
2. **Validate after quantizing**: compare metrics before/after to confirm acceptable accuracy.
3. **Version your model files**: keep the model type/factory in sync with the saved file.
4. **Batch at serving time**: stack requests into one matrix/tensor for throughput.

## Notes

The facade covers post-training quantization (`ConfigureQuantization`) plus `SaveModel` / `LoadModel`. Quantization-aware training (QAT) wrappers and a built-in serving runtime are configured outside the single facade call today.

## Next Steps

- [LLM Fine-tuning Tutorial](/docs/tutorials/llm-fine-tuning/) — QLoRA quantization
- [Quick Start](/docs/getting-started/quickstart/)
