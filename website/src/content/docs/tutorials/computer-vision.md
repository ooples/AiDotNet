---
title: "Computer Vision"
description: "Build image models with AiDotNet."
order: 4
section: "Tutorials"
---

Train image models through the `AiModelBuilder` facade. Images are `Tensor<T>` batches, and the general-purpose `NeuralNetwork<T>` builds an appropriate network from a `NeuralNetworkArchitecture<T>`.

## Image Classification

Flatten each image into a feature vector (or feed a convolutional architecture) and train like any other model.

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

// 100 synthetic 28x28 grayscale images (784 features), one-hot labels for 10 classes.
var rng = new Random(42);
var trainX = new Tensor<double>(new[] { 100, 784 });
var trainY = new Tensor<double>(new[] { 100, 10 });
for (int i = 0; i < 100; i++)
{
    for (int j = 0; j < 784; j++) trainX[new[] { i, j }] = rng.NextDouble();
    trainY[new[] { i, i % 10 }] = 1.0;
}

var model = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
    inputFeatures: 784, numClasses: 10, complexity: NetworkComplexity.Medium));

var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(model)
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

var scores = result.Predict(trainX);
int predicted = 0;
for (int c = 1; c < 10; c++)
    if (scores[new[] { 0, c }] > scores[new[] { 0, predicted }]) predicted = c;
Console.WriteLine($"Predicted class for image 0: {predicted}");
```

## GPU Acceleration

Vision models are compute-heavy — add `ConfigureGpuAcceleration()` to use the GPU when available.

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

var rng = new Random(1);
var trainX = new Tensor<float>(new[] { 64, 256 });
var trainY = new Tensor<float>(new[] { 64, 5 });
for (int i = 0; i < 64; i++)
{
    for (int j = 0; j < 256; j++) trainX[new[] { i, j }] = (float)rng.NextDouble();
    trainY[new[] { i, i % 5 }] = 1f;
}

var model = new NeuralNetwork<float>(new NeuralNetworkArchitecture<float>(
    inputFeatures: 256, numClasses: 5, complexity: NetworkComplexity.Medium));

var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
    .ConfigureModel(model)
    .ConfigureGpuAcceleration()
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine($"Trained on GPU when available; output [{string.Join(", ", result.Predict(trainX).Shape)}]");
```

## Reading Metrics

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

var rng = new Random(3);
var trainX = new Tensor<double>(new[] { 120, 64 });
var trainY = new Tensor<double>(new[] { 120, 4 });
for (int i = 0; i < 120; i++)
{
    for (int j = 0; j < 64; j++) trainX[new[] { i, j }] = rng.NextDouble();
    trainY[new[] { i, i % 4 }] = 1.0;
}

var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
    .ConfigureModel(new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
        inputFeatures: 64, numClasses: 4, complexity: NetworkComplexity.Simple)))
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine($"Layers: {result.LayerCount}, params: {result.TotalTrainableParameters:N0}");
```

## Best Practices

1. **Normalize pixels**: scale inputs to `[0, 1]` before training.
2. **Use GPU**: add `ConfigureGpuAcceleration()` for image-scale workloads.
3. **Augment data**: more varied training images improve generalization.
4. **Right-size complexity**: start `Medium`, increase only if the model underfits.

## Notes

The facade trains image classifiers built from `NeuralNetwork<T>` / `NeuralNetworkArchitecture<T>`. Task-specific vision pipelines — object detection (Mask R-CNN / YOLO heads), semantic segmentation, and OCR engines — are configured through their own model types rather than a single facade call today.

## Next Steps

- [Neural Network Training](/docs/examples/neural-network-training/)
- [Classification Tutorial](/docs/tutorials/classification/)
