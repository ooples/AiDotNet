---
title: "Loss Functions"
description: "Loss function reference."
order: 4
section: "Reference"
---


Reference for the loss functions in AiDotNet. They live in `AiDotNet.LossFunctions`, implement `ILossFunction<T>` (`CalculateLoss` / `CalculateDerivative`), and plug into training via `ConfigureLossFunction(...)`.

---

## Classification Losses

| Loss Function | Use Case |
|:--------------|:---------|
| `BinaryCrossEntropyLoss<T>` | Binary classification |
| `HingeLoss<T>` | SVM-style binary classification |
| `SquaredHingeLoss<T>` | Smooth differentiable hinge |
| `ModifiedHuberLoss<T>` | Robust binary |
| `CrossEntropyLoss<T>` | Multi-class |
| `CategoricalCrossEntropyLoss<T>` | Multi-class with one-hot |
| `SparseCategoricalCrossEntropyLoss<T>` | Multi-class with integer labels |
| `WeightedCrossEntropyLoss<T>` | Imbalanced classes |
| `FocalLoss<T>` | Class imbalance (down-weights easy examples) |

---

## Regression Losses

| Loss Function | Use Case |
|:--------------|:---------|
| `MeanSquaredErrorLoss<T>` | General regression |
| `MeanAbsoluteErrorLoss<T>` | Robust to outliers |
| `RootMeanSquaredErrorLoss<T>` | Same scale as target |
| `MeanBiasErrorLoss<T>` | Directional bias |
| `HuberLoss<T>` | Balanced robustness |
| `LogCoshLoss<T>` | Smooth L1 approximation |
| `CharbonnierLoss<T>` | Image restoration |
| `PoissonLoss<T>` | Count data |
| `QuantileLoss<T>` | Prediction intervals |

---

## Segmentation & Detection

| Loss Function | Use Case |
|:--------------|:---------|
| `DiceLoss<T>` | Segmentation |
| `JaccardLoss<T>` | Object detection (IoU) |
| `ScaleInvariantDepthLoss<T>` | Depth estimation |

---

## Contrastive / Metric Learning

| Loss Function | Use Case |
|:--------------|:---------|
| `ContrastiveLoss<T>` | Similarity learning |
| `TripletLoss<T>` | Face recognition |
| `NTXentLoss<T>` | Self-supervised |
| `InfoNCELoss<T>` | Contrastive learning |
| `CosineSimilarityLoss<T>` | Embedding alignment |

---

## Reconstruction & GAN

| Loss Function | Use Case |
|:--------------|:---------|
| `PerceptualLoss<T>` | Super-resolution |
| `MAEReconstructionLoss<T>` | Masked autoencoders |
| `WassersteinLoss<T>` | WGAN |
| `MarginLoss<T>` | Pairwise ranking |

---

## Sequence & Self-Supervised

| Loss Function | Use Case |
|:--------------|:---------|
| `CTCLoss<T>` | Speech recognition |
| `BYOLLoss<T>` | Self-supervised |
| `BarlowTwinsLoss<T>` | Self-supervised |
| `DINOLoss<T>` | Vision transformers |

---

## Creating a Loss Function

```csharp
using AiDotNet.LossFunctions;

var mse = new MeanSquaredErrorLoss<float>();
var crossEntropy = new CrossEntropyLoss<float>();
var focal = new FocalLoss<float>();
var huber = new HuberLoss<float>();
var dice = new DiceLoss<float>();
var triplet = new TripletLoss<float>();
var wasserstein = new WassersteinLoss<float>();
var ctc = new CTCLoss<float>();
```

## Evaluating a Loss

`CalculateLoss(predicted, actual)` returns the scalar loss; `CalculateDerivative(...)` returns the gradient.

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new MeanSquaredErrorLoss<float>();

var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
var gradient = loss.CalculateDerivative(predicted, actual);
Console.WriteLine($"Loss: {value:F4}, gradient length: {gradient.Length}");
```

## Using a Loss with AiModelBuilder

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

var rng = new Random(0);
var trainX = new Tensor<float>(new[] { 64, 16 });
var trainY = new Tensor<float>(new[] { 64, 3 });
for (int i = 0; i < 64; i++)
{
    for (int j = 0; j < 16; j++) trainX[new[] { i, j }] = (float)rng.NextDouble();
    trainY[new[] { i, i % 3 }] = 1f;
}

var model = new NeuralNetwork<float>(new NeuralNetworkArchitecture<float>(
    inputFeatures: 16, numClasses: 3, complexity: NetworkComplexity.Simple));

var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
    .ConfigureModel(model)
    .ConfigureLossFunction(new FocalLoss<float>())
    .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
    .BuildAsync();

Console.WriteLine($"Trained with Focal loss; output [{string.Join(", ", result.Predict(trainX).Shape)}]");
```

---

## Loss Selection Guide

| Task | Recommended Loss |
|:-----|:-----------------|
| Binary classification | `BinaryCrossEntropyLoss` |
| Multi-class classification | `CrossEntropyLoss` |
| Imbalanced classification | `FocalLoss` |
| Regression | `MeanSquaredErrorLoss` or `HuberLoss` |
| Segmentation | `DiceLoss` + `CrossEntropyLoss` |
| Object detection | `JaccardLoss` |
| Face recognition | `TripletLoss` |
| Contrastive learning | `NTXentLoss` or `InfoNCELoss` |
| GANs | `WassersteinLoss` |
| Speech recognition | `CTCLoss` |
| Self-supervised | `BYOLLoss` or `DINOLoss` |
