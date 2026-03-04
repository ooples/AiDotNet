---
title: "Loss Functions"
description: "Loss function reference."
order: 4
section: "Reference"
---



Complete reference for all 37 loss functions in AiDotNet.

---

## Classification Losses

### Binary Classification

| Loss Function | Description | Use Case |
|:--------------|:------------|:---------|
| `BinaryCrossEntropyLoss<T>` | -[y·log(p) + (1-y)·log(1-p)] | Binary classification |
| `HingeLoss<T>` | max(0, 1 - y·p) | SVM-style binary classification |
| `SquaredHingeLoss<T>` | (max(0, 1 - y·p))^2 | Smooth differentiable hinge |
| `ModifiedHuberLoss<T>` | Modified hinge + quadratic | Robust binary |

```csharp
var loss = new BinaryCrossEntropyLoss<float>();
var value = loss.CalculateLoss(predictions, targets);
```

### Multi-Class Classification

| Loss Function | Description | Use Case |
|:--------------|:------------|:---------|
| `CrossEntropyLoss<T>` | Standard cross entropy | Multi-class |
| `CategoricalCrossEntropyLoss<T>` | One-hot encoded cross entropy | Multi-class with one-hot |
| `SparseCategoricalCrossEntropyLoss<T>` | Integer label cross entropy | Multi-class with indices |
| `WeightedCrossEntropyLoss<T>` | Class-weighted cross entropy | Imbalanced classes |

```csharp
var loss = new CrossEntropyLoss<float>();
```

### Focal Loss (Imbalanced Data)

| Loss Function | Description | Use Case |
|:--------------|:------------|:---------|
| `FocalLoss<T>` | Down-weights easy examples | Class imbalance |

```csharp
var loss = new FocalLoss<float>();
```

---

## Regression Losses

| Loss Function | Description | Use Case |
|:--------------|:------------|:---------|
| `MeanSquaredErrorLoss<T>` | Mean squared error: (y - p)^2 | General regression |
| `MeanAbsoluteErrorLoss<T>` | Mean absolute error | Robust to outliers |
| `RootMeanSquaredErrorLoss<T>` | Square root of MSE | Same scale as target |
| `MeanBiasErrorLoss<T>` | Mean bias error | Directional bias |
| `HuberLoss<T>` | Quadratic near 0, linear elsewhere | Balanced robustness |
| `LogCoshLoss<T>` | log(cosh(y - p)) | Smooth L1 approximation |
| `CharbonnierLoss<T>` | Differentiable L1 | Image restoration |
| `PoissonLoss<T>` | Poisson negative log likelihood | Count data |
| `QuantileLoss<T>` | Quantile regression | Prediction intervals |

```csharp
var loss = new HuberLoss<float>();
```

---

## Segmentation Losses

| Loss Function | Description | Use Case |
|:--------------|:------------|:---------|
| `DiceLoss<T>` | 1 - Dice coefficient | Segmentation |
| `JaccardLoss<T>` | 1 - Intersection over Union | Object detection |
| `ScaleInvariantDepthLoss<T>` | Scale-invariant depth | Depth estimation |

```csharp
var loss = new DiceLoss<float>();
```

---

## Contrastive/Metric Learning

| Loss Function | Description | Use Case |
|:--------------|:------------|:---------|
| `ContrastiveLoss<T>` | Siamese networks | Similarity learning |
| `TripletLoss<T>` | Anchor, positive, negative | Face recognition |
| `NTXentLoss<T>` | Normalized temperature cross-entropy | Self-supervised |
| `InfoNCELoss<T>` | Information NCE | Contrastive learning |
| `NoiseContrastiveEstimationLoss<T>` | NCE for large vocabularies | Word embeddings |
| `CosineSimilarityLoss<T>` | 1 - cosine similarity | Embedding alignment |

```csharp
var loss = new TripletLoss<float>();
```

---

## Reconstruction Losses

| Loss Function | Description | Use Case |
|:--------------|:------------|:---------|
| `MAEReconstructionLoss<T>` | MAE for reconstruction | Masked autoencoders |
| `PerceptualLoss<T>` | Feature space distance | Super-resolution |
| `RealESRGANLoss<T>` | Combined perceptual + adversarial | Image enhancement |
| `RotationPredictionLoss<T>` | Rotation prediction | Self-supervised pretext |

```csharp
var loss = new PerceptualLoss<float>();
```

---

## GAN Losses

| Loss Function | Description | Use Case |
|:--------------|:------------|:---------|
| `WassersteinLoss<T>` | Wasserstein distance | WGAN |
| `MarginLoss<T>` | Margin-based loss | Pairwise ranking |

```csharp
var loss = new WassersteinLoss<float>();
```

---

## Other Losses

| Loss Function | Description | Use Case |
|:--------------|:------------|:---------|
| `ElasticNetLoss<T>` | L1 + L2 combined | Regularized regression |
| `ExponentialLoss<T>` | Exponential loss | AdaBoost |
| `OrdinalRegressionLoss<T>` | Ordered categories | Ordinal prediction |
| `QuantumLoss<T>` | Quantum-inspired loss | Quantum ML |

---

## Sequence Losses

| Loss Function | Description | Use Case |
|:--------------|:------------|:---------|
| `CTCLoss<T>` | Connectionist Temporal Classification | Speech recognition |

```csharp
var loss = new CTCLoss<float>();
```

---

## Self-Supervised Learning Losses

| Loss Function | Description | Use Case |
|:--------------|:------------|:---------|
| `BYOLLoss<T>` | Bootstrap Your Own Latent | Self-supervised |
| `BarlowTwinsLoss<T>` | Redundancy reduction | Self-supervised |
| `DINOLoss<T>` | Self-distillation | Vision transformers |

---

## Usage Examples

### With AiModelBuilder

```csharp
var result = await new AiModelBuilder<float, float[][], float[]>()
    .ConfigureModel(model)
    .ConfigureLossFunction(new FocalLoss<float>())
    .BuildAsync(trainData, trainLabels);
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
