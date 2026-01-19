---
layout: default
title: Loss Functions
parent: Reference
nav_order: 4
permalink: /reference/loss-functions/
---

# Loss Functions
{: .no_toc }

Complete reference for all 37+ loss functions in AiDotNet.
{: .fs-6 .fw-300 }

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Classification Losses

### Binary Classification

| Loss Function | Formula | Use Case |
|:--------------|:--------|:---------|
| `BinaryCrossEntropyLoss<T>` | -[y·log(p) + (1-y)·log(1-p)] | Binary classification |
| `BCEWithLogitsLoss<T>` | BCELoss with sigmoid | Numerically stable |
| `HingeLoss<T>` | max(0, 1 - y·p) | SVM-style |
| `SquaredHingeLoss<T>` | (max(0, 1 - y·p))² | Smooth hinge |

```csharp
var loss = new BinaryCrossEntropyLoss<float>();
var value = loss.Compute(predictions, targets);
```

### Multi-Class Classification

| Loss Function | Description | Use Case |
|:--------------|:------------|:---------|
| `CrossEntropyLoss<T>` | Standard cross entropy | Multi-class |
| `NLLLoss<T>` | Negative log likelihood | After log_softmax |
| `SoftmaxCrossEntropyLoss<T>` | Softmax + cross entropy | Combined operation |

```csharp
var loss = new CrossEntropyLoss<float>(
    labelSmoothing: 0.1f,
    ignoreIndex: -100);
```

### Focal Loss (Imbalanced Data)

| Loss Function | Description | Use Case |
|:--------------|:------------|:---------|
| `FocalLoss<T>` | Down-weights easy examples | Class imbalance |
| `BinaryFocalLoss<T>` | Focal loss for binary | Highly imbalanced |

```csharp
var loss = new FocalLoss<float>(
    gamma: 2.0f,  // Focusing parameter
    alpha: 0.25f); // Class weight
```

---

## Regression Losses

### Standard Losses

| Loss Function | Formula | Use Case |
|:--------------|:--------|:---------|
| `MSELoss<T>` | (y - p)² | General regression |
| `MAELoss<T>` | \|y - p\| | Robust to outliers |
| `HuberLoss<T>` | Quadratic near 0, linear elsewhere | Balanced |
| `SmoothL1Loss<T>` | Huber with delta=1 | Object detection |
| `LogCoshLoss<T>` | log(cosh(y - p)) | Smooth approximation |

```csharp
var loss = new HuberLoss<float>(delta: 1.0f);
```

### Quantile Losses

| Loss Function | Description | Use Case |
|:--------------|:------------|:---------|
| `QuantileLoss<T>` | Quantile regression | Prediction intervals |
| `PinballLoss<T>` | Same as quantile | Probabilistic forecasting |

```csharp
var loss = new QuantileLoss<float>(quantile: 0.9f);
```

---

## Segmentation Losses

| Loss Function | Description | Use Case |
|:--------------|:------------|:---------|
| `DiceLoss<T>` | 1 - Dice coefficient | Segmentation |
| `IoULoss<T>` | 1 - Intersection over Union | Object detection |
| `TverskyLoss<T>` | Generalized Dice | Imbalanced segmentation |
| `FocalTverskyLoss<T>` | Focal + Tversky | Hard pixels |
| `LovaszHingeLoss<T>` | Lovasz extension | IoU optimization |
| `BoundaryLoss<T>` | Distance to boundary | Edge-aware |

```csharp
var loss = new DiceLoss<float>(smooth: 1e-6f);
```

---

## Contrastive/Metric Learning

| Loss Function | Description | Use Case |
|:--------------|:------------|:---------|
| `ContrastiveLoss<T>` | Siamese networks | Similarity learning |
| `TripletLoss<T>` | Anchor, positive, negative | Face recognition |
| `TripletMarginLoss<T>` | Triplet with margin | Embedding learning |
| `NTXentLoss<T>` | Normalized temperature | Self-supervised |
| `InfoNCELoss<T>` | Information NCE | Contrastive learning |
| `SupConLoss<T>` | Supervised contrastive | With labels |

```csharp
var loss = new TripletLoss<float>(margin: 1.0f);
```

---

## Reconstruction Losses

| Loss Function | Description | Use Case |
|:--------------|:------------|:---------|
| `ReconstructionLoss<T>` | MSE for reconstruction | Autoencoders |
| `VAELoss<T>` | Reconstruction + KL divergence | VAE training |
| `BetaVAELoss<T>` | VAE with beta | Disentanglement |
| `SSIMLoss<T>` | 1 - SSIM | Image quality |
| `PerceptualLoss<T>` | Feature space distance | Super-resolution |

```csharp
var loss = new VAELoss<float>(
    reconstructionWeight: 1.0f,
    klWeight: 0.001f);
```

---

## GAN Losses

| Loss Function | Description | Use Case |
|:--------------|:------------|:---------|
| `AdversarialLoss<T>` | Standard GAN loss | Basic GANs |
| `WassersteinLoss<T>` | Wasserstein distance | WGAN |
| `HingeLoss<T>` | GAN hinge loss | Spectral normalization |
| `LSGANLoss<T>` | Least squares GAN | Stable training |
| `NonSaturatingLoss<T>` | Non-saturating | Better gradients |

```csharp
var generatorLoss = new NonSaturatingLoss<float>(mode: LossMode.Generator);
var discriminatorLoss = new NonSaturatingLoss<float>(mode: LossMode.Discriminator);
```

---

## Ranking Losses

| Loss Function | Description | Use Case |
|:--------------|:------------|:---------|
| `MarginRankingLoss<T>` | Pairwise ranking | Learning to rank |
| `MultiMarginLoss<T>` | Multi-class margin | Multi-class ranking |
| `ListwiseLoss<T>` | Full list ranking | Search ranking |

```csharp
var loss = new MarginRankingLoss<float>(margin: 1.0f);
```

---

## Regularization Losses

| Loss Function | Description | Use Case |
|:--------------|:------------|:---------|
| `L1Loss<T>` | L1 regularization | Sparsity |
| `L2Loss<T>` | L2 regularization | Weight decay |
| `ElasticNetLoss<T>` | L1 + L2 | Combined |
| `KLDivLoss<T>` | KL divergence | Distribution matching |
| `JSDivLoss<T>` | Jensen-Shannon | Symmetric KL |

---

## Sequence Losses

| Loss Function | Description | Use Case |
|:--------------|:------------|:---------|
| `CTCLoss<T>` | Connectionist Temporal | Speech recognition |
| `SequenceLoss<T>` | Padded sequences | Seq2seq |
| `LabelSmoothingLoss<T>` | Smoothed targets | NLP models |

```csharp
var loss = new CTCLoss<float>(blank: 0, zeroInfinity: true);
```

---

## Combined Losses

```csharp
// Combine multiple losses
var loss = new CombinedLoss<float>(
    new (ILoss<float>, float)[]
    {
        (new CrossEntropyLoss<float>(), 1.0f),
        (new DiceLoss<float>(), 0.5f),
        (new FocalLoss<float>(), 0.3f)
    });
```

---

## Usage Examples

### With AiModelBuilder

```csharp
var result = await new AiModelBuilder<float, Tensor<float>, int>()
    .ConfigureModel(model)
    .ConfigureLossFunction(new FocalLoss<float>(gamma: 2.0f))
    .BuildAsync(trainData, trainLabels);
```

### Custom Training Loop

```csharp
var loss = new CrossEntropyLoss<float>(labelSmoothing: 0.1f);

foreach (var batch in dataLoader)
{
    var predictions = model.Forward(batch.Input);
    var lossValue = loss.Compute(predictions, batch.Target);
    lossValue.Backward();
    optimizer.Step();
}
```

---

## Loss Selection Guide

| Task | Recommended Loss |
|:-----|:-----------------|
| Binary classification | BCEWithLogitsLoss |
| Multi-class classification | CrossEntropyLoss |
| Imbalanced classification | FocalLoss |
| Regression | MSELoss or HuberLoss |
| Segmentation | DiceLoss + CrossEntropyLoss |
| Object detection | SmoothL1Loss + FocalLoss |
| Face recognition | TripletLoss |
| Autoencoders | ReconstructionLoss |
| VAE | VAELoss |
| GANs | AdversarialLoss or WassersteinLoss |
| Speech recognition | CTCLoss |
