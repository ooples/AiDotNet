---
title: "InceptionScore<T>"
description: "InceptionScore<T> — Models & Types in AiDotNet.Metrics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Metrics`

_No summary documentation available yet._

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InceptionScore(ConvolutionalNeuralNetwork<>,Int32,Int32)` | Initializes a new instance of Inception Score calculator. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InceptionNetwork` | Gets the Inception network used for classification. |
| `NumClasses` | Gets the number of classes in the classifier. |
| `NumSplits` | Gets or sets the number of splits for computing IS with uncertainty. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeComprehensiveMetrics(Tensor<>,Tensor<>)` | Computes both IS and FID if a FID calculator is provided. |
| `ComputeIS(Tensor<>)` | Computes the Inception Score for a set of generated images. |
| `ComputeISForSplit(Tensor<>)` | Computes Inception Score for a single split of images. |
| `ComputeISWithUncertainty(Tensor<>)` | Computes Inception Score with mean and standard deviation across splits. |
| `ComputeKLDivergence(Matrix<>,Int32,Vector<>)` | Computes KL divergence between conditional p(y\|x) and marginal p(y). |
| `ComputeMarginalDistribution(Matrix<>)` | Computes the marginal distribution p(y) by averaging over all images. |
| `CreateDummyPredictions(Int32)` | Creates dummy predictions for testing when no Inception network is available. |
| `ExtractImageSubset(Tensor<>,Int32,Int32)` | Extracts a subset of images from the tensor. |
| `GetPredictions(Tensor<>)` | Gets class probability predictions for all images using Inception network. |
| `Softmax(Tensor<>)` | Applies softmax activation to convert logits to probabilities. |

