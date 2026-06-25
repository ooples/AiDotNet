---
title: "LEOAlgorithm<T, TInput, TOutput>"
description: "Implementation of Latent Embedding Optimization (LEO) meta-learning algorithm."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Latent Embedding Optimization (LEO) meta-learning algorithm.

## For Beginners

Imagine your neural network has millions of parameters.
Updating them all with just 5 examples is risky - you might overfit badly.
LEO learns to "compress" the parameter space into maybe 64 numbers.
When adapting to a new task:

## How It Works

LEO (Latent Embedding Optimization) performs meta-learning by learning a low-dimensional
latent space for model parameters. This enables fast adaptation even for large models
by working in a compressed representation space.

**Key Innovation:** Instead of adapting parameters directly (like MAML), LEO:

This is safer because adjusting 64 numbers can't cause as much overfitting
as adjusting millions of parameters.

**Variational Aspect:** LEO uses a variational autoencoder-like setup where:

- Encoder outputs mean μ and variance σ² of a Gaussian distribution
- Latent code is sampled: z ~ N(μ, σ²)
- KL divergence regularizes z toward a standard Gaussian

This prevents the latent space from collapsing and enables uncertainty estimation.

Reference: Rusu, A. A., Rao, D., Sygnowski, J., et al. (2019).
Meta-Learning with Latent Embedding Optimization. ICLR 2019.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LEOAlgorithm(LEOOptions<,,>)` | Initializes a new instance of the LEOAlgorithm class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` | Gets the algorithm type identifier for this meta-learner. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AccumulateVectors(Vector<>,Vector<>)` | Accumulates vectors element-wise. |
| `Adapt(IMetaLearningTask<,,>)` | Adapts the meta-learned model to a new task using latent space optimization. |
| `AdaptLatentCode(Vector<>,Vector<>,,)` | Adapts the latent code using gradient descent. |
| `ClassifyWithParams(,Vector<>)` | Classifies input using the given classifier parameters. |
| `CloneVector(Vector<>)` | Clones a vector. |
| `ComputeAllGradients(IMetaLearningTask<,,>,Vector<>,Vector<>,Vector<>,Vector<>,)` | Computes gradients for all network components. |
| `ComputeFiniteDiffGradients(Vector<>,Double,Func<>)` | Computes gradients using finite differences. |
| `ComputeKLDivergence(Vector<>,Vector<>)` | Computes KL divergence between latent distribution and standard Gaussian. |
| `ComputeLatentGradients(Vector<>,Vector<>,,)` | Computes gradients with respect to the latent code. |
| `ConvertFromVector(Vector<>)` | Converts a vector to the output type. |
| `DecodeLatent(Vector<>)` | Decodes latent code to classifier parameters. |
| `DivideVector(Vector<>,)` | Divides all elements of a vector by a scalar. |
| `EncodeToLatent(Vector<>)` | Encodes embeddings to latent distribution (mean and variance). |
| `ExtractEmbeddings()` | Extracts embeddings from input using the feature encoder. |
| `InitializeNetworkWeights` | Initializes encoder, decoder weights. |
| `InitializeOrthogonal(Vector<>,Int32)` | Initializes weights with orthogonal initialization (approximation). |
| `InitializeVector(Vector<>,Int32,Double)` | Initializes a vector using Xavier/He initialization. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step using LEO's latent space adaptation approach. |
| `NormalizeBlock(Vector<>,Int32,Int32)` | Normalizes a block of weights to unit norm. |
| `SampleLatent(Vector<>,Vector<>)` | Samples from the latent distribution using reparameterization trick. |
| `SampleStandardNormal` | Samples from standard normal distribution. |

