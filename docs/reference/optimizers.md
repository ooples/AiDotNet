---
layout: default
title: Optimizers
parent: Reference
nav_order: 3
permalink: /reference/optimizers/
---

# Optimizers
{: .no_toc }

Complete reference for all 42+ optimization algorithms in AiDotNet.
{: .fs-6 .fw-300 }

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## First-Order Optimizers

### SGD Family

| Optimizer | Description | Use Case |
|:----------|:------------|:---------|
| `SGDOptimizer<T>` | Stochastic Gradient Descent | General purpose |
| `MomentumOptimizer<T>` | SGD with momentum | Faster convergence |
| `NesterovOptimizer<T>` | Nesterov Accelerated Gradient | Look-ahead momentum |

```csharp
var optimizer = new SGDOptimizer<float>(
    learningRate: 0.01f,
    momentum: 0.9f,
    weightDecay: 1e-4f,
    nesterov: true);
```

### Adam Family

| Optimizer | Description | Use Case |
|:----------|:------------|:---------|
| `AdamOptimizer<T>` | Adaptive Moment Estimation | General deep learning |
| `AdamWOptimizer<T>` | Adam with decoupled weight decay | Transformers, large models |
| `AdaMaxOptimizer<T>` | Adam with infinity norm | Sparse gradients |
| `NAdamOptimizer<T>` | Nesterov Adam | Improved convergence |
| `RAdam<T>` | Rectified Adam | Stable training |
| `AdamP<T>` | Adam with projection | Vision models |

```csharp
var optimizer = new AdamWOptimizer<float>(
    learningRate: 3e-4f,
    beta1: 0.9f,
    beta2: 0.999f,
    epsilon: 1e-8f,
    weightDecay: 0.01f);
```

### Adaptive Learning Rate

| Optimizer | Description | Use Case |
|:----------|:------------|:---------|
| `AdaGradOptimizer<T>` | Adaptive gradient | Sparse features |
| `AdaDeltaOptimizer<T>` | Adaptive delta | No learning rate tuning |
| `RMSpropOptimizer<T>` | Root Mean Square propagation | RNNs |
| `AdaFactorOptimizer<T>` | Memory-efficient adaptive | Large models |

### LAMB/LARS

| Optimizer | Description | Use Case |
|:----------|:------------|:---------|
| `LARSOptimizer<T>` | Layer-wise Adaptive Rate Scaling | Large batch training |
| `LAMBOptimizer<T>` | Layer-wise Adaptive Moments | BERT pre-training |

```csharp
var optimizer = new LAMBOptimizer<float>(
    learningRate: 0.001f,
    beta1: 0.9f,
    beta2: 0.999f,
    trustCoeff: 0.001f);
```

### Modern Optimizers

| Optimizer | Description | Use Case |
|:----------|:------------|:---------|
| `LionOptimizer<T>` | Evolved Sign Momentum | Vision, language models |
| `Prodigy<T>` | Automatic learning rate | No tuning needed |
| `ScheduleFree<T>` | No schedule needed | Simplified training |
| `Sophia<T>` | Second-order information | LLM training |
| `Muon<T>` | Momentum-based | Research |

```csharp
var optimizer = new LionOptimizer<float>(
    learningRate: 1e-4f,
    beta1: 0.9f,
    beta2: 0.99f,
    weightDecay: 0.0f);
```

### Memory-Bounded Streaming Optimizers

When `NeuralNetworkBase.StreamingTrainingEnabled` is enabled, the streaming training path now resolves configured first-order optimizers to optimizer-in-backward variants with 8-bit block-quantized state. This keeps gradients and moment buffers bounded while preserving the configured optimizer family and learning-rate scheduler handoff.

| Configured optimizer | Streaming variant | Quantized state |
|:---------------------|:------------------|:----------------|
| Gradient descent, SGD, mini-batch GD, proximal GD | `StreamingSgd8Bit<T>` | none |
| Momentum, Nesterov | `StreamingMomentum8Bit<T>`, `StreamingNesterov8Bit<T>` | 1 signed buffer |
| RMSprop, Adagrad | `StreamingRmsProp8Bit<T>`, `StreamingAdagrad8Bit<T>` | 1 unsigned buffer |
| AdaDelta, FTRL | `StreamingAdaDelta8Bit<T>`, `StreamingFtrl8Bit<T>` | 2 unsigned buffers |
| Adam, AdamW, Nadam, AdaMax, Lion, LAMB, LARS | matching `Streaming*8Bit<T>` variant | 1-2 buffers |
| AMSGrad or Adam with `UseAMSGrad = true` | `StreamingAMSGrad8Bit<T>` | 3 buffers |

Each state buffer stores one byte per parameter plus one double scale per block, instead of one full-precision numeric value per parameter. With the default 2048-parameter block size, one-buffer optimizers use about 1 byte per parameter plus scale metadata, two-buffer Adam-style optimizers use about 2 bytes per parameter plus metadata, and AMSGrad uses about 3 bytes per parameter plus metadata. Compared with full `double` moment state, this is roughly an 8x reduction for the moment buffers; compared with full `float` moment state, it is roughly a 4x reduction.

Within a single parameter, the per-block update (dequantize moments → apply the rule → requantize) runs in parallel across blocks: each block owns a disjoint parameter/gradient/state slice, so each worker gets its own requantization scratch and the blocks update concurrently with no shared mutable state. The parallel path is gated on parameter size (small parameters such as biases and norm scales stay on the serial path, where thread-dispatch overhead would not pay off) and is deterministic — block `b`'s output depends only on its own slice, so results are identical regardless of how blocks are scheduled.

#### Second-order streaming: L-BFGS

The first-order variants update each parameter in place as its gradient is produced. L-BFGS is a **second-order** method — its search direction depends on the whole gradient plus a curvature history — so it can't update per-parameter-as-gradient-arrives. `StreamingLBFGS<T>` instead buffers the per-parameter gradients into flat vectors as they stream in, then performs the two-loop recursion and parameter update once the full gradient is available.

| Configured optimizer | Streaming variant | Quantized state |
|:---------------------|:------------------|:----------------|
| `LBFGSOptimizer` | `StreamingLBFGS<T>` | `MemorySize` (s, y) history vectors, 8-bit block-quantized |

The dominant memory cost of L-BFGS is its `(s, y)` curvature history — `MemorySize` vector pairs of length *n*, i.e. O(*m·n*). `StreamingLBFGS` stores that history 8-bit block-quantized (~8x smaller than `double`) and dequantizes it **one vector at a time** during the two-loop recursion, so the transient working set stays O(*n*) rather than O(*m·n*). Full BFGS / Newton are intentionally not provided as streaming variants: their O(*n²*) dense Hessian (approximation) does not fit memory-bounded large-model training at all.

---

## Second-Order Optimizers

| Optimizer | Description | Use Case |
|:----------|:------------|:---------|
| `LBFGSOptimizer<T>` | Limited-memory BFGS | Small models, full batch |
| `NewtonOptimizer<T>` | Newton's method | Convex optimization |
| `KFACOptimizer<T>` | Kronecker-factored curvature | Deep networks |
| `ShampooOptimizer<T>` | Preconditioning | Large-scale training |

```csharp
var optimizer = new LBFGSOptimizer<double>(
    maxIterations: 20,
    historySize: 10,
    lineSearch: LineSearch.StrongWolfe);
```

---

## Sparse Optimizers

| Optimizer | Description | Use Case |
|:----------|:------------|:---------|
| `SparseAdamOptimizer<T>` | Adam for sparse gradients | Embeddings |
| `LazyAdamOptimizer<T>` | Lazy parameter updates | Large sparse models |

---

## Evolutionary Optimizers

| Optimizer | Description | Use Case |
|:----------|:------------|:---------|
| `GeneticOptimizer<T>` | Genetic Algorithm | Hyperparameter search |
| `EvolutionStrategy<T>` | Evolution strategies | Neural architecture |
| `CMAESOptimizer<T>` | Covariance Matrix Adaptation | Black-box optimization |
| `ParticleSwarmOptimizer<T>` | Particle Swarm | Global optimization |
| `DifferentialEvolution<T>` | Differential evolution | Continuous optimization |

```csharp
var optimizer = new GeneticOptimizer<double>(
    populationSize: 100,
    mutationRate: 0.1,
    crossoverRate: 0.8,
    elitismCount: 5);
```

---

## Learning Rate Schedulers

### Step-Based

| Scheduler | Description |
|:----------|:------------|
| `StepLR` | Decay by factor at fixed intervals |
| `MultiStepLR` | Decay at specified milestones |
| `ExponentialLR` | Exponential decay |

```csharp
var scheduler = new StepLR(
    optimizer: optimizer,
    stepSize: 30,
    gamma: 0.1f);
```

### Epoch-Based

| Scheduler | Description |
|:----------|:------------|
| `CosineAnnealingLR` | Cosine annealing |
| `CosineAnnealingWarmRestarts` | Cosine with restarts |
| `LinearLR` | Linear decay |
| `PolynomialLR` | Polynomial decay |

```csharp
var scheduler = new CosineAnnealingLR(
    optimizer: optimizer,
    tMax: 100,
    etaMin: 1e-6f);
```

### Warmup Schedulers

| Scheduler | Description |
|:----------|:------------|
| `WarmupLinearSchedule` | Linear warmup |
| `WarmupCosineSchedule` | Warmup + cosine decay |
| `WarmupConstantSchedule` | Warmup + constant |

```csharp
var scheduler = new WarmupCosineSchedule(
    optimizer: optimizer,
    warmupSteps: 1000,
    totalSteps: 100000);
```

### Adaptive Schedulers

| Scheduler | Description |
|:----------|:------------|
| `ReduceLROnPlateau` | Reduce when metric plateaus |
| `CyclicLR` | Cyclic learning rate |
| `OneCycleLR` | One cycle policy |

```csharp
var scheduler = new ReduceLROnPlateau(
    optimizer: optimizer,
    mode: "min",
    factor: 0.1f,
    patience: 10);
```

---

## Usage Examples

### Basic Training Loop

```csharp
var optimizer = new AdamWOptimizer<float>(learningRate: 3e-4f);
var scheduler = new CosineAnnealingLR(optimizer, tMax: epochs);

for (int epoch = 0; epoch < epochs; epoch++)
{
    foreach (var batch in dataLoader)
    {
        optimizer.ZeroGrad();
        var loss = model.Forward(batch.Input);
        loss.Backward();
        optimizer.Step();
    }
    scheduler.Step();
}
```

### With AiModelBuilder

```csharp
var result = await new AiModelBuilder<float, Tensor<float>, int>()
    .ConfigureModel(model)
    .ConfigureOptimizer(new AdamWOptimizer<float>(learningRate: 3e-4f))
    .ConfigureLearningRateScheduler(new CosineAnnealingLR(tMax: 100))
    .BuildAsync(trainData, trainLabels);
```

### Gradient Clipping

```csharp
var optimizer = new AdamWOptimizer<float>(
    learningRate: 3e-4f,
    maxGradNorm: 1.0f);  // Gradient clipping
```

---

## Optimizer Selection Guide

| Task | Recommended Optimizer |
|:-----|:---------------------|
| General deep learning | AdamW |
| Transformers/LLMs | AdamW + cosine schedule |
| Large batch training | LAMB |
| Vision models | SGD + momentum or Lion |
| RNNs | RMSprop or Adam |
| Small datasets | L-BFGS |
| Hyperparameter search | Genetic/PSO |
| Memory-constrained | AdaFactor |

---

## Hyperparameter Guidelines

| Parameter | Typical Range | Notes |
|:----------|:--------------|:------|
| Learning rate | 1e-5 to 1e-2 | Start with 3e-4 for Adam |
| Weight decay | 0 to 0.1 | 0.01 is common |
| Beta1 (momentum) | 0.9 to 0.95 | 0.9 is standard |
| Beta2 | 0.99 to 0.999 | 0.999 for Adam |
| Epsilon | 1e-8 to 1e-6 | 1e-8 is standard |
