---
title: "Optimizers"
description: "Optimizer reference for training neural networks."
order: 3
section: "Reference"
---



Complete reference for the 40+ optimization algorithms and learning rate schedulers in AiDotNet.

---

## First-Order Optimizers

### SGD Family

| Optimizer | Description | Use Case |
|:----------|:------------|:---------|
| `StochasticGradientDescentOptimizer<T>` | Stochastic Gradient Descent | General purpose |
| `MomentumOptimizer<T>` | SGD with momentum | Faster convergence |
| `NesterovAcceleratedGradientOptimizer<T>` | Nesterov Accelerated Gradient | Look-ahead momentum |
| `MiniBatchGradientDescentOptimizer<T>` | Mini-batch SGD | Large datasets |

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
| `NadamOptimizer<T>` | Nesterov Adam | Improved convergence |
| `AMSGradOptimizer<T>` | AMSGrad variant | Stable training |
| `PrototypeAdamOptimizer<T>` | Prototype Adam | Experimental |

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
| `AdagradOptimizer<T>` | Adaptive gradient | Sparse features |
| `AdaDeltaOptimizer<T>` | Adaptive delta | No learning rate tuning |
| `RootMeanSquarePropagationOptimizer<T>` | Root Mean Square propagation | RNNs |
| `FTRLOptimizer<T>` | Follow-the-Regularized-Leader | Online learning |

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
| `ADMMOptimizer<T>` | Alternating Direction Method of Multipliers | Constrained optimization |
| `ConjugateGradientOptimizer<T>` | Conjugate gradient | Large sparse systems |
| `TrustRegionOptimizer<T>` | Trust region method | Robust convergence |

```csharp
var optimizer = new LionOptimizer<float>(
    learningRate: 1e-4f,
    beta1: 0.9f,
    beta2: 0.99f,
    weightDecay: 0.0f);
```

---

## Second-Order Optimizers

| Optimizer | Description | Use Case |
|:----------|:------------|:---------|
| `LBFGSOptimizer<T>` | Limited-memory BFGS | Small models, full batch |
| `BFGSOptimizer<T>` | Broyden-Fletcher-Goldfarb-Shanno | Convex optimization |
| `NewtonMethodOptimizer<T>` | Newton's method | Quadratic convergence |
| `LevenbergMarquardtOptimizer<T>` | Levenberg-Marquardt | Nonlinear least squares |
| `DFPOptimizer<T>` | Davidon-Fletcher-Powell | Quasi-Newton method |

```csharp
var optimizer = new LBFGSOptimizer<double>(
    maxIterations: 20,
    historySize: 10,
    lineSearch: LineSearch.StrongWolfe);
```

---

## Gradient Descent Variants

| Optimizer | Description | Use Case |
|:----------|:------------|:---------|
| `GradientDescentOptimizer<T>` | Standard gradient descent | Simple problems |
| `ProximalGradientDescentOptimizer<T>` | Proximal gradient | L1 regularization |
| `CoordinateDescentOptimizer<T>` | Coordinate descent | Feature selection |

---

## Evolutionary Optimizers

| Optimizer | Description | Use Case |
|:----------|:------------|:---------|
| `GeneticOptimizer<T>` | Genetic Algorithm | Hyperparameter search |
| `GeneticAlgorithmOptimizer<T>` | Full GA implementation | Complex search spaces |
| `CMAESOptimizer<T>` | Covariance Matrix Adaptation | Black-box optimization |
| `ParticleSwarmOptimizer<T>` | Particle Swarm | Global optimization |
| `DifferentialEvolutionOptimizer<T>` | Differential evolution | Continuous optimization |
| `SimulatedAnnealingOptimizer<T>` | Simulated Annealing | Combinatorial optimization |
| `GreyWolfOptimizer<T>` | Grey Wolf optimization | Metaheuristic search |
| `AntColonyOptimizer<T>` | Ant Colony optimization | Discrete optimization |

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

---

## Error Handling

Validate optimizer parameters before training to catch configuration issues early:

```csharp
try
{
    var result = await new AiModelBuilder<float, Tensor<float>, int>()
        .ConfigureModel(model)
        .ConfigureOptimizer(new AdamWOptimizer<float>(learningRate: 3e-4f))
        .BuildAsync(trainData, trainLabels);
}
catch (ArgumentOutOfRangeException ex)
{
    Console.WriteLine($"Invalid optimizer parameter: {ex.Message}");
}
catch (InvalidOperationException ex)
{
    Console.WriteLine($"Training failed: {ex.Message}");
}
```

Common issues:
- **Learning rate too high**: Loss diverges (NaN). Try reducing by 10x.
- **Learning rate too low**: Loss barely decreases. Try increasing by 10x.
- **NaN gradients**: Enable gradient clipping via `maxGradNorm`.
