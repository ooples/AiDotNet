# iMAML (implicit Model-Agnostic Meta-Learning) Usage Guide

## Overview

iMAML is a memory-efficient variant of MAML that uses implicit differentiation to compute meta-gradients. Instead of backpropagating through all adaptation steps, it uses the implicit function theorem to directly compute gradients at the adapted parameters, significantly reducing memory requirements.

## Key Features

- **Constant Memory Complexity**: O(N) space regardless of adaptation steps
- **True Second-Order Optimization**: No need for explicit backpropagation through inner loop
- **Configurable Hessian-Vector Products**: Finite differences or automatic differentiation
- **Preconditioned Conjugate Gradient Solver**: Jacobi, LBFGS, or no preconditioning
- **Adaptive Learning Rates**: Adam-style optimization for inner loop
- **Line Search Support**: Optional for optimal step size finding

## Basic Usage

### 1. Creating an iMAML Algorithm

```csharp
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.Models.Options;

// Create iMAML options with sensible defaults
var options = new iMAMLAlgorithmOptions<double, Matrix<double>, Vector<double>>
{
    // Base model configuration
    BaseModel = yourNeuralNetwork,
    LossFunction = new MeanSquaredErrorLoss<double>(),

    // Learning rates
    InnerLearningRate = 0.01,
    OuterLearningRate = 0.001,

    // Adaptation configuration
    AdaptationSteps = 5,

    // iMAML specific settings
    LambdaRegularization = 1.0,
    ConjugateGradientIterations = 10,
    ConjugateGradientTolerance = 1e-8,

    // Hessian-vector product method
    HessianVectorProductMethod = HessianVectorProductMethod.FiniteDifferences,
    FiniteDifferencesEpsilon = 1e-5,

    // Preconditioning
    CGPreconditioningMethod = CGPreconditioningMethod.Jacobi,

    // Adaptive learning rates
    UseAdaptiveInnerLearningRate = true,
    MinInnerLearningRate = 1e-6,
    MaxInnerLearningRate = 0.1,

    // Line search (optional)
    EnableLineSearch = false,
    LineSearchMaxIterations = 20
};

// Create iMAML algorithm instance
var imaml = new iMAMLAlgorithm<double, Matrix<double>, Vector<double>>(options);
```

### 2. Preparing Tasks

```csharp
using AiDotNet.MetaLearning.Data;

// Create a task with support set (training examples) and query set (test examples)
var supportInput = Matrix<double>.Random(numSupportExamples, numFeatures, -1, 1);
var supportOutput = Vector<double>.Random(numSupportExamples, -1, 1);
var queryInput = Matrix<double>.Random(numQueryExamples, numFeatures, -1, 1);
var queryOutput = Vector<double>.Random(numQueryExamples, -1, 1);

var task = new Task<double, Matrix<double>, Vector<double>>(
    supportInput: supportInput,
    supportOutput: supportOutput,
    queryInput: queryInput,
    queryOutput: queryOutput,
    numWays: 5,           // Number of classes
    numShots: 1,          // Examples per class in support set
    numQueryPerClass: 15, // Query examples per class
    taskId: "task-001"
);

// Create a batch of tasks
var taskBatch = new TaskBatch<double, Matrix<double>, Vector<double>>(tasks);
```

### 3. Meta-Training

```csharp
// Meta-train on a batch of tasks
var metaLoss = imaml.MetaTrain(taskBatch);
Console.WriteLine($"Meta-training loss: {metaLoss:F6}");

// Meta-train for multiple epochs
for (int epoch = 0; epoch < 100; epoch++)
{
    var batch = sampleTasks(batchSize: 4);
    var loss = imaml.MetaTrain(batch);

    if (epoch % 10 == 0)
    {
        Console.WriteLine($"Epoch {epoch}: Meta-loss = {loss:F6}");
    }
}
```

### 4. Adapting to New Tasks

```csharp
// Adapt the meta-model to a new task
var adaptedModel = imaml.Adapt(newTask);

// Use the adapted model for predictions
var predictions = adaptedModel.Predict(testInput);
```

## Advanced Configuration

### Choosing Hessian-Vector Product Method

```csharp
// For better accuracy (but more computation)
options.HessianVectorProductMethod = HessianVectorProductMethod.AutomaticDifferentiation;

// For better stability and speed (recommended for most cases)
options.HessianVectorProductMethod = HessianVectorProductMethod.FiniteDifferences;
options.FiniteDifferencesEpsilon = 1e-5;
```

### Configuring Preconditioning

```csharp
// Jacobi preconditioning (good default)
options.CGPreconditioningMethod = CGPreconditioningMethod.Jacobi;

// No preconditioning (faster but may need more iterations)
options.CGPreconditioningMethod = CGPreconditioningMethod.None;

// LBFGS preconditioning (for large problems)
options.CGPreconditioningMethod = CGPreconditioningMethod.LBFGS;
```

### Adaptive Learning Rates

```csharp
// Enable Adam-style adaptive learning rates
options.UseAdaptiveInnerLearningRate = true;

// Configure bounds
options.MinInnerLearningRate = 1e-7;
options.MaxInnerLearningRate = 0.05;

// Custom hyperparameters
// (accessed through AdaptiveLearningRateState in extended implementation)
```

### Line Search

```csharp
// Enable line search for optimal step sizes
options.EnableLineSearch = true;
options.LineSearchReduction = 0.5;
options.LineSearchMinStep = 1e-10;
options.LineSearchMaxIterations = 20;
```

## Performance Tips

### 1. Memory Efficiency
- iMAML's memory usage is constant regardless of adaptation steps
- Use larger adaptation steps without memory concerns
- Monitor memory with `MemoryDiagnoser` in benchmarks

### 2. Computational Efficiency
- Use finite differences for Hessian-vector products (default)
- Adjust CG iterations based on problem size
- Consider Jacobi preconditioning for better convergence

### 3. Convergence Speed
- Use adaptive learning rates in the inner loop
- Enable line search for difficult tasks
- Tune regularization strength (λ) based on task complexity

## Comparison with MAML

| Feature | MAML | iMAML |
|--------|------|-------|
| Memory Complexity | O(K·N) | O(N) |
| Backpropagation | Through inner loop | Implicit only |
| Adaptation Steps | Limited by memory | Unlimited |
| Convergence Speed | Fast | Slightly slower |
| Implementation | Simpler | More complex |

## Best Practices

### 1. When to Use iMAML
- Tasks requiring many adaptation steps
- Large models with many parameters
- Memory-constrained environments
- When second-order accuracy is important

### 2. Hyperparameter Tuning
- **λ (Lambda Regularization)**: Start with 1.0, increase if unstable
- **CG Iterations**: 10-20 is usually sufficient
- **CG Tolerance**: 1e-8 is a good balance of precision/speed
- **Epsilon**: 1e-5 for finite differences, 1e-7 for automatic

### 3. Common Pitfalls
- Don't set λ too high (over-regularization)
- Don't use too few CG iterations (poor convergence)
- Monitor for numerical instability with very small epsilon

## Troubleshooting

### Poor Convergence
1. Increase λ regularization
2. Check CG tolerance and iterations
3. Verify Hessian-vector product method
4. Try adaptive learning rates

### Memory Issues
1. Ensure you're using iMAML, not MAML
2. Check for unintended model copying
3. Monitor GC pressure

### Numerical Instability
1. Increase epsilon for finite differences
2. Use adaptive learning rates
3. Check gradient magnitudes

## Example: Full Training Loop

```csharp
using AiDotNet.MetaLearning.Training;

// Create trainer
var trainerOptions = new MetaTrainerOptions
{
    NumEpochs = 100,
    TasksPerEpoch = 100,
    MetaBatchSize = 4,
    NumWays = 5,
    NumShots = 1,
    NumQueryPerClass = 15,
    ValInterval = 5,
    ValTasks = 50,
    LogInterval = 1,
    CheckpointInterval = 10,
    CheckpointDir = "./checkpoints",
    EarlyStoppingPatience = 20,
    Verbose = true
};

var trainer = new MetaTrainer<double, Matrix<double>, Vector<double>>(
    algorithm: imaml,
    trainDataset: trainEpisodicDataset,
    valDataset: valEpisodicDataset,
    options: trainerOptions
);

// Train the model
var trainingHistory = trainer.Train();

// Access training metrics
foreach (var metrics in trainingHistory)
{
    Console.WriteLine($"Epoch {metrics.Epoch}: Train = {metrics.TrainLoss:F6}, " +
                     $"Val = {(metrics.ValLoss?.ToString("F6") ?? "N/A")}");
}
```

## References

- Rajeswaran, A., Finn, C., Kakade, S. M., & Levine, S. (2019). Meta-learning with implicit gradients.
- Original MAML paper: Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks

## API Reference

See the XML documentation in the source code for complete API details.