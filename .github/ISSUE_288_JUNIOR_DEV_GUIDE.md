# Issue #288: Junior Developer Implementation Guide

## Documentation and Examples for Meta-Learning

**This issue creates documentation and runnable examples for ALL meta-learning features.**

### What You're Building:

1. **ReptileExample.cs**: Complete working example of Reptile meta-learning
2. **MAMLExample.cs**: Complete working example of MAML meta-learning
3. **SEALExample.cs**: Complete working example of SEAL meta-learning
4. **MetaLearning.md**: Comprehensive documentation explaining concepts and usage

---

## Understanding the Goal

This issue is about **making meta-learning accessible to beginners**. The goal is not just documentation, but:

1. **Self-contained examples** that can run independently
2. **Heavily commented code** explaining every step
3. **Before/after metrics** showing meta-learning actually works
4. **Clear documentation** explaining concepts, not just APIs

**Target Audience**: Developers new to meta-learning who want to:
- Understand what meta-learning does
- See it work on real problems
- Learn how to use the AiDotNet meta-learning APIs
- Get started quickly without reading research papers

---

## Phase 1: Runnable Examples

### AC 1.1: Reptile Example

**File**: `testconsole/Examples/MetaLearning/ReptileExample.cs`

**Requirements**:
- Self-contained (doesn't depend on other examples)
- Uses synthetic sine wave regression task
- Prints metrics before and after training
- Heavily commented to explain each step
- Demonstrates significant improvement

**Structure**:

```csharp
namespace AiDotNet.TestConsole.Examples.MetaLearning;

/// <summary>
/// Demonstrates meta-learning using the Reptile algorithm on a synthetic sine wave regression task.
/// </summary>
/// <remarks>
/// <para><b>What This Example Shows:</b>
/// - How to set up an episodic data loader for meta-learning
/// - How to configure and train a Reptile meta-learner
/// - How meta-learning enables rapid adaptation to new tasks
/// - The difference in performance before and after meta-training
/// </para>
/// <para><b>The Problem:</b>
/// We want to train a model that can quickly learn to fit sine waves with different
/// amplitudes and phases, given only a few data points (few-shot regression).
///
/// Traditional approach: Train on one specific sine wave with thousands of points.
/// Meta-learning approach: Train on many different sine waves with few points each,
///                        learning how to quickly adapt to new sine waves.
/// </para>
/// </remarks>
public static class ReptileExample
{
    public static void Run()
    {
        Console.WriteLine("=== Reptile Meta-Learning Example ===");
        Console.WriteLine();

        // STEP 1: Create synthetic sine wave dataset
        // ==========================================
        Console.WriteLine("[Step 1] Creating synthetic sine wave dataset...");

        // Generate 1000 different sine wave tasks
        // Each task has different amplitude (0.1 to 5.0) and phase (0 to 2π)
        var (datasetX, datasetY, taskMetadata) = GenerateSineWaveDataset(
            numTasks: 1000,
            pointsPerTask: 100,
            minAmplitude: 0.1,
            maxAmplitude: 5.0);

        Console.WriteLine($"  Generated {numTasks} sine wave tasks");
        Console.WriteLine($"  Each task has {pointsPerTask} data points");
        Console.WriteLine();

        // STEP 2: Set up episodic data loader
        // ====================================
        Console.WriteLine("[Step 2] Setting up episodic data loader...");

        // For regression, we treat different sine waves as different "classes"
        // Support set: 10 points to learn from (K-shot)
        // Query set: 50 points to test adaptation
        var dataLoader = new UniformEpisodicDataLoader<double, Matrix<double>, Vector<double>>(
            datasetX: datasetX,
            datasetY: datasetY,
            nWay: 1,           // Regression: 1 "class" (the sine wave function)
            kShot: 10,         // 10 support points to learn from
            queryShots: 50     // 50 query points to test on
        );

        Console.WriteLine("  Episodic data loader configured:");
        Console.WriteLine("    - Support set: 10 points (to adapt from)");
        Console.WriteLine("    - Query set: 50 points (to evaluate on)");
        Console.WriteLine();

        // STEP 3: Create base model
        // ==========================
        Console.WriteLine("[Step 3] Creating neural network model...");

        // Simple 2-layer MLP: [1 input] → [40 hidden] → [40 hidden] → [1 output]
        var neuralNet = new SequentialModel<double, Matrix<double>, Vector<double>>()
            .AddLayer(new DenseLayer<double>(inputSize: 1, outputSize: 40))
            .AddActivation(new ReLUActivation<double>())
            .AddLayer(new DenseLayer<double>(inputSize: 40, outputSize: 40))
            .AddActivation(new ReLUActivation<double>())
            .AddLayer(new DenseLayer<double>(inputSize: 40, outputSize: 1));

        Console.WriteLine("  Model architecture: [1 → 40 → 40 → 1]");
        Console.WriteLine();

        // STEP 4: Configure Reptile trainer
        // ==================================
        Console.WriteLine("[Step 4] Configuring Reptile meta-learner...");

        var config = new ReptileTrainerConfig<double>(
            innerLearningRate: 0.01,      // Learning rate for task adaptation
            metaLearningRate: 0.001,      // Learning rate for meta-update
            innerSteps: 5,                // Gradient steps per task
            metaBatchSize: 4,             // Number of tasks per meta-update
            numMetaIterations: 1000       // Total meta-training iterations
        );

        var metaLearner = new ReptileTrainer<double, Matrix<double>, Vector<double>>(
            metaModel: neuralNet,
            lossFunction: new MeanSquaredError<double>(),
            dataLoader: dataLoader,
            config: config
        );

        Console.WriteLine("  Configuration:");
        Console.WriteLine($"    - Inner learning rate: {config.InnerLearningRate}");
        Console.WriteLine($"    - Meta learning rate: {config.MetaLearningRate}");
        Console.WriteLine($"    - Inner steps: {config.InnerSteps}");
        Console.WriteLine($"    - Meta batch size: {config.MetaBatchSize}");
        Console.WriteLine($"    - Meta iterations: {config.NumMetaIterations}");
        Console.WriteLine();

        // STEP 5: Evaluate BEFORE meta-training
        // ======================================
        Console.WriteLine("[Step 5] Evaluating BEFORE meta-training...");

        var preTrainingResult = metaLearner.Evaluate(numTasks: 100);

        Console.WriteLine("  Baseline performance (untrained):");
        Console.WriteLine($"    - Average query loss: {preTrainingResult.LossStats.Mean:F4}");
        Console.WriteLine($"    - Loss std dev: {preTrainingResult.LossStats.StdDev:F4}");
        Console.WriteLine("    → Model cannot adapt to new sine waves yet!");
        Console.WriteLine();

        // STEP 6: Meta-training
        // =====================
        Console.WriteLine("[Step 6] Starting meta-training (this may take a few minutes)...");
        Console.WriteLine();

        var trainingResult = metaLearner.Train();

        Console.WriteLine();
        Console.WriteLine("  Meta-training complete!");
        Console.WriteLine($"    - Training time: {trainingResult.TrainingTime.TotalMinutes:F1} minutes");
        Console.WriteLine($"    - Final meta-loss: {trainingResult.FinalLoss:F4}");
        Console.WriteLine($"    - Best meta-loss: {trainingResult.BestLoss:F4}");
        Console.WriteLine();

        // STEP 7: Evaluate AFTER meta-training
        // =====================================
        Console.WriteLine("[Step 7] Evaluating AFTER meta-training...");

        var postTrainingResult = metaLearner.Evaluate(numTasks: 100);

        Console.WriteLine("  Meta-trained performance:");
        Console.WriteLine($"    - Average query loss: {postTrainingResult.LossStats.Mean:F4}");
        Console.WriteLine($"    - Loss std dev: {postTrainingResult.LossStats.StdDev:F4}");
        Console.WriteLine();

        // STEP 8: Show improvement
        // =========================
        double improvement = preTrainingResult.LossStats.Mean - postTrainingResult.LossStats.Mean;
        double improvementPercent = (improvement / preTrainingResult.LossStats.Mean) * 100;

        Console.WriteLine("[Summary]");
        Console.WriteLine($"  Loss improvement: {improvement:F4} ({improvementPercent:F1}% better)");

        if (postTrainingResult.LossStats.Mean < preTrainingResult.LossStats.Mean * 0.5)
        {
            Console.WriteLine("  ✓ SUCCESS! Meta-learning enabled rapid adaptation to new sine waves!");
        }
        else
        {
            Console.WriteLine("  ! WARNING: Meta-learning did not converge. Try:");
            Console.WriteLine("    - Increasing numMetaIterations");
            Console.WriteLine("    - Adjusting learning rates");
            Console.WriteLine("    - Increasing innerSteps");
        }

        Console.WriteLine();

        // STEP 9: Demonstrate adaptation on a specific new task
        // ======================================================
        Console.WriteLine("[Step 9] Testing adaptation on a brand new sine wave...");

        var newTask = dataLoader.GetNextTask();
        var adaptationResult = metaLearner.AdaptAndEvaluate(newTask);

        Console.WriteLine($"  Before adaptation: Loss = {adaptationResult.PerStepLosses[0]:F4}");
        Console.WriteLine($"  After {config.InnerSteps} adaptation steps: Loss = {adaptationResult.QueryLoss:F4}");
        Console.WriteLine($"  Adaptation time: {adaptationResult.AdaptationTimeMs:F1}ms");
        Console.WriteLine();
        Console.WriteLine("  → Model quickly adapted to new sine wave from just 10 points!");
        Console.WriteLine();
    }

    /// <summary>
    /// Generates a synthetic dataset of sine wave tasks for meta-learning.
    /// </summary>
    private static (Matrix<double> X, Vector<double> Y, List<SineWaveTask>) GenerateSineWaveDataset(
        int numTasks,
        int pointsPerTask,
        double minAmplitude,
        double maxAmplitude)
    {
        var random = new Random(42);
        var tasks = new List<SineWaveTask>();
        var allX = new List<double[]>();
        var allY = new List<double>();

        for (int i = 0; i < numTasks; i++)
        {
            // Random amplitude and phase for this task
            double amplitude = random.NextDouble() * (maxAmplitude - minAmplitude) + minAmplitude;
            double phase = random.NextDouble() * 2 * Math.PI;

            tasks.Add(new SineWaveTask(amplitude, phase));

            // Generate points for this sine wave
            for (int j = 0; j < pointsPerTask; j++)
            {
                double x = random.NextDouble() * 2 * Math.PI;  // Random x in [0, 2π]
                double y = amplitude * Math.Sin(x + phase);     // y = A * sin(x + φ)

                allX.Add(new[] { x });
                allY.Add(y);
            }
        }

        // Convert to Matrix/Vector
        var datasetX = new Matrix<double>(allX.Count, 1);
        for (int i = 0; i < allX.Count; i++)
        {
            datasetX[i, 0] = allX[i][0];
        }

        var datasetY = new Vector<double>(allY.ToArray());

        return (datasetX, datasetY, tasks);
    }

    private record SineWaveTask(double Amplitude, double Phase);
}
```

**Key Teaching Points in Comments**:
1. Why we use meta-learning (learn to learn vs learn once)
2. What each configuration parameter does
3. Why we evaluate before and after training
4. What the metrics mean
5. How to interpret the results

### AC 1.2: MAML Example

**File**: `testconsole/Examples/MetaLearning/MAMLExample.cs`

**Differences from Reptile**:
- Uses `MAMLTrainer` instead of `ReptileTrainer`
- Explains first-order vs second-order MAML
- Shows gradient-through-gradient computation
- Compares performance to Reptile

**Structure**: Similar to ReptileExample.cs, but with MAML-specific explanations.

### AC 1.3: SEAL Example

**File**: `testconsole/Examples/MetaLearning/SEALExample.cs`

**Key Differences**:
- Uses image classification task (rotated MNIST)
- Demonstrates self-supervised pre-training
- Shows active learning selection
- Explains pseudo-labeling
- Compares to MAML/Reptile baselines

**Synthetic Task**: Classify rotated MNIST digits
- Support set: 5 images per digit (5-shot)
- Query set: 50 images per digit (treated as unlabeled)
- Self-supervised task: Predict rotation angle (0°, 90°, 180°, 270°)

---

## Phase 2: Comprehensive Documentation

### AC 2.1: Create `docs/MetaLearning.md`

**File**: `docs/MetaLearning.md`

**Structure**:

```markdown
# Meta-Learning in AiDotNet

## Introduction to Meta-Learning

### What is Meta-Learning?

Meta-learning, or "learning to learn," trains models to quickly adapt to new tasks
with very few examples. Instead of training once on thousands of examples, meta-learning
trains across many small tasks to develop rapid adaptation capabilities.

**Traditional vs Meta-Learning:**

| Approach | Training Data | Adaptation Speed | Example |
|----------|--------------|------------------|---------|
| Traditional | 10,000 cat/dog images | Slow (retrain from scratch) | Can classify cats vs dogs |
| Meta-Learning | 100 tasks × 25 images | Fast (5 gradient steps) | Can classify ANY new category from 5 examples |

**Real-World Applications:**
- Few-shot image classification
- Rapid robot adaptation
- Personalized recommendations
- Drug discovery
- Medical diagnosis

### Key Concepts

#### N-way K-shot Learning

**N-way**: Number of classes in a task
**K-shot**: Number of labeled examples per class

Example: **5-way 3-shot classification**
```
Support Set (training):
  Class 1: [example1, example2, example3]  (3 shots)
  Class 2: [example4, example5, example6]  (3 shots)
  Class 3: [example7, example8, example9]  (3 shots)
  Class 4: [example10, example11, example12]  (3 shots)
  Class 5: [example13, example14, example15]  (3 shots)

Query Set (testing):
  New examples from all 5 classes to evaluate adaptation
```

#### Support Set vs Query Set

- **Support Set**: Small labeled dataset for adapting to the task (like training data)
- **Query Set**: Held-out examples for evaluating adaptation (like test data)

The model trains on the support set for a few steps, then is evaluated on the query set.

#### Meta-Training vs Meta-Testing

- **Meta-Training**: Train across many tasks to learn good initialization
- **Meta-Testing**: Evaluate on new, unseen tasks to measure few-shot ability

---

## The EpisodicDataLoader

### Why Meta-Learning Needs a Special Data Loader

Traditional data loaders sample individual examples. Meta-learning requires sampling
entire **tasks** (episodes) with support and query sets.

### Using EpisodicDataLoader

```csharp
// Configure 5-way 5-shot with 15 queries per class
var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(
    datasetX: trainingFeatures,
    datasetY: trainingLabels,
    nWay: 5,          // 5 classes per task
    kShot: 5,         // 5 support examples per class
    queryShots: 15    // 15 query examples per class
);

// Sample a task
var task = dataLoader.GetNextTask();

// task.SupportSetX: (25 × 784) - 25 images, 784 pixels each
// task.SupportSetY: (25 × 5) - 25 one-hot labels
// task.QuerySetX: (75 × 784) - 75 query images
// task.QuerySetY: (75 × 5) - 75 query labels
```

### Available Data Loaders

1. **UniformEpisodicDataLoader**: Random sampling (use for most cases)
2. **BalancedEpisodicDataLoader**: Ensures balanced class distribution
3. **StratifiedEpisodicDataLoader**: Stratified sampling for class balance
4. **CurriculumEpisodicDataLoader**: Progressive difficulty (easy → hard tasks)

---

## Algorithms and Usage

### 1. Reptile

**Intuition**: Move meta-parameters toward task-adapted parameters.

**When to Use**:
- Simple, fast meta-learning
- Limited computational resources
- First-order gradients only

**Example**:
```csharp
var config = new ReptileTrainerConfig<double>(
    innerLearningRate: 0.01,
    metaLearningRate: 0.001,
    innerSteps: 5,
    metaBatchSize: 4,
    numMetaIterations: 1000
);

var metaLearner = new ReptileTrainer<double, Tensor<double>, Tensor<double>>(
    metaModel: neuralNetwork,
    lossFunction: new CrossEntropyLoss<double>(),
    dataLoader: dataLoader,
    config: config
);

var result = metaLearner.Train();
```

**[Full Example](../testconsole/Examples/MetaLearning/ReptileExample.cs)**

### 2. MAML (Model-Agnostic Meta-Learning)

**Intuition**: Find initialization where few gradient steps lead to good performance.

**When to Use**:
- State-of-the-art few-shot performance
- Can afford second-order gradients
- Need theoretically principled approach

**Example**:
```csharp
var config = new MAMLTrainerConfig<double>(
    innerLearningRate: 0.01,
    metaLearningRate: 0.001,
    innerSteps: 5,
    metaBatchSize: 4,
    numMetaIterations: 1000,
    useFirstOrderApproximation: false  // Full MAML (slower but better)
);

var metaLearner = new MAMLTrainer<double, Tensor<double>, Tensor<double>>(
    metaModel: neuralNetwork,
    lossFunction: new CrossEntropyLoss<double>(),
    dataLoader: dataLoader,
    config: config
);

var result = metaLearner.Train();
```

**[Full Example](../testconsole/Examples/MetaLearning/MAMLExample.cs)**

### 3. SEAL (Self-supervised + Active Learning)

**Intuition**: Use unlabeled query data via self-supervision and pseudo-labeling.

**When to Use**:
- Large unlabeled query sets
- Image classification tasks
- Want to leverage all available data

**Example**:
```csharp
var config = new SEALTrainerConfig<double>(
    innerLearningRate: 0.01,
    metaLearningRate: 0.001,
    selfSupervisedSteps: 10,
    supervisedSteps: 5,
    activeLearningK: 20,
    metaBatchSize: 4,
    numMetaIterations: 1000
);

var metaLearner = new SEALTrainer<double, Tensor<double>, Tensor<double>>(
    metaModel: neuralNetwork,
    lossFunction: new CrossEntropyLoss<double>(),
    selfSupervisedLoss: new RotationPredictionLoss<double>(),
    dataLoader: dataLoader,
    config: config
);

var result = metaLearner.Train();
```

**[Full Example](../testconsole/Examples/MetaLearning/SEALExample.cs)**

---

## Complete Workflow

### 1. Prepare Data

```csharp
// Load your dataset (e.g., Omniglot, Mini-ImageNet, MNIST)
var (features, labels) = LoadDataset();

// Create episodic data loader
var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(
    datasetX: features,
    datasetY: labels,
    nWay: 5,
    kShot: 5,
    queryShots: 15
);
```

### 2. Create Model

```csharp
// Any model that implements IFullModel<T, TInput, TOutput>
var model = new SequentialModel<double, Tensor<double>, Tensor<double>>()
    .AddConvLayer(filters: 32, kernelSize: 3)
    .AddReLU()
    .AddMaxPool(poolSize: 2)
    .AddConvLayer(filters: 64, kernelSize: 3)
    .AddReLU()
    .AddMaxPool(poolSize: 2)
    .AddFlatten()
    .AddDense(outputSize: 128)
    .AddReLU()
    .AddDense(outputSize: 5);  // N-way classification
```

### 3. Configure Meta-Learner

```csharp
var config = new ReptileTrainerConfig<double>(
    innerLearningRate: 0.01,
    metaLearningRate: 0.001,
    innerSteps: 5,
    metaBatchSize: 4,
    numMetaIterations: 1000
);

var metaLearner = new ReptileTrainer<double, Tensor<double>, Tensor<double>>(
    metaModel: model,
    lossFunction: new CrossEntropyLoss<double>(),
    dataLoader: dataLoader,
    config: config
);
```

### 4. Meta-Train

```csharp
// Option 1: Simple training (uses config settings)
var trainingResult = metaLearner.Train();

// Option 2: Manual loop with monitoring
for (int iter = 0; iter < 1000; iter++)
{
    var stepResult = metaLearner.MetaTrainStep(batchSize: 4);

    if (iter % 100 == 0)
    {
        var evalResult = metaLearner.Evaluate(numTasks: 100);
        Console.WriteLine($"Iter {iter}: Accuracy = {evalResult.AccuracyStats.Mean:P2}");
    }
}
```

### 5. Evaluate

```csharp
// Evaluate on held-out test tasks
var testResult = metaLearner.Evaluate(numTasks: 1000);

Console.WriteLine($"Test Accuracy: {testResult.AccuracyStats.Mean:P2}");
Console.WriteLine($"95% CI: [{testResult.AccuracyStats.ConfidenceIntervalLow:P2}, " +
                  $"{testResult.AccuracyStats.ConfidenceIntervalHigh:P2}]");
```

### 6. Deploy (Adapt to New Task)

```csharp
// In production: Quickly adapt to a new task
var newTask = GetNewTaskFromUser();  // User provides 5 examples per class

var adaptationResult = metaLearner.AdaptAndEvaluate(newTask);

Console.WriteLine($"Adapted in {adaptationResult.AdaptationTimeMs}ms");
Console.WriteLine($"Query Accuracy: {adaptationResult.QueryAccuracy:P2}");
```

---

## Hyperparameter Tuning

### Learning Rates

**Inner Learning Rate (Task Adaptation)**:
- Start with: 0.01
- Increase if: Inner loop loss doesn't decrease
- Decrease if: Inner loop loss oscillates

**Meta Learning Rate (Meta-Update)**:
- Start with: 0.001 (10x smaller than inner rate)
- Increase if: Meta-training converges too slowly
- Decrease if: Meta-training is unstable

### Number of Steps

**Inner Steps**:
- Start with: 5
- Increase if: Tasks need more adaptation
- Decrease if: Overfitting to support set

**Meta Iterations**:
- Start with: 1000
- Increase if: Meta-loss still decreasing
- Monitor: Validation task performance

### Batch Size

**Meta Batch Size**:
- Start with: 4
- Increase if: Gradients are noisy
- Decrease if: Out of memory
- Typical range: 2-32

---

## Common Issues and Solutions

### Issue: Model doesn't improve after meta-training

**Possible Causes**:
1. Learning rates too high → Decrease both rates by 10x
2. Too few meta-iterations → Increase to 5000-10000
3. Too few inner steps → Increase to 10-20
4. Data loader configuration wrong → Verify N-way K-shot settings

### Issue: Meta-training is too slow

**Solutions**:
1. Use Reptile instead of MAML (10x faster)
2. Use first-order MAML approximation
3. Reduce meta batch size
4. Use GPU acceleration

### Issue: Overfitting to support set

**Symptoms**: High support accuracy, low query accuracy

**Solutions**:
1. Reduce inner steps
2. Add regularization to inner loop
3. Increase query set size
4. Use early stopping in inner loop

---

## Performance Benchmarks

### Expected Results on Standard Datasets

**Omniglot (5-way 5-shot)**:
- Random guess: 20%
- Reptile: 95-98%
- MAML: 96-99%
- SEAL: 97-99%

**Mini-ImageNet (5-way 5-shot)**:
- Random guess: 20%
- Reptile: 45-50%
- MAML: 48-52%
- SEAL: 50-55%

**Note**: Results vary based on model architecture and hyperparameters.

---

## Further Reading

### Papers
- MAML: Finn et al., ICML 2017
- Reptile: Nichol et al., arXiv 2018
- SEAL: Hao et al., ICLR 2019

### Tutorials
- [Runnable Reptile Example](../testconsole/Examples/MetaLearning/ReptileExample.cs)
- [Runnable MAML Example](../testconsole/Examples/MetaLearning/MAMLExample.cs)
- [Runnable SEAL Example](../testconsole/Examples/MetaLearning/SEALExample.cs)

### API Reference
- `IMetaLearner<T, TInput, TOutput>` interface
- `IEpisodicDataLoader<T, TInput, TOutput>` interface
- `MetaLearningTask<T, TInput, TOutput>` data structure
```

---

## Common Pitfalls to Avoid

### 1. Not Including Code Snippets

❌ **WRONG**:
```markdown
To use Reptile, create a ReptileTrainer and call Train().
```

✅ **CORRECT**:
```markdown
To use Reptile, create a ReptileTrainer and call Train():

```csharp
var metaLearner = new ReptileTrainer<double, Tensor<double>, Tensor<double>>(
    metaModel: neuralNetwork,
    lossFunction: new CrossEntropyLoss<double>(),
    dataLoader: dataLoader,
    config: config
);

var result = metaLearner.Train();
```

### 2. Assuming Prior Knowledge

❌ **WRONG**:
```markdown
Meta-learning uses gradient-based optimization in the inner loop.
```

✅ **CORRECT**:
```markdown
Meta-learning uses **two levels of optimization**:
- **Inner loop**: Quickly adapt to a specific task (5-10 gradient steps)
- **Outer loop**: Update meta-parameters across many tasks

Think of it like:
- Inner loop: Learning to solve this specific math problem
- Outer loop: Learning how to approach math problems in general
```

### 3. No Visual Examples

Include diagrams, tables, and formatted output to make concepts clear:

```markdown
### N-way K-shot Structure

| Component | 5-way 3-shot | Description |
|-----------|--------------|-------------|
| N-way | 5 | Number of classes |
| K-shot | 3 | Examples per class |
| Support set size | 15 | 5 classes × 3 shots |
| Query set size | 50 | 5 classes × 10 queries |
```

### 4. Not Showing Expected Output

Always include expected console output:

```markdown
Expected Output:
```
[Step 1] Creating synthetic sine wave dataset...
  Generated 1000 sine wave tasks
  Each task has 100 data points

[Step 5] Evaluating BEFORE meta-training...
  Baseline performance (untrained):
    - Average query loss: 2.4531
    → Model cannot adapt to new sine waves yet!

[Step 7] Evaluating AFTER meta-training...
  Meta-trained performance:
    - Average query loss: 0.3214
  ✓ SUCCESS! Meta-learning enabled rapid adaptation!
```
```

### 5. Not Linking Examples and Docs

Documentation should **link to runnable examples**:

```markdown
For a complete working example, see [ReptileExample.cs](../testconsole/Examples/MetaLearning/ReptileExample.cs).
```

---

## Testing Your Documentation

### Checklist:

- [ ] Can a beginner understand what meta-learning is?
- [ ] Are all code snippets complete and runnable?
- [ ] Does every example print before/after metrics?
- [ ] Are all configuration parameters explained?
- [ ] Is there a complete end-to-end workflow?
- [ ] Are common issues and solutions documented?
- [ ] Do all links work?
- [ ] Is the documentation organized logically?

### Test with a Real Beginner:

Ask someone unfamiliar with meta-learning to:
1. Read the documentation
2. Run the examples
3. Explain back what meta-learning does
4. Implement a simple meta-learning task

If they succeed, your documentation is effective!

---

## Integration with testconsole/Program.cs

Add menu options for each example:

```csharp
// In Program.cs
public static void Main()
{
    Console.WriteLine("AiDotNet Examples");
    Console.WriteLine("=================");
    Console.WriteLine();
    Console.WriteLine("Meta-Learning Examples:");
    Console.WriteLine("  1. Reptile (Sine Wave Regression)");
    Console.WriteLine("  2. MAML (Sine Wave Regression)");
    Console.WriteLine("  3. SEAL (Rotated MNIST Classification)");
    Console.WriteLine();
    Console.Write("Select example: ");

    var choice = Console.ReadLine();

    switch (choice)
    {
        case "1":
            ReptileExample.Run();
            break;
        case "2":
            MAMLExample.Run();
            break;
        case "3":
            SEALExample.Run();
            break;
        default:
            Console.WriteLine("Invalid choice");
            break;
    }
}
```

---

## Definition of Done

### Phase 1: Examples
- [ ] `ReptileExample.cs` created with heavily commented code
- [ ] `MAMLExample.cs` created with MAML-specific explanations
- [ ] `SEALExample.cs` created with self-supervised learning demo
- [ ] All examples are self-contained and runnable
- [ ] All examples print before/after metrics
- [ ] All examples demonstrate significant improvement
- [ ] Examples added to `Program.cs` menu

### Phase 2: Documentation
- [ ] `docs/MetaLearning.md` created
- [ ] Introduction to meta-learning section complete
- [ ] EpisodicDataLoader section with code examples
- [ ] Algorithm sections (Reptile, MAML, SEAL) with usage
- [ ] Complete workflow section (data → train → deploy)
- [ ] Hyperparameter tuning guide
- [ ] Common issues and solutions
- [ ] All code snippets tested and verified
- [ ] All links to examples work

### Quality Checks
- [ ] Documentation is beginner-friendly (no jargon without explanation)
- [ ] All concepts illustrated with examples
- [ ] Expected output shown for all code snippets
- [ ] Real-world applications explained
- [ ] Performance benchmarks provided
- [ ] Further reading references included

---

## Next Steps

1. **Start with ReptileExample.cs**: It's the simplest algorithm
2. **Test the example**: Verify it actually works and shows improvement
3. **Then create MAMLExample.cs**: Similar structure but different algorithm
4. **Then SEALExample.cs**: More complex, uses different dataset
5. **Finally write MetaLearning.md**: Document all three algorithms

**Remember**: The goal is to make meta-learning accessible, not just documented!
