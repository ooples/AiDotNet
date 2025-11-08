# Issue #291: Junior Developer Implementation Guide

## Implement MAML Baseline for Meta-Learning Benchmarks

**This issue implements MAML (Model-Agnostic Meta-Learning), the most influential meta-learning algorithm.**

### What You're Building:

1. **MAMLTrainer<T, TInput, TOutput>**: Main trainer implementing MAML algorithm
2. **First-order MAML (FOMAML)**: Efficient approximation (similar to Reptile)
3. **Second-order MAML**: Full algorithm with gradient-through-gradient
4. **Gradient tracking**: Backpropagation through inner loop optimization
5. **Integration tests**: Prove MAML enables few-shot learning

---

## Understanding MAML

### What is MAML?

**MAML** (Model-Agnostic Meta-Learning) finds an initialization θ where a few gradient steps lead to good performance across many tasks.

**Key Idea**: Learn parameters that are easy to adapt, not parameters that work well on their own!

### The MAML Intuition

```
Bad Initialization (random):
├─ Task 1: 100 gradient steps → 60% accuracy
├─ Task 2: 100 gradient steps → 55% accuracy
└─ Task 3: 100 gradient steps → 58% accuracy
    → Slow adaptation, mediocre performance

MAML Initialization:
├─ Task 1: 5 gradient steps → 85% accuracy
├─ Task 2: 5 gradient steps → 87% accuracy
└─ Task 3: 5 gradient steps → 86% accuracy
    → Fast adaptation, excellent performance
```

**The Magic**: MAML finds θ that's "pre-adapted" to many tasks, making fine-tuning extremely efficient!

### MAML vs Reptile

| Feature | Reptile | MAML |
|---------|---------|------|
| **Inner Loop** | Standard gradient descent | Tracked gradient descent |
| **Outer Loop** | Move toward adapted parameters | Gradient through inner loop |
| **Gradients** | First-order only | First or second-order |
| **Complexity** | Low | High (second-order) / Medium (first-order) |
| **Performance** | Good | Very Good / Best |
| **Memory** | Low | High (second-order) / Medium (first-order) |

**When to Use**:
- **Reptile**: Fast baseline, limited resources
- **FOMAML (First-Order MAML)**: Balanced performance/speed
- **MAML (Second-Order)**: State-of-the-art performance, have GPU

---

## The MAML Algorithm

### Two-Level Optimization

```
OUTER LOOP (Meta-Optimization):
Goal: Find θ that enables fast adaptation

for each meta-iteration:
    Sample batch of tasks: T1, T2, ..., Tn

    for each task Ti:
        INNER LOOP (Task Adaptation):
        Goal: Adapt θ to this specific task

        θ_i = θ  # Clone meta-parameters
        for k steps:
            # CRITICAL: Keep gradient history!
            θ_i = θ_i - α * ∇L(θ_i, support_set_i)

        # Evaluate adapted model on query set
        loss_i = L(θ_i, query_set_i)

    # Meta-update: Gradient through ENTIRE inner loop!
    meta_loss = mean(loss_i for all tasks)
    θ = θ - β * ∇_θ meta_loss  # Backprop through inner loop!
```

### The Key Difference: Gradient-Through-Gradient

**Reptile**:
```python
# Inner loop
θ_task = θ
for k steps:
    θ_task = θ_task - α * ∇L(θ_task, support)

# Outer loop: Simple parameter difference
θ = θ + β * (θ_task - θ)
```

**MAML**:
```python
# Inner loop (WITH gradient tracking)
θ_task = θ
for k steps:
    grad = ∇L(θ_task, support)  # Track this gradient!
    θ_task = θ_task - α * grad

# Outer loop: Gradient through inner loop!
loss = L(θ_task, query)
θ = θ - β * ∇_θ loss  # Backprop through all k inner steps!
```

**Why this matters**:
- Reptile: Only knows final adapted parameters
- MAML: Knows HOW parameters got adapted (full gradient path)
- Result: MAML can optimize for adaptability, not just final performance

### First-Order vs Second-Order MAML

**Second-Order MAML (Full)**:
```
Computes: ∇_θ L(θ - α * ∇L(θ, support), query)
Requires: Second derivatives (Hessian)
Cost: High memory + computation
Performance: Best

Think: "How should I change θ so that gradient descent moves it to the right place?"
```

**First-Order MAML (FOMAML)**:
```
Approximation: Ignore second derivatives
Computes: ∇_θ L(θ', query) where θ' = adapted parameters
Cost: Low (similar to Reptile)
Performance: Very Good (90% of full MAML)

Think: "Which θ leads to good adapted parameters?" (simpler question)
```

**Practical Recommendation**: Start with FOMAML, use full MAML only if needed.

---

## Phase 1: MAML Trainer Implementation

### AC 1.1: Scaffolding MAMLTrainer

**File**: `src/MetaLearning/Trainers/MAMLTrainer.cs`

**Architecture**:

```csharp
using AiDotNet.Data.Abstractions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Config;
using AiDotNet.Models.Results;
using System.Diagnostics;

namespace AiDotNet.MetaLearning.Trainers;

/// <summary>
/// Implementation of MAML (Model-Agnostic Meta-Learning) algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <remarks>
/// <para>
/// MAML (Finn et al., 2017) is the most influential meta-learning algorithm. It finds model
/// parameters that enable rapid adaptation to new tasks with few gradient steps.
/// </para>
/// <para><b>Algorithm - MAML with Batch Meta-Training:</b>
/// <code>
/// Initialize: θ (meta-parameters)
///
/// for iteration = 1 to N:
///     # Sample batch of tasks
///     tasks = SampleTasks(batch_size)
///     meta_gradients = []
///
///     for each task in tasks:
///         # Inner loop: Adapt to task (WITH gradient tracking)
///         θ_task = Clone(θ)
///         for step = 1 to K:
///             grad_support = ∇L(θ_task, support_set)
///             θ_task = θ_task - α * grad_support  # Track gradients!
///
///         # Evaluate on query set
///         loss_query = L(θ_task, query_set)
///
///         # Compute meta-gradient (backprop through inner loop)
///         if use_first_order:
///             meta_grad = ∇_θ_task loss_query  # First-order approximation
///         else:
///             meta_grad = ∇_θ loss_query       # Full second-order
///
///         meta_gradients.append(meta_grad)
///
///     # Outer loop: Meta-update
///     avg_meta_grad = mean(meta_gradients)
///     θ = θ - β * avg_meta_grad
///
/// return θ
/// </code>
/// </para>
/// <para><b>Why MAML Works:</b>
///
/// MAML optimizes for <b>adaptability</b>, not just performance:
///
/// 1. **Inner Loop**: Shows how θ adapts to each task
/// 2. **Query Loss**: Measures quality of adaptation
/// 3. **Meta-Gradient**: Tells us how to change θ to improve adaptation
/// 4. **Outer Loop**: Updates θ to be more adaptable
///
/// Result: θ that's "pre-adapted" for rapid fine-tuning on new tasks!
/// </para>
/// <para><b>For Beginners:</b> MAML is like learning to learn.
///
/// Traditional learning:
/// - Start: Random initialization
/// - Goal: Find parameters that work well on training data
/// - Problem: Slow adaptation to new tasks
///
/// MAML:
/// - Start: Random initialization
/// - Goal: Find parameters that ADAPT QUICKLY to any task
/// - Process: Simulate adapting to many tasks, optimize for fast adaptation
/// - Result: Can learn new tasks from just 5 examples!
///
/// Analogy: Learning study techniques vs memorizing facts
/// - Memorizing: Learn calculus by memorizing every formula (traditional)
/// - Study techniques: Learn HOW to learn math quickly (MAML)
/// - When you see new math (new task), study techniques help you learn faster!
/// </para>
/// <para><b>First-Order vs Second-Order MAML:</b>
///
/// <b>Second-Order MAML (Full)</b>:
/// - Computes gradients through the inner loop optimization
/// - Requires second derivatives (expensive)
/// - Best performance
/// - High memory usage
/// - Use when: Need state-of-the-art results, have GPU
///
/// <b>First-Order MAML (FOMAML)</b>:
/// - Ignores second derivatives (approximation)
/// - Much faster and memory-efficient
/// - 90% of full MAML's performance
/// - Use when: Want good results without high cost
///
/// <b>Recommendation</b>: Start with first-order, upgrade to second-order if needed.
/// </para>
/// </remarks>
public class MAMLTrainer<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly bool _useFirstOrderApproximation;

    /// <summary>
    /// Initializes a new instance of the MAMLTrainer.
    /// </summary>
    /// <param name="metaModel">The model to meta-train.</param>
    /// <param name="lossFunction">Loss function for both inner and outer loops.</param>
    /// <param name="dataLoader">Episodic data loader for sampling meta-learning tasks.</param>
    /// <param name="config">Configuration object containing all hyperparameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when any parameter is null.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration validation fails.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a MAML trainer for few-shot learning.
    ///
    /// MAML finds a model initialization that can quickly adapt to new tasks.
    /// It does this by training across many tasks and optimizing for fast adaptation.
    ///
    /// <b>Parameters explained:</b>
    /// - <b>metaModel:</b> Your neural network to meta-train
    /// - <b>lossFunction:</b> How to measure errors (used in both inner and outer loops)
    /// - <b>dataLoader:</b> Provides N-way K-shot tasks for meta-training
    /// - <b>config:</b> All hyperparameters (learning rates, steps, first/second-order, etc.)
    ///
    /// <b>Typical configuration:</b>
    /// - Inner learning rate: 0.01 (task adaptation)
    /// - Meta learning rate: 0.001 (meta-optimization)
    /// - Inner steps: 5 (gradient steps per task)
    /// - Meta batch size: 4 (tasks per meta-update)
    /// - First-order: true (use FOMAML for efficiency)
    /// </para>
    /// </remarks>
    public MAMLTrainer(
        IFullModel<T, TInput, TOutput> metaModel,
        ILossFunction<T> lossFunction,
        IEpisodicDataLoader<T, TInput, TOutput> dataLoader,
        MAMLTrainerConfig<T>? config = null)
        : base(metaModel, lossFunction, dataLoader, config ?? new MAMLTrainerConfig<T>())
    {
        var mamlConfig = (MAMLTrainerConfig<T>)Configuration;
        _useFirstOrderApproximation = mamlConfig.UseFirstOrderApproximation;
    }

    // Implementation methods to follow...
}
```

### AC 1.2: Implement the Train Method

**Implementation Structure**:

```csharp
public override MetaTrainingStepResult<T> MetaTrainStep(int batchSize)
{
    if (batchSize < 1)
        throw new ArgumentException("Batch size must be at least 1", nameof(batchSize));

    var startTime = Stopwatch.StartNew();
    var config = (MAMLTrainerConfig<T>)Configuration;

    // Save original meta-parameters
    Vector<T> originalParameters = MetaModel.GetParameters();

    var metaGradients = new List<Vector<T>>();
    var taskLosses = new List<T>();
    var taskAccuracies = new List<T>();

    // Process each task in the batch
    for (int taskIdx = 0; taskIdx < batchSize; taskIdx++)
    {
        // Sample task
        MetaLearningTask<T, TInput, TOutput> task = DataLoader.GetNextTask();

        // Reset model to original meta-parameters
        MetaModel.SetParameters(originalParameters.Clone());

        // ===================================================================
        // INNER LOOP: Task Adaptation (with gradient tracking)
        // ===================================================================
        InnerLoopAdaptation(task, config.InnerSteps, config.InnerLearningRate);

        // ===================================================================
        // QUERY EVALUATION: Measure adaptation quality
        // ===================================================================
        T queryLoss = ComputeLoss(MetaModel, task.QuerySetX, task.QuerySetY);
        T queryAccuracy = ComputeAccuracy(MetaModel, task.QuerySetX, task.QuerySetY);

        taskLosses.Add(queryLoss);
        taskAccuracies.Add(queryAccuracy);

        // ===================================================================
        // META-GRADIENT: Backprop through inner loop
        // ===================================================================
        Vector<T> metaGrad;
        if (_useFirstOrderApproximation)
        {
            // First-order: Gradient w.r.t. adapted parameters
            metaGrad = ComputeFirstOrderMetaGradient(task.QuerySetX, task.QuerySetY);
        }
        else
        {
            // Second-order: Gradient w.r.t. original parameters (through inner loop)
            metaGrad = ComputeSecondOrderMetaGradient(
                task,
                originalParameters,
                config.InnerSteps,
                config.InnerLearningRate);
        }

        metaGradients.Add(metaGrad);
    }

    // ===================================================================
    // OUTER LOOP: Meta-Update
    // ===================================================================
    Vector<T> averageMetaGrad = AverageVectors(metaGradients);
    Vector<T> scaledMetaGrad = averageMetaGrad.Multiply(config.MetaLearningRate);
    Vector<T> newMetaParameters = originalParameters.Subtract(scaledMetaGrad);
    MetaModel.SetParameters(newMetaParameters);

    _currentIteration++;
    startTime.Stop();

    // Return metrics
    var lossVector = new Vector<T>(taskLosses.ToArray());
    var accuracyVector = new Vector<T>(taskAccuracies.ToArray());
    T meanLoss = StatisticsHelper<T>.CalculateMean(lossVector);
    T meanAccuracy = StatisticsHelper<T>.CalculateMean(accuracyVector);

    return new MetaTrainingStepResult<T>(
        metaLoss: meanLoss,
        taskLoss: meanLoss,
        accuracy: meanAccuracy,
        numTasks: batchSize,
        iteration: _currentIteration,
        timeMs: startTime.Elapsed.TotalMilliseconds);
}
```

**Inner Loop Implementation**:

```csharp
/// <summary>
/// Performs inner loop adaptation with gradient tracking.
/// </summary>
/// <param name="task">The meta-learning task.</param>
/// <param name="steps">Number of inner loop steps.</param>
/// <param name="learningRate">Inner loop learning rate.</param>
/// <remarks>
/// <para>
/// The inner loop adapts the model to a specific task using the support set.
/// CRITICAL: Gradients must be tracked through this process for second-order MAML.
/// </para>
/// <para><b>For Beginners:</b> This is where the model quickly learns the task.
///
/// Example: 5-way 5-shot image classification
/// - Start: Meta-initialized parameters θ
/// - Step 1: Train on 25 support examples → θ₁
/// - Step 2: Train on 25 support examples → θ₂
/// - Step 3: Train on 25 support examples → θ₃
/// - Step 4: Train on 25 support examples → θ₄
/// - Step 5: Train on 25 support examples → θ₅ (adapted parameters)
///
/// After just 5 steps on 25 examples, the model can classify new images!
/// This only works because θ was meta-trained to adapt quickly.
/// </para>
/// </remarks>
private void InnerLoopAdaptation(
    MetaLearningTask<T, TInput, TOutput> task,
    int steps,
    T learningRate)
{
    for (int step = 0; step < steps; step++)
    {
        // Standard training step on support set
        // NOTE: For second-order MAML, the computation graph is preserved
        MetaModel.Train(task.SupportSetX, task.SupportSetY);
    }
}
```

**First-Order Meta-Gradient**:

```csharp
/// <summary>
/// Computes first-order meta-gradient (FOMAML approximation).
/// </summary>
/// <param name="queryX">Query set inputs.</param>
/// <param name="queryY">Query set labels.</param>
/// <returns>Meta-gradient w.r.t. adapted parameters.</returns>
/// <remarks>
/// <para>
/// First-order MAML ignores second derivatives for efficiency.
/// The gradient is computed w.r.t. the adapted parameters, not the original parameters.
/// </para>
/// <para><b>Mathematical Formulation:</b>
///
/// Full MAML:
///   ∇_θ L(θ - α∇L(θ, support), query)
///   ↑ Gradient w.r.t. ORIGINAL parameters (expensive)
///
/// FOMAML:
///   ∇_θ' L(θ', query) where θ' = θ - α∇L(θ, support)
///   ↑ Gradient w.r.t. ADAPTED parameters (cheap)
///
/// Approximation: Treat inner loop as a black box that produces θ'
/// Then compute gradient of query loss w.r.t. θ'
/// Finally use this as approximate gradient for θ
/// </para>
/// <para><b>For Beginners:</b> This is the efficient version of MAML.
///
/// Instead of asking:
/// "How should I change the START point θ so gradient descent reaches a better END point?"
/// (Requires tracking the entire path)
///
/// We ask:
/// "Which END point θ' is good?"
/// (Only requires evaluating the final position)
///
/// This gives 90% of the benefit at 10% of the cost!
/// </para>
/// </remarks>
private Vector<T> ComputeFirstOrderMetaGradient(TInput queryX, TOutput queryY)
{
    // Compute loss on query set
    T queryLoss = ComputeLoss(MetaModel, queryX, queryY);

    // Gradient w.r.t. current (adapted) parameters
    var gradient = MetaModel.ComputeGradient(queryLoss);

    return gradient;
}
```

**Second-Order Meta-Gradient** (Advanced):

```csharp
/// <summary>
/// Computes second-order meta-gradient (full MAML).
/// </summary>
/// <param name="task">The meta-learning task.</param>
/// <param name="originalParams">Original meta-parameters before inner loop.</param>
/// <param name="innerSteps">Number of inner loop steps.</param>
/// <param name="innerLearningRate">Inner loop learning rate.</param>
/// <returns>Meta-gradient w.r.t. original parameters.</returns>
/// <remarks>
/// <para>
/// Second-order MAML computes the gradient of query loss w.r.t. the ORIGINAL parameters,
/// backpropagating through the entire inner loop optimization process.
/// </para>
/// <para><b>Mathematical Formulation:</b>
///
/// Let φ = θ - α∇L(θ, support) be the adapted parameters after one inner step.
///
/// Full MAML meta-gradient:
///   ∇_θ L(φ, query)
///   = ∇_φ L(φ, query) · ∂φ/∂θ                    (chain rule)
///   = ∇_φ L(φ, query) · (I - α∇²L(θ, support))   (Hessian!)
///
/// Where:
/// - I is the identity matrix
/// - ∇²L(θ, support) is the Hessian (second derivatives)
/// - This requires computing and storing the Hessian (expensive!)
///
/// For K inner steps, this compounds K times.
/// </para>
/// <para><b>Implementation Strategies:</b>
///
/// <b>Strategy 1: Finite Differences (Simple but Approximate)</b>
/// - Compute ∇_θ L(θ + ε, query) and ∇_θ L(θ - ε, query)
/// - Approximate: (∇_θ L(θ + ε) - ∇_θ L(θ - ε)) / (2ε)
/// - Pros: Easy to implement
/// - Cons: Approximate, numerically unstable
///
/// <b>Strategy 2: Reverse-Mode Differentiation (Correct but Complex)</b>
/// - Build full computation graph through inner loop
/// - Backpropagate through graph to compute exact gradient
/// - Pros: Exact gradient
/// - Cons: High memory (stores entire computation graph)
///
/// <b>Strategy 3: Implicit Differentiation (Advanced)</b>
/// - Solve linear system instead of storing Hessian
/// - Pros: Lower memory than reverse-mode
/// - Cons: Complex implementation
///
/// <b>Recommendation</b>: Start with first-order MAML, implement second-order only if needed.
/// </para>
/// <para><b>For Beginners:</b> This is the "full power" version of MAML.
///
/// Imagine you're learning to shoot basketball free throws:
///
/// First-order MAML:
/// - Question: "Is this shooting position (θ') good?"
/// - Answer: Yes/No based on results
///
/// Second-order MAML:
/// - Question: "How should I change my starting position (θ) so that after practicing
///   (inner loop), I end up in a better shooting position (θ')?"
/// - Answer: Requires understanding HOW practice changes your position (Hessian)
///
/// Second-order is more powerful but much more complex to compute!
/// </para>
/// </remarks>
private Vector<T> ComputeSecondOrderMetaGradient(
    MetaLearningTask<T, TInput, TOutput> task,
    Vector<T> originalParams,
    int innerSteps,
    T innerLearningRate)
{
    // This is a simplified implementation placeholder
    // Full implementation requires automatic differentiation framework
    // with support for higher-order derivatives

    // For a production implementation, you would:
    // 1. Build computation graph tracking all inner loop operations
    // 2. Compute query loss using adapted parameters
    // 3. Backpropagate through the entire graph to original parameters
    // 4. This requires framework support (e.g., PyTorch, TensorFlow)

    // For now, fall back to first-order approximation
    Console.WriteLine("WARNING: Second-order MAML not fully implemented. Using first-order approximation.");
    return ComputeFirstOrderMetaGradient(task.QuerySetX, task.QuerySetY);

    // TODO: Implement full second-order meta-gradient computation
    // This requires extending the automatic differentiation system
}
```

---

## Phase 2: Testing

### AC 2.1: Unit Tests

**File**: `tests/UnitTests/MetaLearning/MAMLTrainerTests.cs`

```csharp
using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.Losses;
using AiDotNet.MetaLearning.Config;
using AiDotNet.MetaLearning.Trainers;
using Xunit;

namespace AiDotNet.Tests.MetaLearning;

public class MAMLTrainerTests
{
    [Fact]
    public void MetaTrainStep_FirstOrder_CompletesWithoutError()
    {
        // Arrange
        var (dataLoader, model) = CreateTestSetup();

        var config = new MAMLTrainerConfig<double>(
            innerLearningRate: 0.01,
            metaLearningRate: 0.001,
            innerSteps: 5,
            metaBatchSize: 2,
            numMetaIterations: 1,
            useFirstOrderApproximation: true  // FOMAML
        );

        var trainer = new MAMLTrainer<double, Tensor<double>, Tensor<double>>(
            metaModel: model,
            lossFunction: new CrossEntropyLoss<double>(),
            dataLoader: dataLoader,
            config: config
        );

        // Act
        var result = trainer.MetaTrainStep(batchSize: 2);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(2, result.NumTasks);
        Assert.True(result.TimeMs > 0);
        Assert.False(double.IsNaN(Convert.ToDouble(result.MetaLoss)));
    }

    [Fact]
    public void MetaTrainStep_UpdatesModelParameters()
    {
        // Arrange
        var (dataLoader, model) = CreateTestSetup();
        var initialParams = model.GetParameters().Clone();

        var trainer = new MAMLTrainer<double, Tensor<double>, Tensor<double>>(
            metaModel: model,
            lossFunction: new MeanSquaredError<double>(),
            dataLoader: dataLoader,
            config: new MAMLTrainerConfig<double>()
        );

        // Act
        trainer.MetaTrainStep(batchSize: 2);
        var updatedParams = model.GetParameters();

        // Assert: Parameters should have changed
        Assert.NotEqual(initialParams, updatedParams);
    }

    [Fact]
    public void FirstOrderMAML_FasterThan_SecondOrderMAML()
    {
        // Arrange
        var (dataLoader, model1) = CreateTestSetup();
        var (_, model2) = CreateTestSetup();

        var fomamlTrainer = new MAMLTrainer<double, Tensor<double>, Tensor<double>>(
            metaModel: model1,
            lossFunction: new MeanSquaredError<double>(),
            dataLoader: dataLoader,
            config: new MAMLTrainerConfig<double>(useFirstOrderApproximation: true)
        );

        var mamlTrainer = new MAMLTrainer<double, Tensor<double>, Tensor<double>>(
            metaModel: model2,
            lossFunction: new MeanSquaredError<double>(),
            dataLoader: dataLoader,
            config: new MAMLTrainerConfig<double>(useFirstOrderApproximation: false)
        );

        // Act
        var fomamlResult = fomamlTrainer.MetaTrainStep(batchSize: 4);
        var mamlResult = mamlTrainer.MetaTrainStep(batchSize: 4);

        // Assert: FOMAML should be faster
        Assert.True(fomamlResult.TimeMs < mamlResult.TimeMs * 2,
            $"FOMAML ({fomamlResult.TimeMs}ms) should be faster than MAML ({mamlResult.TimeMs}ms)");
    }
}
```

### AC 2.2: Integration Test

**File**: `tests/UnitTests/MetaLearning/MAMLTrainerIntegrationTests.cs`

```csharp
using AiDotNet.Data.Loaders;
using AiDotNet.Losses;
using AiDotNet.MetaLearning.Config;
using AiDotNet.MetaLearning.Trainers;
using AiDotNet.Models;
using Xunit;

namespace AiDotNet.Tests.MetaLearning;

public class MAMLTrainerIntegrationTests
{
    [Fact]
    public void MAML_ImprovesFewShotRegression_OnSineWaves()
    {
        // Arrange: Create synthetic sine wave dataset
        var (datasetX, datasetY) = GenerateSineWaveDataset(
            numTasks: 1000,
            pointsPerTask: 100);

        var dataLoader = new UniformEpisodicDataLoader<double, Matrix<double>, Vector<double>>(
            datasetX: datasetX,
            datasetY: datasetY,
            nWay: 1,          // Regression: 1 function
            kShot: 10,        // 10 points to fit from
            queryShots: 50    // 50 points to evaluate on
        );

        // Create neural network model (2-layer MLP)
        var model = new SequentialModel<double, Matrix<double>, Vector<double>>()
            .AddDense(inputSize: 1, outputSize: 40)
            .AddReLU()
            .AddDense(inputSize: 40, outputSize: 40)
            .AddReLU()
            .AddDense(inputSize: 40, outputSize: 1);

        var config = new MAMLTrainerConfig<double>(
            innerLearningRate: 0.01,
            metaLearningRate: 0.001,
            innerSteps: 5,
            metaBatchSize: 4,
            numMetaIterations: 100,  // Limited for test speed
            useFirstOrderApproximation: true
        );

        var trainer = new MAMLTrainer<double, Matrix<double>, Vector<double>>(
            metaModel: model,
            lossFunction: new MeanSquaredError<double>(),
            dataLoader: dataLoader,
            config: config
        );

        // Evaluate before meta-training
        var preTrainingLoss = EvaluateAverageLoss(trainer, numTasks: 50);

        // Act: Meta-train
        var trainingResult = trainer.Train();

        // Evaluate after meta-training
        var postTrainingLoss = EvaluateAverageLoss(trainer, numTasks: 50);

        // Assert: Significant improvement
        double improvement = Convert.ToDouble(preTrainingLoss) - Convert.ToDouble(postTrainingLoss);
        double improvementPercent = improvement / Convert.ToDouble(preTrainingLoss);

        Assert.True(improvementPercent > 0.5,  // At least 50% improvement
            $"Expected >50% improvement, got {improvementPercent * 100:F1}%");

        Assert.True(Convert.ToDouble(postTrainingLoss) < 0.5,  // Absolute loss < 0.5
            $"Expected loss <0.5, got {Convert.ToDouble(postTrainingLoss):F2}");
    }

    [Fact]
    public void MAML_ComparablePerformance_ToReptile()
    {
        // Arrange: Same dataset for both
        var (datasetX, datasetY) = GenerateSineWaveDataset(1000, 100);

        var dataLoader1 = new UniformEpisodicDataLoader<double, Matrix<double>, Vector<double>>(...);
        var dataLoader2 = new UniformEpisodicDataLoader<double, Matrix<double>, Vector<double>>(...);

        var mamlModel = CreateModel();
        var reptileModel = CreateModel();

        var mamlTrainer = new MAMLTrainer<double, Matrix<double>, Vector<double>>(...);
        var reptileTrainer = new ReptileTrainer<double, Matrix<double>, Vector<double>>(...);

        // Act: Train both
        mamlTrainer.Train();
        reptileTrainer.Train();

        // Evaluate both
        var mamlLoss = EvaluateAverageLoss(mamlTrainer, 100);
        var reptileLoss = EvaluateAverageLoss(reptileTrainer, 100);

        // Assert: MAML should be at least as good as Reptile
        Assert.True(Convert.ToDouble(mamlLoss) <= Convert.ToDouble(reptileLoss) * 1.1,
            $"MAML ({mamlLoss:F4}) should be comparable to Reptile ({reptileLoss:F4})");
    }

    private (Matrix<double> X, Vector<double> Y) GenerateSineWaveDataset(
        int numTasks,
        int pointsPerTask)
    {
        var random = new Random(42);
        var allX = new List<double>();
        var allY = new List<double>();

        for (int task = 0; task < numTasks; task++)
        {
            // Random amplitude and phase
            double amplitude = random.NextDouble() * 4.9 + 0.1;  // [0.1, 5.0]
            double phase = random.NextDouble() * 2 * Math.PI;

            for (int point = 0; point < pointsPerTask; point++)
            {
                double x = random.NextDouble() * 2 * Math.PI;
                double y = amplitude * Math.Sin(x + phase);

                allX.Add(x);
                allY.Add(y);
            }
        }

        var datasetX = new Matrix<double>(allX.Count, 1);
        for (int i = 0; i < allX.Count; i++)
        {
            datasetX[i, 0] = allX[i];
        }

        var datasetY = new Vector<double>(allY.ToArray());

        return (datasetX, datasetY);
    }
}
```

---

## Common Pitfalls to Avoid

### 1. Not Tracking Gradients in Inner Loop

❌ **WRONG**:
```csharp
// Inner loop without gradient tracking (like Reptile)
for (int step = 0; step < innerSteps; step++)
{
    model.Train(supportX, supportY);
    // Gradients discarded after each step
}
```

✅ **CORRECT** (First-Order):
```csharp
// Track final adapted parameters
for (int step = 0; step < innerSteps; step++)
{
    model.Train(supportX, supportY);
}
// Compute gradient w.r.t. adapted parameters
var metaGrad = model.ComputeGradient(queryLoss);
```

✅ **CORRECT** (Second-Order):
```csharp
// Track entire computation graph
var graph = new ComputationGraph();
for (int step = 0; step < innerSteps; step++)
{
    graph.Track(() => model.Train(supportX, supportY));
}
// Backpropagate through graph
var metaGrad = graph.Backpropagate(queryLoss, originalParams);
```

### 2. Using Support Set for Meta-Gradient

❌ **WRONG**:
```csharp
// Computing meta-gradient on support set (overfitting)
var metaGrad = ComputeGradient(model, supportX, supportY);
```

✅ **CORRECT**:
```csharp
// Computing meta-gradient on query set (generalization)
var metaGrad = ComputeGradient(model, queryX, queryY);
```

### 3. Wrong Learning Rate Scales

❌ **WRONG**:
```csharp
double innerLR = 0.001;  // Too small
double metaLR = 0.01;    // Too large (larger than inner!)
```

✅ **CORRECT**:
```csharp
double innerLR = 0.01;   // Larger (fast task adaptation)
double metaLR = 0.001;   // Smaller (slow meta-learning)
```

**Rule of Thumb**: Meta learning rate = 10% of inner learning rate

### 4. Forgetting to Reset to Original Parameters

❌ **WRONG**:
```csharp
// All tasks adapt from previously adapted parameters!
for (int i = 0; i < batchSize; i++)
{
    var task = dataLoader.GetNextTask();
    InnerLoop(task);  // Starts from wherever last task ended
}
```

✅ **CORRECT**:
```csharp
var originalParams = model.GetParameters();
for (int i = 0; i < batchSize; i++)
{
    var task = dataLoader.GetNextTask();
    model.SetParameters(originalParams.Clone());  // Reset!
    InnerLoop(task);
}
```

---

## Definition of Done

### Phase 1: MAMLTrainer Implementation
- [ ] `MAMLTrainer<T, TInput, TOutput>` class created
- [ ] Inherits from `MetaLearnerBase<T, TInput, TOutput>`
- [ ] Implements `MetaTrainStep()` method
- [ ] Inner loop adaptation implemented
- [ ] First-order meta-gradient computation
- [ ] Second-order meta-gradient (placeholder or full implementation)
- [ ] Configuration flag for first/second-order
- [ ] Meta-update using computed gradients

### Phase 2: Testing
- [ ] Unit tests for first-order MAML
- [ ] Unit tests verify parameter updates
- [ ] Unit tests verify first-order is faster
- [ ] Integration test on sine wave regression
- [ ] Test shows >50% improvement over baseline
- [ ] Test shows final loss <0.5
- [ ] Comparison test with Reptile

### Code Quality
- [ ] Comprehensive XML documentation
- [ ] "For Beginners" sections explaining concepts
- [ ] Mathematical formulations documented
- [ ] Generic type parameters used correctly
- [ ] `NumOps` used for all arithmetic
- [ ] No `default!` operator used
- [ ] Proper error handling
- [ ] Test coverage >80%

---

## Next Steps

1. **Start with scaffolding**: Create `MAMLTrainer` class structure
2. **Implement first-order MAML**: Easier, practical approximation
3. **Write unit tests**: Verify first-order works
4. **Write integration test**: Prove MAML enables few-shot learning
5. **Document second-order approach**: Explain what full MAML requires
6. **Create example** (Issue #288): Runnable demo comparing MAML vs Reptile

**Remember**: First-order MAML (FOMAML) gives 90% of full MAML's performance at 10% of the cost - start there!

---

## Further Reading

### Papers
- **MAML**: Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks", ICML 2017
- **FOMAML**: Nichol et al., "On First-Order Meta-Learning Algorithms", arXiv 2018

### Related Files (To Be Created)
- `src/MetaLearning/Trainers/MAMLTrainer.cs`
- `src/MetaLearning/Config/MAMLTrainerConfig.cs`
- `tests/UnitTests/MetaLearning/MAMLTrainerTests.cs`
- `tests/UnitTests/MetaLearning/MAMLTrainerIntegrationTests.cs`

### Dependencies
- ✅ Issue #290: Episodic Data Abstractions (completed)
- ✅ Issue #292: Reptile (reference implementation)

### Dependents
- ⏳ Issue #288: Documentation and Examples (needs MAML example)
