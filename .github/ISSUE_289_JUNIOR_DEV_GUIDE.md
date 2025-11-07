# Issue #289: Junior Developer Implementation Guide

## Implement SEAL Meta-Learning Algorithm (Core Integration)

**This issue implements the SEAL (Self-supervised and Episodic Active Learning) meta-learning algorithm.**

### What You're Building:

1. **SEALTrainer<T, TInput, TOutput>**: Main trainer implementing SEAL algorithm
2. **Self-supervised loss function**: For pre-training on unlabeled data (rotation prediction)
3. **Active learning selection**: Choose most confident examples to pseudo-label
4. **Three-phase training loop**: Self-supervised → Active learning → Supervised fine-tuning
5. **Integration tests**: Prove SEAL outperforms baseline methods

---

## Understanding SEAL

### What is SEAL?

**SEAL** (Self-supervised and Episodic Active Learning) combines three powerful ideas to improve few-shot learning:

1. **Self-supervised learning**: Learn representations from unlabeled data
2. **Active learning**: Select most informative examples to label
3. **Pseudo-labeling**: Use model's confident predictions as additional training data

**Key Insight**: In few-shot learning, the query set is often large but unlabeled. SEAL leverages this unlabeled data to improve adaptation!

### The SEAL Advantage

```
Traditional Meta-Learning (MAML/Reptile):
  Support set: 25 labeled examples (5-way 5-shot)
  Query set: 75 examples (only used for evaluation)
  → Waste: 75 examples not used for training!

SEAL:
  Support set: 25 labeled examples (5-way 5-shot)
  Query set: 75 examples (used for self-supervised learning + pseudo-labeling)
  → Benefit: All 100 examples contribute to learning!
```

### Three-Phase Algorithm

```
Phase 1: Self-Supervised Pre-Training (on Query Set)
├─ Goal: Learn good representations without labels
├─ Method: Create artificial task (e.g., rotation prediction)
├─ Data: Query set (treated as unlabeled)
└─ Steps: 10-20 gradient updates

    Example: Rotation prediction
    - Rotate each image by 0°, 90°, 180°, 270°
    - Train model to predict rotation angle
    - Model learns: edges, shapes, spatial relationships
    - All without any class labels!

Phase 2: Active Learning Selection (on Query Set)
├─ Goal: Select most confident predictions to pseudo-label
├─ Method: Use pre-trained model to make predictions
├─ Data: Query set predictions
└─ Output: Top-K confident examples with pseudo-labels

    Example: Select top 20 most confident predictions
    - Model predicts: [Cat:95%, Dog:92%, Bird:88%, Car:87%, Tree:45%]
    - Select: Top 4 examples (confidence > 85%)
    - Pseudo-label: Assign predicted class as "ground truth"

Phase 3: Supervised Fine-Tuning (on Support + Pseudo-Labeled)
├─ Goal: Adapt to task using real + pseudo labels
├─ Method: Standard supervised learning
├─ Data: Support set (25 real labels) + Pseudo-labeled set (20 examples)
└─ Steps: 5-10 gradient updates on combined set (45 total examples)
```

### Why This Works

1. **Self-supervision provides initialization**:
   - Better starting point than random
   - Model already understands image structure
   - Reduces gradient steps needed for adaptation

2. **Active learning adds training data**:
   - Model is confident on some query examples
   - These confident predictions are usually correct
   - More training data = better adaptation

3. **Combined approach is synergistic**:
   - Self-supervision helps make confident predictions
   - Confident predictions provide more training data
   - More training data leads to better final model

### SEAL vs MAML vs Reptile

| Algorithm | Uses Query Set? | Complexity | Performance | When to Use |
|-----------|----------------|------------|-------------|-------------|
| Reptile | No (only evaluation) | Low | Good | Fast baseline |
| MAML | No (only evaluation) | High | Very Good | State-of-the-art |
| SEAL | Yes (self-supervised + pseudo-labeling) | Medium | Best | Large query sets, image tasks |

---

## Phase 1: SEALTrainer Implementation

### AC 1.1: Scaffolding SEALTrainer

**File**: `src/MetaLearning/Trainers/SEALTrainer.cs`

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
/// Implementation of SEAL (Self-supervised and Episodic Active Learning) meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <remarks>
/// <para>
/// SEAL (Hao et al., 2019) extends meta-learning by leveraging unlabeled data in the query set
/// through self-supervised learning and active learning. This enables better few-shot performance
/// by utilizing all available data, not just the labeled support set.
/// </para>
/// <para><b>Algorithm - SEAL Three-Phase Training:</b>
/// <code>
/// Initialize: θ (meta-parameters)
///
/// for iteration = 1 to N:
///     # Sample task with support (labeled) and query (treat as unlabeled)
///     task = SampleTask()
///     θ_task = Clone(θ)
///
///     # Phase 1: Self-Supervised Pre-Training on Query Set
///     for step = 1 to K_self:
///         # Create self-supervised task (e.g., rotation prediction)
///         X_rotated, Y_rotation = ApplyRotations(task.QuerySetX)
///         θ_task = θ_task - α * ∇L_self(θ_task, X_rotated, Y_rotation)
///
///     # Phase 2: Active Learning - Select Confident Examples
///     predictions = θ_task(task.QuerySetX)
///     confidence = max(predictions, axis=1)  # Max probability per example
///     top_k_indices = argsort(confidence)[-K_active:]
///     X_pseudo = task.QuerySetX[top_k_indices]
///     Y_pseudo = argmax(predictions[top_k_indices])  # Pseudo-labels
///
///     # Phase 3: Supervised Fine-Tuning on Support + Pseudo-Labeled
///     X_combined = Concatenate(task.SupportSetX, X_pseudo)
///     Y_combined = Concatenate(task.SupportSetY, Y_pseudo)
///
///     for step = 1 to K_supervised:
///         θ_task = θ_task - α * ∇L_supervised(θ_task, X_combined, Y_combined)
///
///     # Outer Loop: Reptile-style Meta-Update
///     θ = θ + ε * (θ_task - θ)
///
/// return θ
/// </code>
/// </para>
/// <para><b>Why SEAL Works:</b>
///
/// 1. **Self-Supervised Learning**: Helps model learn useful representations without labels
///    - Example: Rotation prediction teaches spatial understanding
///    - Better initialization than random parameters
///    - Particularly effective for image data
///
/// 2. **Active Learning**: Leverages model's confident predictions
///    - High-confidence predictions are usually correct
///    - Effectively increases training data size
///    - Reduces label requirements
///
/// 3. **Combined Approach**: Synergistic effects
///    - Self-supervision → better representations
///    - Better representations → more confident predictions
///    - Confident predictions → more training data
///    - More training data → better final model
/// </para>
/// <para><b>For Beginners:</b> SEAL is like learning from a textbook with some answers missing:
///
/// Traditional approach:
/// - You only study the 5 example problems with solutions (support set)
/// - Ignore the 50 practice problems without solutions (query set)
/// - Result: Limited practice
///
/// SEAL approach:
/// - Phase 1: Figure out patterns in ALL problems (self-supervised on query set)
/// - Phase 2: Solve the problems you're confident about (active learning)
/// - Phase 3: Study both example problems AND your confident solutions (supervised)
/// - Result: More practice, better performance!
///
/// The key insight: Even unlabeled data contains valuable information!
/// </para>
/// </remarks>
public class SEALTrainer<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly ISelfSupervisedLoss<T> _selfSupervisedLoss;

    /// <summary>
    /// Initializes a new instance of the SEALTrainer.
    /// </summary>
    /// <param name="metaModel">The model to meta-train.</param>
    /// <param name="lossFunction">Loss function for supervised training.</param>
    /// <param name="selfSupervisedLoss">Loss function for self-supervised pre-training.</param>
    /// <param name="dataLoader">Episodic data loader for sampling meta-learning tasks.</param>
    /// <param name="config">Configuration object containing all hyperparameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when any parameter is null.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration validation fails.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a SEAL trainer for few-shot learning.
    ///
    /// SEAL improves on MAML/Reptile by using unlabeled query set data through:
    /// 1. Self-supervised pre-training (learn patterns without labels)
    /// 2. Active learning (select confident predictions)
    /// 3. Pseudo-labeling (use predictions as additional training data)
    ///
    /// <b>Parameters explained:</b>
    /// - <b>metaModel:</b> Your neural network to meta-train
    /// - <b>lossFunction:</b> For supervised learning (e.g., CrossEntropyLoss)
    /// - <b>selfSupervisedLoss:</b> For self-supervised pre-training (e.g., RotationPredictionLoss)
    /// - <b>dataLoader:</b> Provides N-way K-shot tasks
    /// - <b>config:</b> All hyperparameters (learning rates, steps, etc.)
    ///
    /// <b>Typical configuration:</b>
    /// - Self-supervised steps: 10-20 (learn representations)
    /// - Active learning K: 10-30 (number of pseudo-labels)
    /// - Supervised steps: 5-10 (fine-tune on combined data)
    /// </para>
    /// </remarks>
    public SEALTrainer(
        IFullModel<T, TInput, TOutput> metaModel,
        ILossFunction<T> lossFunction,
        ISelfSupervisedLoss<T> selfSupervisedLoss,
        IEpisodicDataLoader<T, TInput, TOutput> dataLoader,
        SEALTrainerConfig<T>? config = null)
        : base(metaModel, lossFunction, dataLoader, config ?? new SEALTrainerConfig<T>())
    {
        _selfSupervisedLoss = selfSupervisedLoss ?? throw new ArgumentNullException(nameof(selfSupervisedLoss));
    }

    // Implementation methods to follow...
}
```

**Key Design Decisions**:

1. **Inherits from MetaLearnerBase**: Reuses common meta-learning logic
2. **Requires ISelfSupervisedLoss**: For phase 1 pre-training
3. **Configuration-based**: All hyperparameters in SEALTrainerConfig
4. **Generic type parameters**: Works with any numeric type and data format

### AC 1.2: Implement the Train Method - Three Phases

**Implementation Structure**:

```csharp
public override MetaTrainingStepResult<T> MetaTrainStep(int batchSize)
{
    if (batchSize < 1)
        throw new ArgumentException("Batch size must be at least 1", nameof(batchSize));

    var startTime = Stopwatch.StartNew();
    var config = (SEALTrainerConfig<T>)Configuration;

    // Save original meta-parameters
    Vector<T> originalParameters = MetaModel.GetParameters();

    var parameterUpdates = new List<Vector<T>>();
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
        // PHASE 1: Self-Supervised Pre-Training on Query Set
        // ===================================================================
        SelfSupervisedPreTraining(task, config.SelfSupervisedSteps);

        // ===================================================================
        // PHASE 2: Active Learning - Select Confident Examples
        // ===================================================================
        var (pseudoX, pseudoY) = ActiveLearningSelection(task, config.ActiveLearningK);

        // ===================================================================
        // PHASE 3: Supervised Fine-Tuning on Support + Pseudo-Labeled
        // ===================================================================
        SupervisedFineTuning(task, pseudoX, pseudoY, config.SupervisedSteps);

        // Get adapted parameters
        Vector<T> adaptedParameters = MetaModel.GetParameters();
        Vector<T> parameterUpdate = adaptedParameters.Subtract(originalParameters);
        parameterUpdates.Add(parameterUpdate);

        // Evaluate on query set
        T queryLoss = ComputeLoss(MetaModel, task.QuerySetX, task.QuerySetY);
        T queryAccuracy = ComputeAccuracy(MetaModel, task.QuerySetX, task.QuerySetY);

        taskLosses.Add(queryLoss);
        taskAccuracies.Add(queryAccuracy);
    }

    // Outer loop: Meta-update (Reptile-style)
    Vector<T> averageUpdate = AverageVectors(parameterUpdates);
    Vector<T> scaledUpdate = averageUpdate.Multiply(config.MetaLearningRate);
    Vector<T> newMetaParameters = originalParameters.Add(scaledUpdate);
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

**Phase 1 Implementation - Self-Supervised Pre-Training**:

```csharp
/// <summary>
/// Performs self-supervised pre-training on the query set.
/// </summary>
/// <param name="task">The meta-learning task.</param>
/// <param name="steps">Number of self-supervised training steps.</param>
/// <remarks>
/// <para>
/// Self-supervised learning creates artificial tasks from unlabeled data.
/// For images, a common approach is rotation prediction:
/// - Rotate each image by 0°, 90°, 180°, 270°
/// - Train model to predict which rotation was applied
/// - Model learns spatial relationships and features
/// </para>
/// <para><b>For Beginners:</b> Think of this as learning to recognize patterns without knowing what they mean.
///
/// Example: Looking at 75 unlabeled animal photos
/// - You can learn: ears, tails, fur patterns, body shapes
/// - You don't know: which animal is which (no labels)
/// - But learning these features helps later classification!
/// </para>
/// </remarks>
private void SelfSupervisedPreTraining(
    MetaLearningTask<T, TInput, TOutput> task,
    int steps)
{
    for (int step = 0; step < steps; step++)
    {
        // Create self-supervised task from query set
        // Example: For images, rotate and predict rotation angle
        var (augmentedX, augmentedY) = _selfSupervisedLoss.CreateTask(task.QuerySetX);

        // Train on self-supervised task
        MetaModel.Train(augmentedX, augmentedY);
    }
}
```

**Phase 2 Implementation - Active Learning Selection**:

```csharp
/// <summary>
/// Selects the most confident examples from query set and pseudo-labels them.
/// </summary>
/// <param name="task">The meta-learning task.</param>
/// <param name="k">Number of examples to select.</param>
/// <returns>Pseudo-labeled examples (input, output pairs).</returns>
/// <remarks>
/// <para>
/// Active learning selects examples where the model is most confident:
/// 1. Make predictions on entire query set
/// 2. Calculate confidence (max probability) for each prediction
/// 3. Select top-K most confident predictions
/// 4. Use predicted class as "pseudo-label"
/// </para>
/// <para><b>For Beginners:</b> This is like solving problems you're confident about.
///
/// After self-supervised learning, model looks at 75 unlabeled images:
/// - Image 1: 95% sure it's a cat → High confidence, select it
/// - Image 2: 92% sure it's a dog → High confidence, select it
/// - Image 3: 60% sure it's a bird → Low confidence, skip it
/// - Image 4: 55% sure it's a car → Low confidence, skip it
///
/// Result: 20 examples with high-confidence predictions to use as training data
///
/// Why this works: High-confidence predictions are usually correct!
/// </para>
/// </remarks>
private (TInput pseudoX, TOutput pseudoY) ActiveLearningSelection(
    MetaLearningTask<T, TInput, TOutput> task,
    int k)
{
    // Make predictions on query set
    TOutput predictions = MetaModel.Predict(task.QuerySetX);

    // Calculate confidence for each prediction (max probability)
    var confidences = CalculateConfidences(predictions);

    // Select top-K most confident indices
    var topKIndices = SelectTopK(confidences, k);

    // Extract selected examples
    TInput pseudoX = ExtractExamples(task.QuerySetX, topKIndices);

    // Create pseudo-labels from predictions
    TOutput pseudoY = ExtractPredictions(predictions, topKIndices);

    return (pseudoX, pseudoY);
}

/// <summary>
/// Calculates confidence scores (max probability) for each prediction.
/// </summary>
private Vector<T> CalculateConfidences(TOutput predictions)
{
    // For classification: confidence = max probability per example
    // predictions shape: (N, C) where N = examples, C = classes
    // confidence shape: (N,) - max probability for each example

    if (predictions is Tensor<T> tensor)
    {
        int numExamples = tensor.Shape[0];
        int numClasses = tensor.Shape[1];
        var confidences = new Vector<T>(numExamples);

        for (int i = 0; i < numExamples; i++)
        {
            T maxProb = NumOps.Zero;
            for (int j = 0; j < numClasses; j++)
            {
                T prob = tensor[i, j];
                if (NumOps.GreaterThan(prob, maxProb))
                {
                    maxProb = prob;
                }
            }
            confidences[i] = maxProb;
        }

        return confidences;
    }

    throw new NotSupportedException($"Confidence calculation not supported for {typeof(TOutput)}");
}

/// <summary>
/// Selects indices of top-K most confident examples.
/// </summary>
private List<int> SelectTopK(Vector<T> confidences, int k)
{
    // Create (index, confidence) pairs
    var pairs = new List<(int index, T confidence)>();
    for (int i = 0; i < confidences.Length; i++)
    {
        pairs.Add((i, confidences[i]));
    }

    // Sort by confidence descending
    pairs.Sort((a, b) =>
    {
        if (NumOps.GreaterThan(a.confidence, b.confidence))
            return -1;  // a is more confident
        else if (NumOps.LessThan(a.confidence, b.confidence))
            return 1;   // b is more confident
        else
            return 0;   // Equal confidence
    });

    // Take top-K indices
    return pairs.Take(k).Select(p => p.index).ToList();
}
```

**Phase 3 Implementation - Supervised Fine-Tuning**:

```csharp
/// <summary>
/// Performs supervised fine-tuning on combined support set and pseudo-labeled examples.
/// </summary>
/// <param name="task">The meta-learning task.</param>
/// <param name="pseudoX">Pseudo-labeled input examples.</param>
/// <param name="pseudoY">Pseudo-labels.</param>
/// <param name="steps">Number of supervised training steps.</param>
/// <remarks>
/// <para>
/// Supervised fine-tuning combines real labeled data (support set) with high-confidence
/// pseudo-labeled data from active learning. This effectively increases the training set size.
/// </para>
/// <para><b>For Beginners:</b> Now you have MORE training data to learn from:
///
/// Original support set (real labels):
/// - 25 examples (5 classes × 5 shots)
///
/// Pseudo-labeled set (confident predictions):
/// - 20 examples (selected in Phase 2)
///
/// Combined training set:
/// - 45 examples total (25 real + 20 pseudo)
///
/// Training on 45 examples gives better performance than just 25!
///
/// Think of it like:
/// - 25 textbook examples with solutions (support set)
/// - 20 practice problems you're confident you solved correctly (pseudo-labels)
/// - Study both together for better exam performance!
/// </para>
/// </remarks>
private void SupervisedFineTuning(
    MetaLearningTask<T, TInput, TOutput> task,
    TInput pseudoX,
    TOutput pseudoY,
    int steps)
{
    // Combine support set and pseudo-labeled set
    TInput combinedX = CombineInputs(task.SupportSetX, pseudoX);
    TOutput combinedY = CombineOutputs(task.SupportSetY, pseudoY);

    // Standard supervised training on combined set
    for (int step = 0; step < steps; step++)
    {
        MetaModel.Train(combinedX, combinedY);
    }
}

/// <summary>
/// Combines two input sets (support + pseudo-labeled).
/// </summary>
private TInput CombineInputs(TInput input1, TInput input2)
{
    if (input1 is Tensor<T> tensor1 && input2 is Tensor<T> tensor2)
    {
        // Concatenate along first dimension (examples)
        int totalExamples = tensor1.Shape[0] + tensor2.Shape[0];
        int[] newShape = tensor1.Shape.ToArray();
        newShape[0] = totalExamples;

        var combined = new Tensor<T>(newShape);

        // Copy tensor1
        for (int i = 0; i < tensor1.Shape[0]; i++)
        {
            for (int j = 0; j < tensor1.Shape[1]; j++)
            {
                combined[i, j] = tensor1[i, j];
            }
        }

        // Copy tensor2
        for (int i = 0; i < tensor2.Shape[0]; i++)
        {
            for (int j = 0; j < tensor2.Shape[1]; j++)
            {
                combined[tensor1.Shape[0] + i, j] = tensor2[i, j];
            }
        }

        return (TInput)(object)combined;
    }

    throw new NotSupportedException($"Combining not supported for {typeof(TInput)}");
}
```

---

## Phase 2: Self-Supervised Loss Function

### Rotation Prediction Loss

**File**: `src/Losses/RotationPredictionLoss.cs`

```csharp
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Losses;

/// <summary>
/// Self-supervised loss function based on rotation prediction.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <remarks>
/// <para>
/// Rotation prediction is a self-supervised task where:
/// 1. Images are rotated by 0°, 90°, 180°, or 270°
/// 2. Model predicts which rotation was applied (4-class classification)
/// 3. Model learns spatial relationships and features
/// </para>
/// <para><b>For Beginners:</b> This teaches the model to understand image structure without labels.
///
/// Imagine showing someone 100 photos, each rotated randomly:
/// - They learn to recognize: which way is "up", spatial relationships, object orientations
/// - They don't need to know: what the objects are
///
/// After this training, when you show them 5 labeled cat photos:
/// - They already understand image structure
/// - They just need to learn: "cats look like THIS"
/// - Much faster than learning everything from scratch!
/// </para>
/// </remarks>
public class RotationPredictionLoss<T> : ISelfSupervisedLoss<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly Random _random = new Random();

    /// <summary>
    /// Creates a self-supervised rotation prediction task from input data.
    /// </summary>
    /// <param name="input">Original input images.</param>
    /// <returns>Rotated images and rotation labels (0=0°, 1=90°, 2=180°, 3=270°).</returns>
    public (Tensor<T> augmentedX, Tensor<T> augmentedY) CreateTask(Tensor<T> input)
    {
        int numImages = input.Shape[0];
        int height = input.Shape[1];
        int width = input.Shape[2];
        int channels = input.Shape.Length > 3 ? input.Shape[3] : 1;

        // Create rotated versions (4 rotations per image)
        var augmentedX = new Tensor<T>(new[] { numImages * 4, height, width, channels });
        var augmentedY = new Tensor<T>(new[] { numImages * 4, 4 });  // 4-class one-hot

        int outputIdx = 0;
        for (int i = 0; i < numImages; i++)
        {
            // Extract original image
            var image = ExtractImage(input, i);

            // Create 4 rotations
            for (int rotation = 0; rotation < 4; rotation++)
            {
                var rotatedImage = RotateImage(image, rotation * 90);

                // Store rotated image
                StoreImage(augmentedX, rotatedImage, outputIdx);

                // Store rotation label (one-hot)
                for (int j = 0; j < 4; j++)
                {
                    augmentedY[outputIdx, j] = (j == rotation) ? NumOps.One : NumOps.Zero;
                }

                outputIdx++;
            }
        }

        return (augmentedX, augmentedY);
    }

    /// <summary>
    /// Rotates an image by the specified angle.
    /// </summary>
    /// <param name="image">Original image.</param>
    /// <param name="angleDegrees">Rotation angle (0, 90, 180, or 270).</param>
    /// <returns>Rotated image.</returns>
    private Tensor<T> RotateImage(Tensor<T> image, int angleDegrees)
    {
        int height = image.Shape[0];
        int width = image.Shape[1];
        int channels = image.Shape.Length > 2 ? image.Shape[2] : 1;

        var rotated = new Tensor<T>(image.Shape);

        switch (angleDegrees)
        {
            case 0:
                // No rotation
                return image.Clone();

            case 90:
                // Rotate 90° clockwise: (i, j) → (j, height-1-i)
                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        for (int c = 0; c < channels; c++)
                        {
                            rotated[j, height - 1 - i, c] = image[i, j, c];
                        }
                    }
                }
                break;

            case 180:
                // Rotate 180°: (i, j) → (height-1-i, width-1-j)
                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        for (int c = 0; c < channels; c++)
                        {
                            rotated[height - 1 - i, width - 1 - j, c] = image[i, j, c];
                        }
                    }
                }
                break;

            case 270:
                // Rotate 270° clockwise (90° counter-clockwise): (i, j) → (width-1-j, i)
                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        for (int c = 0; c < channels; c++)
                        {
                            rotated[width - 1 - j, i, c] = image[i, j, c];
                        }
                    }
                }
                break;

            default:
                throw new ArgumentException($"Unsupported rotation angle: {angleDegrees}");
        }

        return rotated;
    }
}
```

---

## Phase 3: Testing

### AC 2.1: Unit Tests

**File**: `tests/UnitTests/MetaLearning/SEALTrainerTests.cs`

```csharp
using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.Losses;
using AiDotNet.MetaLearning.Config;
using AiDotNet.MetaLearning.Trainers;
using Xunit;

namespace AiDotNet.Tests.MetaLearning;

public class SEALTrainerTests
{
    [Fact]
    public void MetaTrainStep_SingleIteration_CompletesWithoutError()
    {
        // Arrange
        var (dataLoader, model) = CreateTestSetup();

        var config = new SEALTrainerConfig<double>(
            selfSupervisedSteps: 5,
            supervisedSteps: 3,
            activeLearningK: 10,
            innerLearningRate: 0.01,
            metaLearningRate: 0.001,
            metaBatchSize: 2,
            numMetaIterations: 1
        );

        var trainer = new SEALTrainer<double, Tensor<double>, Tensor<double>>(
            metaModel: model,
            lossFunction: new CrossEntropyLoss<double>(),
            selfSupervisedLoss: new RotationPredictionLoss<double>(),
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

        var trainer = new SEALTrainer<double, Tensor<double>, Tensor<double>>(
            metaModel: model,
            lossFunction: new CrossEntropyLoss<double>(),
            selfSupervisedLoss: new RotationPredictionLoss<double>(),
            dataLoader: dataLoader,
            config: new SEALTrainerConfig<double>()
        );

        // Act
        trainer.MetaTrainStep(batchSize: 2);
        var updatedParams = model.GetParameters();

        // Assert: Parameters should have changed
        Assert.NotEqual(initialParams, updatedParams);
    }
}
```

### AC 2.2: Integration Test

**File**: `tests/UnitTests/MetaLearning/SEALTrainerIntegrationTests.cs`

```csharp
using AiDotNet.Data.Loaders;
using AiDotNet.Losses;
using AiDotNet.MetaLearning.Config;
using AiDotNet.MetaLearning.Trainers;
using AiDotNet.Models;
using Xunit;

namespace AiDotNet.Tests.MetaLearning;

public class SEALTrainerIntegrationTests
{
    [Fact]
    public void SEAL_ImprovesFewShotClassification_OnRotatedMNIST()
    {
        // Arrange: Create synthetic rotated MNIST dataset
        var (datasetX, datasetY) = GenerateRotatedMNISTDataset(
            numClasses: 10,
            examplesPerClass: 100);

        var dataLoader = new UniformEpisodicDataLoader<double, Tensor<double>, Tensor<double>>(
            datasetX: datasetX,
            datasetY: datasetY,
            nWay: 5,
            kShot: 5,
            queryShots: 15
        );

        // Create CNN model
        var model = CreateCNNModel();

        var config = new SEALTrainerConfig<double>(
            selfSupervisedSteps: 10,
            supervisedSteps: 5,
            activeLearningK: 20,
            innerLearningRate: 0.01,
            metaLearningRate: 0.001,
            metaBatchSize: 4,
            numMetaIterations: 100  // Limited for test speed
        );

        var trainer = new SEALTrainer<double, Tensor<double>, Tensor<double>>(
            metaModel: model,
            lossFunction: new CrossEntropyLoss<double>(),
            selfSupervisedLoss: new RotationPredictionLoss<double>(),
            dataLoader: dataLoader,
            config: config
        );

        // Evaluate before meta-training
        var preTrainingAccuracy = EvaluateAccuracy(trainer, numTasks: 50);

        // Act: Meta-train
        var trainingResult = trainer.Train();

        // Evaluate after meta-training
        var postTrainingAccuracy = EvaluateAccuracy(trainer, numTasks: 50);

        // Assert: Significant improvement
        double improvement = Convert.ToDouble(postTrainingAccuracy) - Convert.ToDouble(preTrainingAccuracy);

        Assert.True(improvement > 0.3,  // At least 30% improvement
            $"Expected >30% improvement, got {improvement * 100:F1}%");

        Assert.True(Convert.ToDouble(postTrainingAccuracy) > 0.6,  // At least 60% absolute
            $"Expected >60% accuracy, got {Convert.ToDouble(postTrainingAccuracy) * 100:F1}%");
    }

    private (Tensor<double> X, Tensor<double> Y) GenerateRotatedMNISTDataset(
        int numClasses,
        int examplesPerClass)
    {
        // Generate synthetic images: 28x28 grayscale
        int imageSize = 28;
        int totalExamples = numClasses * examplesPerClass;

        var datasetX = new Tensor<double>(new[] { totalExamples, imageSize, imageSize, 1 });
        var datasetY = new Tensor<double>(new[] { totalExamples, numClasses });

        var random = new Random(42);

        for (int classIdx = 0; classIdx < numClasses; classIdx++)
        {
            for (int exampleIdx = 0; exampleIdx < examplesPerClass; exampleIdx++)
            {
                int idx = classIdx * examplesPerClass + exampleIdx;

                // Generate synthetic digit-like pattern
                for (int i = 0; i < imageSize; i++)
                {
                    for (int j = 0; j < imageSize; j++)
                    {
                        // Simple pattern based on class
                        double value = Math.Sin((i + classIdx) * 0.5) * Math.Cos((j + classIdx) * 0.5);
                        value = (value + 1.0) / 2.0;  // Normalize to [0, 1]

                        // Add random rotation effect
                        double angle = random.NextDouble() * 2 * Math.PI;
                        value += Math.Sin(angle) * 0.1;

                        datasetX[idx, i, j, 0] = value;
                    }
                }

                // One-hot label
                for (int k = 0; k < numClasses; k++)
                {
                    datasetY[idx, k] = (k == classIdx) ? 1.0 : 0.0;
                }
            }
        }

        return (datasetX, datasetY);
    }
}
```

---

## Common Pitfalls to Avoid

### 1. Using Query Set for Supervised Training

❌ **WRONG**:
```csharp
// Phase 1: Train on query set with labels (defeats the purpose!)
model.Train(task.QuerySetX, task.QuerySetY);
```

✅ **CORRECT**:
```csharp
// Phase 1: Self-supervised training on query set (NO labels)
var (augmentedX, rotationLabels) = CreateRotationTask(task.QuerySetX);
model.Train(augmentedX, rotationLabels);
```

### 2. Selecting Random Examples Instead of Confident

❌ **WRONG**:
```csharp
// Random selection (no active learning)
var randomIndices = RandomSample(querySetSize, k);
```

✅ **CORRECT**:
```csharp
// Confidence-based selection (active learning)
var predictions = model.Predict(querySetX);
var confidences = CalculateConfidences(predictions);
var topKIndices = SelectTopK(confidences, k);
```

### 3. Wrong Phase Ordering

❌ **WRONG**:
```csharp
// Supervised first (can't make confident predictions yet)
SupervisedFineTuning();
SelfSupervisedPreTraining();
ActiveLearningSelection();
```

✅ **CORRECT**:
```csharp
// Correct order: Self-supervised → Active → Supervised
SelfSupervisedPreTraining();  // Learn representations
ActiveLearningSelection();     // Select confident examples
SupervisedFineTuning();        // Train on real + pseudo labels
```

### 4. Not Using Pseudo-Labels in Phase 3

❌ **WRONG**:
```csharp
// Only train on support set (wastes pseudo-labels)
model.Train(task.SupportSetX, task.SupportSetY);
```

✅ **CORRECT**:
```csharp
// Train on combined set (support + pseudo-labeled)
var combinedX = Concatenate(task.SupportSetX, pseudoX);
var combinedY = Concatenate(task.SupportSetY, pseudoY);
model.Train(combinedX, combinedY);
```

---

## Definition of Done

### Phase 1: SEALTrainer Implementation
- [ ] `SEALTrainer<T, TInput, TOutput>` class created
- [ ] Inherits from `MetaLearnerBase<T, TInput, TOutput>`
- [ ] Implements `MetaTrainStep()` with three phases
- [ ] Self-supervised pre-training phase implemented
- [ ] Active learning selection phase implemented
- [ ] Supervised fine-tuning phase implemented
- [ ] Reptile-style meta-update implemented
- [ ] Confidence calculation implemented
- [ ] Top-K selection implemented
- [ ] Input/output combining implemented

### Phase 2: Self-Supervised Loss
- [ ] `ISelfSupervisedLoss<T>` interface created
- [ ] `RotationPredictionLoss<T>` implemented
- [ ] Image rotation logic (0°, 90°, 180°, 270°)
- [ ] Augmented task creation
- [ ] Works with Tensor<T> inputs

### Phase 3: Testing
- [ ] Unit tests for `MetaTrainStep()` created
- [ ] Unit tests for parameter updates
- [ ] Integration test on synthetic dataset
- [ ] Test shows >30% improvement over baseline
- [ ] Test shows >60% absolute accuracy
- [ ] All tests pass

### Code Quality
- [ ] Comprehensive XML documentation
- [ ] "For Beginners" sections in all docs
- [ ] Generic type parameters used correctly
- [ ] `NumOps` used for all arithmetic
- [ ] No `default!` operator used
- [ ] Proper error handling
- [ ] Test coverage >80%

---

## Next Steps

1. **Implement `SEALTrainer` skeleton**: Start with class structure
2. **Implement Phase 1**: Self-supervised pre-training
3. **Implement Phase 2**: Active learning selection
4. **Implement Phase 3**: Supervised fine-tuning
5. **Implement `RotationPredictionLoss`**: Self-supervised loss function
6. **Write unit tests**: Verify each phase works
7. **Write integration test**: Prove SEAL improves performance
8. **Create example** (Issue #288): Runnable demo in testconsole

**Remember**: SEAL's power comes from using ALL available data, not just labeled examples!
