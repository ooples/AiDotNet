using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.ContinualLearning.Results;
using AiDotNet.ContinualLearning.Strategies;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ContinualLearning.Trainers;

/// <summary>
/// Configuration options for the GEM trainer.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> These options control how the GEM trainer operates.
/// GEM prevents forgetting by ensuring gradients don't increase loss on previous tasks.</para>
/// </remarks>
public class GEMTrainerOptions<T>
{
    /// <summary>
    /// Memory strength parameter (gamma) for gradient projection.
    /// Higher values enforce stricter constraints on previous task gradients.
    /// Default: 0.5.
    /// </summary>
    public double? MemoryStrength { get; set; }

    /// <summary>
    /// Margin for gradient projection. Allows small violations of the constraint.
    /// Default: 0.0 (strict constraint).
    /// </summary>
    public double? ProjectionMargin { get; set; }

    /// <summary>
    /// Number of examples to store per task for gradient computation.
    /// Default: 256.
    /// </summary>
    public int? ExamplesPerTask { get; set; }

    /// <summary>
    /// Whether to use the average gradient (A-GEM) instead of full GEM.
    /// A-GEM is more efficient but may be less effective.
    /// Default: false.
    /// </summary>
    public bool? UseAveragedGEM { get; set; }

    /// <summary>
    /// Whether to compute validation metrics after each epoch.
    /// Default: true if validation data is provided.
    /// </summary>
    public bool? ComputeValidationMetrics { get; set; }

    /// <summary>
    /// Gradient clipping threshold. Gradients with L2 norm above this will be clipped.
    /// Default: null (no clipping).
    /// </summary>
    public double? GradientClipThreshold { get; set; }

    /// <summary>
    /// Maximum number of iterations for quadratic programming solver.
    /// Default: 100.
    /// </summary>
    public int? MaxQPIterations { get; set; }

    /// <summary>
    /// Tolerance for constraint satisfaction in gradient projection.
    /// Default: 1e-6.
    /// </summary>
    public double? ConstraintTolerance { get; set; }
}

/// <summary>
/// Continual learning trainer using Gradient Episodic Memory (GEM).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This trainer implements continual learning using GEM,
/// which prevents catastrophic forgetting by ensuring that gradient updates don't
/// increase the loss on any previous task.</para>
///
/// <para><b>How GEM Works:</b></para>
/// <list type="number">
/// <item><description>Store a subset of examples from each task in episodic memory</description></item>
/// <item><description>Before each gradient update, compute gradients on stored examples</description></item>
/// <item><description>Project the current gradient to not conflict with previous task gradients</description></item>
/// <item><description>This ensures the model doesn't forget previous tasks</description></item>
/// </list>
///
/// <para><b>Gradient Projection Constraint:</b></para>
/// <code>
/// For each previous task k: g · g_k ≥ 0
/// If violated, project: g' = g - sum_k(α_k * g_k)
/// </code>
/// <para>Where α_k are solved via quadratic programming.</para>
///
/// <para><b>A-GEM Variant:</b> Uses only the average gradient of a random subset
/// of previous task examples, making it more efficient but potentially less effective.</para>
///
/// <para><b>Usage Example:</b></para>
/// <code>
/// var model = new MyNeuralNetwork();
/// var lossFunction = new CrossEntropyLoss&lt;double&gt;();
/// var config = new ContinualLearnerConfig&lt;double&gt;();
/// var gemStrategy = new GradientEpisodicMemory&lt;double, Matrix, Vector&gt;(lossFunction, 256, 0.5);
/// var trainer = new GEMTrainer&lt;double, Matrix, Vector&gt;(model, lossFunction, config, gemStrategy);
///
/// // Learn tasks sequentially
/// var task1Result = trainer.LearnTask(task1Data);
/// var task2Result = trainer.LearnTask(task2Data);
///
/// // Evaluate on all tasks
/// var evalResult = trainer.EvaluateAllTasks();
/// </code>
///
/// <para><b>References:</b></para>
/// <list type="bullet">
/// <item><description>Lopez-Paz and Ranzato "Gradient Episodic Memory for Continual Learning" (NeurIPS 2017)</description></item>
/// <item><description>Chaudhry et al. "Efficient Lifelong Learning with A-GEM" (ICLR 2019)</description></item>
/// </list>
/// </remarks>
public class GEMTrainer<T, TInput, TOutput> : ContinualLearnerBase<T, TInput, TOutput>
{
    [ThreadStatic]
    private static Random? _random;
    private static Random ThreadRandom => _random ??= RandomHelper.CreateSecureRandom();

    private readonly GEMTrainerOptions<T> _options;
    private readonly GradientEpisodicMemory<T, TInput, TOutput>? _gemStrategy;

    /// <summary>
    /// Gets the GEM-specific strategy if available.
    /// </summary>
    public GradientEpisodicMemory<T, TInput, TOutput>? GEMStrategy => _gemStrategy;

    /// <summary>
    /// Initializes a new GEM trainer with default options.
    /// </summary>
    /// <param name="model">The model to train.</param>
    /// <param name="lossFunction">The loss function for training.</param>
    /// <param name="config">Configuration for continual learning.</param>
    /// <param name="strategy">The GEM strategy (must be GradientEpisodicMemory or compatible).</param>
    public GEMTrainer(
        IFullModel<T, TInput, TOutput> model,
        ILossFunction<T> lossFunction,
        IContinualLearnerConfig<T> config,
        IContinualLearningStrategy<T, TInput, TOutput> strategy)
        : this(model, lossFunction, config, strategy, null)
    {
    }

    /// <summary>
    /// Initializes a new GEM trainer with custom options.
    /// </summary>
    /// <param name="model">The model to train.</param>
    /// <param name="lossFunction">The loss function for training.</param>
    /// <param name="config">Configuration for continual learning.</param>
    /// <param name="strategy">The GEM strategy (must be GradientEpisodicMemory or compatible).</param>
    /// <param name="options">Training options for GEM.</param>
    public GEMTrainer(
        IFullModel<T, TInput, TOutput> model,
        ILossFunction<T> lossFunction,
        IContinualLearnerConfig<T> config,
        IContinualLearningStrategy<T, TInput, TOutput> strategy,
        GEMTrainerOptions<T>? options)
        : base(model, lossFunction, config, strategy)
    {
        _options = options ?? new GEMTrainerOptions<T>();
        _gemStrategy = strategy as GradientEpisodicMemory<T, TInput, TOutput>;
    }

    /// <inheritdoc/>
    protected override ContinualLearningResult<T> TrainOnTask(
        IDataset<T, TInput, TOutput> taskData,
        IDataset<T, TInput, TOutput>? validationData,
        int earlyStoppingPatience)
    {
        var taskId = _tasksLearned;
        var batchSize = Configuration.BatchSize ?? 32;
        var numSamples = taskData.Count;
        var epochsPerTask = Configuration.EpochsPerTask ?? 10;

        var lossHistory = new List<T>();
        var constraintViolationHistory = new List<T>();
        var validationLossHistory = new List<T>();

        var memoryStrength = _options.MemoryStrength ?? 0.5;
        var projectionMargin = _options.ProjectionMargin ?? 0.0;
        var gradientClip = _options.GradientClipThreshold;
        var computeValidation = _options.ComputeValidationMetrics ?? (validationData != null);
        var useAGEM = _options.UseAveragedGEM ?? false;
        var learningRate = Configuration.LearningRate ?? NumOps.FromDouble(0.001);

        T bestValidationLoss = NumOps.FromDouble(double.MaxValue);
        int epochsWithoutImprovement = 0;
        long totalGradientUpdates = 0;
        long totalConstraintViolations = 0;

        // Track accumulated task gradient for storing after training
        Vector<T>? accumulatedTaskGradient = null;
        int gradientAccumulationCount = 0;

        for (int epoch = 0; epoch < epochsPerTask; epoch++)
        {
            T epochLoss = NumOps.Zero;
            T epochViolations = NumOps.Zero;
            int batchCount = 0;

            // Shuffle indices for this epoch
            var indices = Enumerable.Range(0, numSamples)
                .OrderBy(_ => ThreadRandom.Next())
                .ToList();

            // Iterate through batches
            for (int batchStart = 0; batchStart < numSamples; batchStart += batchSize)
            {
                int batchEnd = Math.Min(batchStart + batchSize, numSamples);
                int actualBatchSize = batchEnd - batchStart;

                // Accumulate gradients for the batch
                Vector<T>? batchGradients = null;
                T batchLoss = NumOps.Zero;

                for (int i = batchStart; i < batchEnd; i++)
                {
                    int idx = indices[i];
                    var input = taskData.GetInput(idx);
                    var target = taskData.GetOutput(idx);

                    // Compute gradients for this sample
                    var sampleGradients = Model.ComputeGradients(input, target, LossFunction);

                    // Accumulate gradients
                    if (batchGradients == null)
                    {
                        batchGradients = CloneVector(sampleGradients);
                    }
                    else
                    {
                        AccumulateGradients(batchGradients, sampleGradients);
                    }

                    // Compute sample loss - convert TOutput to Vector<T>
                    var prediction = Model.Predict(input);
                    var predictionVector = ConversionsHelper.ConvertToVector<T, TOutput>(prediction);
                    var targetVector = ConversionsHelper.ConvertToVector<T, TOutput>(target);
                    var sampleLoss = LossFunction.CalculateLoss(predictionVector, targetVector);
                    batchLoss = NumOps.Add(batchLoss, sampleLoss);
                }

                if (batchGradients == null)
                    continue;

                // Average gradients over batch
                AverageGradients(batchGradients, actualBatchSize);

                // Apply gradient clipping if configured
                if (gradientClip.HasValue)
                {
                    ClipGradients(batchGradients, gradientClip.Value);
                }

                // Check if gradient violates any constraint (for monitoring)
                bool violatedConstraint = false;
                if (_gemStrategy != null && _tasksLearned > 0)
                {
                    violatedConstraint = _gemStrategy.ViolatesConstraint(batchGradients);
                    if (violatedConstraint)
                    {
                        totalConstraintViolations++;
                        epochViolations = NumOps.Add(epochViolations, NumOps.FromDouble(1.0));
                    }
                }

                // Adjust gradients using strategy (GEM projects gradients if they violate constraints)
                var adjustedGradients = Strategy.AdjustGradients(batchGradients);

                // Apply gradients to update model
                Model.ApplyGradients(adjustedGradients, learningRate);
                totalGradientUpdates++;

                // Accumulate gradient for task gradient storage
                if (accumulatedTaskGradient == null)
                {
                    accumulatedTaskGradient = CloneVector(batchGradients);
                }
                else
                {
                    AccumulateGradients(accumulatedTaskGradient, batchGradients);
                }
                gradientAccumulationCount++;

                // Track loss
                batchLoss = NumOps.Divide(batchLoss, NumOps.FromDouble(actualBatchSize));
                epochLoss = NumOps.Add(epochLoss, batchLoss);
                batchCount++;
            }

            // Average epoch losses
            if (batchCount > 0)
            {
                epochLoss = NumOps.Divide(epochLoss, NumOps.FromDouble(batchCount));
                epochViolations = NumOps.Divide(epochViolations, NumOps.FromDouble(batchCount));
            }

            lossHistory.Add(epochLoss);
            constraintViolationHistory.Add(epochViolations);

            // Compute validation metrics if requested
            T? validationLoss = default;
            if (computeValidation && validationData != null)
            {
                var valResult = EvaluateOnDataset(validationData);
                validationLoss = valResult.Loss;
                validationLossHistory.Add(valResult.Loss);

                // Early stopping check
                if (NumOps.Compare(valResult.Loss, bestValidationLoss) < 0)
                {
                    bestValidationLoss = valResult.Loss;
                    epochsWithoutImprovement = 0;
                }
                else
                {
                    epochsWithoutImprovement++;
                    if (epochsWithoutImprovement >= earlyStoppingPatience)
                    {
                        // Raise epoch completed event before breaking
                        OnEpochCompleted(taskId, epoch, epochsPerTask, epochLoss, validationLoss);
                        break;
                    }
                }
            }

            // Raise epoch completed event
            OnEpochCompleted(taskId, epoch, epochsPerTask, epochLoss, validationLoss);
        }

        // Store the average task gradient for future constraint checking
        if (_gemStrategy != null && accumulatedTaskGradient != null && gradientAccumulationCount > 0)
        {
            AverageGradients(accumulatedTaskGradient, gradientAccumulationCount);
            _gemStrategy.StoreTaskGradient(accumulatedTaskGradient);
        }

        // Compute final metrics
        var finalTrainingResult = EvaluateOnDataset(taskData);
        var finalValidationResult = validationData != null
            ? EvaluateOnDataset(validationData)
            : (Accuracy: NumOps.Zero, Loss: NumOps.Zero);

        var result = new ContinualLearningResult<T>(
            taskId: taskId,
            trainingLoss: lossHistory.Count > 0 ? lossHistory[^1] : NumOps.Zero,
            trainingAccuracy: finalTrainingResult.Accuracy,
            averagePreviousTaskAccuracy: ComputeAveragePreviousTaskAccuracy(),
            trainingTime: TimeSpan.Zero, // Will be set by base class
            lossHistory: new Vector<T>(lossHistory.ToArray()),
            regularizationLossHistory: new Vector<T>(constraintViolationHistory.ToArray()))
        {
            ValidationLoss = validationData != null ? finalValidationResult.Loss : default,
            ValidationAccuracy = validationData != null ? finalValidationResult.Accuracy : default,
            GradientUpdates = (int)totalGradientUpdates,
            EffectiveLearningRate = learningRate
        };

        return result;
    }

    #region Helper Methods

    private Vector<T> CloneVector(Vector<T> source)
    {
        var result = new Vector<T>(source.Length);
        for (int i = 0; i < source.Length; i++)
        {
            result[i] = source[i];
        }
        return result;
    }

    private void AccumulateGradients(Vector<T> target, Vector<T> source)
    {
        for (int i = 0; i < target.Length; i++)
        {
            target[i] = NumOps.Add(target[i], source[i]);
        }
    }

    private void AverageGradients(Vector<T> gradients, int count)
    {
        var divisor = NumOps.FromDouble(count);
        for (int i = 0; i < gradients.Length; i++)
        {
            gradients[i] = NumOps.Divide(gradients[i], divisor);
        }
    }

    private void ClipGradients(Vector<T> gradients, double threshold)
    {
        T norm = ComputeGradientNorm(gradients);
        double normValue = NumOps.ToDouble(norm);

        if (normValue > threshold)
        {
            var scale = NumOps.FromDouble(threshold / normValue);
            for (int i = 0; i < gradients.Length; i++)
            {
                gradients[i] = NumOps.Multiply(gradients[i], scale);
            }
        }
    }

    private T ComputeGradientNorm(Vector<T> gradients)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < gradients.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(gradients[i], gradients[i]));
        }
        return NumOps.Sqrt(sum);
    }

    #endregion
}
