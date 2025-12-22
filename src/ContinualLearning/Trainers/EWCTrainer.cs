using AiDotNet.Interfaces;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.ContinualLearning.Config;
using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.ContinualLearning.Results;
using AiDotNet.ContinualLearning.Strategies;
using AiDotNet.Helpers;

namespace AiDotNet.ContinualLearning.Trainers;

/// <summary>
/// Configuration options for the EWC trainer.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> These options control how the EWC trainer operates.
/// EWC protects important parameters by adding a regularization penalty when they change.</para>
/// </remarks>
public class EWCTrainerOptions<T>
{
    /// <summary>
    /// Weight for the regularization loss relative to task loss.
    /// Default: 1.0 (equal weighting).
    /// </summary>
    public double? RegularizationWeight { get; set; }

    /// <summary>
    /// Whether to use experience replay alongside EWC regularization.
    /// Default: true.
    /// </summary>
    public bool? UseExperienceReplay { get; set; }

    /// <summary>
    /// Fraction of learning rate to use for replay samples.
    /// Default: 0.5 (half the main learning rate).
    /// </summary>
    public double? ReplayLearningRateFactor { get; set; }

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
    /// Whether to accumulate Fisher Information across tasks (online EWC).
    /// Default: true.
    /// </summary>
    public bool? OnlineEWC { get; set; }
}

/// <summary>
/// Continual learning trainer using Elastic Weight Consolidation (EWC).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This trainer implements continual learning using EWC,
/// which prevents catastrophic forgetting by protecting important parameters from previous tasks.</para>
///
/// <para><b>How EWC Works:</b></para>
/// <list type="number">
/// <item><description>After learning a task, compute the Fisher Information Matrix (FIM)</description></item>
/// <item><description>FIM measures how important each parameter is for the task</description></item>
/// <item><description>When learning new tasks, add a penalty for changing important parameters</description></item>
/// <item><description>L_total = L_task + λ * Σ F_i * (θ_i - θ*_i)²</description></item>
/// </list>
///
/// <para><b>Online vs Offline EWC:</b></para>
/// <list type="bullet">
/// <item><description><b>Offline:</b> Stores FIM for each task separately (higher memory)</description></item>
/// <item><description><b>Online:</b> Maintains running average of FIM (constant memory)</description></item>
/// </list>
///
/// <para><b>Usage Example:</b></para>
/// <code>
/// var model = new MyNeuralNetwork();
/// var lossFunction = new CrossEntropyLoss&lt;double&gt;();
/// var config = new ContinualLearnerConfig&lt;double&gt;();
/// var ewcStrategy = new ElasticWeightConsolidation&lt;double, Matrix, Vector&gt;(lossFunction);
/// var trainer = new EWCTrainer&lt;double, Matrix, Vector&gt;(model, lossFunction, config, ewcStrategy);
///
/// // Learn tasks sequentially
/// var task1Result = trainer.LearnTask(task1Data);
/// var task2Result = trainer.LearnTask(task2Data);
///
/// // Evaluate on all tasks
/// var evalResult = trainer.EvaluateAllTasks();
/// </code>
///
/// <para><b>Reference:</b> Kirkpatrick et al. "Overcoming catastrophic forgetting in neural networks" (PNAS 2017)</para>
/// </remarks>
public class EWCTrainer<T, TInput, TOutput> : ContinualLearnerBase<T, TInput, TOutput>
{
    [ThreadStatic]
    private static Random? _random;
    private static Random ThreadRandom => _random ??= RandomHelper.CreateSecureRandom();

    private readonly EWCTrainerOptions<T> _options;
    private readonly ElasticWeightConsolidation<T, TInput, TOutput>? _ewcStrategy;

    /// <summary>
    /// Gets the EWC-specific strategy if available.
    /// </summary>
    public ElasticWeightConsolidation<T, TInput, TOutput>? EWCStrategy => _ewcStrategy;

    /// <summary>
    /// Initializes a new EWC trainer with default options.
    /// </summary>
    /// <param name="model">The model to train.</param>
    /// <param name="lossFunction">The loss function for training.</param>
    /// <param name="config">Configuration for continual learning.</param>
    /// <param name="strategy">The EWC strategy (must be ElasticWeightConsolidation or compatible).</param>
    public EWCTrainer(
        IFullModel<T, TInput, TOutput> model,
        ILossFunction<T> lossFunction,
        IContinualLearnerConfig<T> config,
        IContinualLearningStrategy<T, TInput, TOutput> strategy)
        : this(model, lossFunction, config, strategy, null)
    {
    }

    /// <summary>
    /// Initializes a new EWC trainer with custom options.
    /// </summary>
    /// <param name="model">The model to train.</param>
    /// <param name="lossFunction">The loss function for training.</param>
    /// <param name="config">Configuration for continual learning.</param>
    /// <param name="strategy">The EWC strategy (must be ElasticWeightConsolidation or compatible).</param>
    /// <param name="options">Training options for EWC.</param>
    public EWCTrainer(
        IFullModel<T, TInput, TOutput> model,
        ILossFunction<T> lossFunction,
        IContinualLearnerConfig<T> config,
        IContinualLearningStrategy<T, TInput, TOutput> strategy,
        EWCTrainerOptions<T>? options)
        : base(model, lossFunction, config, strategy)
    {
        _options = options ?? new EWCTrainerOptions<T>();
        _ewcStrategy = strategy as ElasticWeightConsolidation<T, TInput, TOutput>;
    }

    /// <inheritdoc/>
    protected override ContinualLearningResult<T> TrainOnTask(
        IDataset<T, TInput, TOutput> taskData,
        IDataset<T, TInput, TOutput>? validationData,
        int earlyStoppingPatience)
    {
        var taskId = _tasksLearned;
        var batchSize = Configuration.BatchSize;
        var numSamples = taskData.Count;
        var epochsPerTask = Configuration.EpochsPerTask;

        var lossHistory = new List<T>();
        var regLossHistory = new List<T>();
        var validationLossHistory = new List<T>();

        var useReplay = _options.UseExperienceReplay ?? true;
        var replayLrFactor = _options.ReplayLearningRateFactor ?? 0.5;
        var regWeight = _options.RegularizationWeight ?? 1.0;
        var gradientClip = _options.GradientClipThreshold;
        var computeValidation = _options.ComputeValidationMetrics ?? (validationData != null);
        var learningRate = Configuration.LearningRate ?? NumOps.FromDouble(0.001);

        T bestValidationLoss = NumOps.FromDouble(double.MaxValue);
        int epochsWithoutImprovement = 0;
        long totalGradientUpdates = 0;

        for (int epoch = 0; epoch < epochsPerTask; epoch++)
        {
            T epochTaskLoss = NumOps.Zero;
            T epochRegLoss = NumOps.Zero;
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

                // Compute regularization loss from EWC
                var regLoss = Strategy.ComputeRegularizationLoss(Model);

                // Scale regularization loss by weight
                regLoss = NumOps.Multiply(regLoss, NumOps.FromDouble(regWeight));

                // Apply gradient clipping if configured
                if (gradientClip.HasValue)
                {
                    ClipGradients(batchGradients, gradientClip.Value);
                }

                // Adjust gradients using strategy (EWC adds regularization gradients)
                var adjustedGradients = Strategy.AdjustGradients(batchGradients);

                // Apply gradients to update model
                Model.ApplyGradients(adjustedGradients, learningRate);
                totalGradientUpdates++;

                // Track losses
                batchLoss = NumOps.Divide(batchLoss, NumOps.FromDouble(actualBatchSize));
                epochTaskLoss = NumOps.Add(epochTaskLoss, batchLoss);
                epochRegLoss = NumOps.Add(epochRegLoss, regLoss);
                batchCount++;
            }

            // Experience replay
            if (useReplay && MemoryBuffer.Count > 0)
            {
                int replaySamples = Math.Min(batchSize, MemoryBuffer.Count);
                var replayBatch = MemoryBuffer.SampleBatch(replaySamples);

                if (replayBatch.Count > 0)
                {
                    var replayGradients = ComputeReplayGradients(replayBatch);
                    if (replayGradients != null)
                    {
                        var replayLr = NumOps.Multiply(
                            learningRate,
                            NumOps.FromDouble(replayLrFactor));
                        var adjustedReplayGradients = Strategy.AdjustGradients(replayGradients);
                        Model.ApplyGradients(adjustedReplayGradients, replayLr);
                        totalGradientUpdates++;
                    }
                }
            }

            // Average epoch losses
            if (batchCount > 0)
            {
                epochTaskLoss = NumOps.Divide(epochTaskLoss, NumOps.FromDouble(batchCount));
                epochRegLoss = NumOps.Divide(epochRegLoss, NumOps.FromDouble(batchCount));
            }

            var totalEpochLoss = NumOps.Add(epochTaskLoss, epochRegLoss);
            lossHistory.Add(totalEpochLoss);
            regLossHistory.Add(epochRegLoss);

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
                        OnEpochCompleted(taskId, epoch, epochsPerTask, totalEpochLoss, validationLoss);
                        break;
                    }
                }
            }

            // Raise epoch completed event
            OnEpochCompleted(taskId, epoch, epochsPerTask, totalEpochLoss, validationLoss);
        }

        // Compute final metrics
        var finalTrainingResult = EvaluateOnDataset(taskData);
        var finalValidationResult = validationData != null
            ? EvaluateOnDataset(validationData)
            : (Accuracy: NumOps.Zero, Loss: NumOps.Zero);

        return new ContinualLearningResult<T>(
            taskId: taskId,
            trainingLoss: lossHistory.Count > 0 ? lossHistory[^1] : NumOps.Zero,
            trainingAccuracy: finalTrainingResult.Accuracy,
            averagePreviousTaskAccuracy: ComputeAveragePreviousTaskAccuracy(),
            trainingTime: TimeSpan.Zero, // Will be set by base class
            lossHistory: new Vector<T>(lossHistory.ToArray()),
            regularizationLossHistory: new Vector<T>(regLossHistory.ToArray()))
        {
            ValidationLoss = validationData != null ? finalValidationResult.Loss : default,
            ValidationAccuracy = validationData != null ? finalValidationResult.Accuracy : default,
            GradientUpdates = (int)totalGradientUpdates,
            EffectiveLearningRate = learningRate
        };
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

    private Vector<T>? ComputeReplayGradients(
        IReadOnlyList<Memory.DataPoint<T, TInput, TOutput>> replayBatch)
    {
        Vector<T>? replayGradients = null;

        foreach (var dataPoint in replayBatch)
        {
            var sampleGradients = Model.ComputeGradients(
                dataPoint.Input, dataPoint.Output, LossFunction);

            if (replayGradients == null)
            {
                replayGradients = CloneVector(sampleGradients);
            }
            else
            {
                AccumulateGradients(replayGradients, sampleGradients);
            }
        }

        if (replayGradients != null)
        {
            AverageGradients(replayGradients, replayBatch.Count);
        }

        return replayGradients;
    }

    #endregion
}
