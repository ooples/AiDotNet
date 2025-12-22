using AiDotNet.Interfaces;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.ContinualLearning.Results;
using AiDotNet.ContinualLearning.Strategies;
using AiDotNet.Helpers;

namespace AiDotNet.ContinualLearning.Trainers;

/// <summary>
/// Configuration options for the MAS trainer.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> These options control how the MAS trainer operates.
/// MAS protects important parameters by measuring how sensitive the network output
/// is to each parameter, without needing task labels.</para>
/// </remarks>
public class MASTrainerOptions<T>
{
    /// <summary>
    /// Weight for the regularization loss relative to task loss.
    /// Default: 1.0 (equal weighting).
    /// </summary>
    public double? RegularizationWeight { get; set; }

    /// <summary>
    /// Whether to use experience replay alongside MAS regularization.
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
    /// Whether to compute importance in batches for efficiency.
    /// Default: true.
    /// </summary>
    public bool? UseBatchedImportance { get; set; }

    /// <summary>
    /// Batch size for importance computation.
    /// Default: 32.
    /// </summary>
    public int? ImportanceBatchSize { get; set; }

    /// <summary>
    /// Number of samples to use for importance estimation after task completion.
    /// Default: 256.
    /// </summary>
    public int? ImportanceSamples { get; set; }
}

/// <summary>
/// Continual learning trainer using Memory Aware Synapses (MAS).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This trainer implements continual learning using MAS,
/// which prevents catastrophic forgetting by measuring how sensitive the network
/// output is to each parameter. The key advantage is that MAS is unsupervised -
/// it doesn't need task labels, just input data!</para>
///
/// <para><b>How MAS Works:</b></para>
/// <list type="number">
/// <item><description>After training on a task, compute output sensitivity for each parameter</description></item>
/// <item><description>Parameters causing large output changes are marked as important</description></item>
/// <item><description>When learning new tasks, penalize changes to important parameters</description></item>
/// <item><description>Importance is computed as: Ω_i = (1/N) × Σ_n |∂||F(x_n)||²/∂θ_i|</description></item>
/// </list>
///
/// <para><b>MAS Regularization:</b></para>
/// <code>
/// L_total = L_task + (λ/2) × Σ Ω_i × (θ_i - θ*_i)²
/// </code>
/// <para>Where Ω_i is the accumulated importance and θ*_i are the optimal parameters.</para>
///
/// <para><b>Key Advantages over EWC:</b></para>
/// <list type="bullet">
/// <item><description>Unsupervised: doesn't need task labels, just input data</description></item>
/// <item><description>Can be computed on any unlabeled data distribution</description></item>
/// <item><description>Simpler than Fisher Information - no loss function needed for importance</description></item>
/// <item><description>Works well for transfer learning scenarios</description></item>
/// </list>
///
/// <para><b>Usage Example:</b></para>
/// <code>
/// var model = new MyNeuralNetwork();
/// var lossFunction = new CrossEntropyLoss&lt;double&gt;();
/// var config = new ContinualLearnerConfig&lt;double&gt;();
/// var masStrategy = new MemoryAwareSynapses&lt;double, Matrix, Vector&gt;(lossFunction, 1.0);
/// var trainer = new MASTrainer&lt;double, Matrix, Vector&gt;(model, lossFunction, config, masStrategy);
///
/// // Learn tasks sequentially
/// var task1Result = trainer.LearnTask(task1Data);
/// var task2Result = trainer.LearnTask(task2Data);
///
/// // Evaluate on all tasks
/// var evalResult = trainer.EvaluateAllTasks();
/// </code>
///
/// <para><b>Reference:</b> Aljundi et al. "Memory Aware Synapses: Learning what (not) to forget" (ECCV 2018)</para>
/// </remarks>
public class MASTrainer<T, TInput, TOutput> : ContinualLearnerBase<T, TInput, TOutput>
{
    [ThreadStatic]
    private static Random? _random;
    private static Random ThreadRandom => _random ??= RandomHelper.CreateSecureRandom();

    private readonly MASTrainerOptions<T> _options;
    private readonly MemoryAwareSynapses<T, TInput, TOutput>? _masStrategy;

    /// <summary>
    /// Gets the MAS-specific strategy if available.
    /// </summary>
    public MemoryAwareSynapses<T, TInput, TOutput>? MASStrategy => _masStrategy;

    /// <summary>
    /// Initializes a new MAS trainer with default options.
    /// </summary>
    /// <param name="model">The model to train.</param>
    /// <param name="lossFunction">The loss function for training.</param>
    /// <param name="config">Configuration for continual learning.</param>
    /// <param name="strategy">The MAS strategy (must be MemoryAwareSynapses or compatible).</param>
    public MASTrainer(
        IFullModel<T, TInput, TOutput> model,
        ILossFunction<T> lossFunction,
        IContinualLearnerConfig<T> config,
        IContinualLearningStrategy<T, TInput, TOutput> strategy)
        : this(model, lossFunction, config, strategy, null)
    {
    }

    /// <summary>
    /// Initializes a new MAS trainer with custom options.
    /// </summary>
    /// <param name="model">The model to train.</param>
    /// <param name="lossFunction">The loss function for training.</param>
    /// <param name="config">Configuration for continual learning.</param>
    /// <param name="strategy">The MAS strategy (must be MemoryAwareSynapses or compatible).</param>
    /// <param name="options">Training options for MAS.</param>
    public MASTrainer(
        IFullModel<T, TInput, TOutput> model,
        ILossFunction<T> lossFunction,
        IContinualLearnerConfig<T> config,
        IContinualLearningStrategy<T, TInput, TOutput> strategy,
        MASTrainerOptions<T>? options)
        : base(model, lossFunction, config, strategy)
    {
        _options = options ?? new MASTrainerOptions<T>();
        _masStrategy = strategy as MemoryAwareSynapses<T, TInput, TOutput>;
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

        // Track output sensitivity accumulation for MAS
        T cumulativeOutputSensitivity = NumOps.Zero;
        long outputSensitivitySamples = 0;

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

                    // Compute sample loss
                    var prediction = Model.Predict(input);
                    var predictionVector = ConversionsHelper.ConvertToVector<T, TOutput>(prediction);
                    var targetVector = ConversionsHelper.ConvertToVector<T, TOutput>(target);
                    var sampleLoss = LossFunction.CalculateLoss(predictionVector, targetVector);
                    batchLoss = NumOps.Add(batchLoss, sampleLoss);

                    // Track output sensitivity for monitoring (optional)
                    var outputNorm = ComputeOutputNorm(prediction);
                    cumulativeOutputSensitivity = NumOps.Add(cumulativeOutputSensitivity, outputNorm);
                    outputSensitivitySamples++;
                }

                if (batchGradients == null)
                    continue;

                // Average gradients over batch
                AverageGradients(batchGradients, actualBatchSize);

                // Compute regularization loss from MAS
                var regLoss = Strategy.ComputeRegularizationLoss(Model);
                regLoss = NumOps.Multiply(regLoss, NumOps.FromDouble(regWeight));

                // Apply gradient clipping if configured
                if (gradientClip.HasValue)
                {
                    ClipGradients(batchGradients, gradientClip.Value);
                }

                // Adjust gradients using strategy (MAS adds regularization through loss)
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

        // Compute average output sensitivity for reporting
        T avgOutputSensitivity = outputSensitivitySamples > 0
            ? NumOps.Divide(cumulativeOutputSensitivity, NumOps.FromDouble(outputSensitivitySamples))
            : NumOps.Zero;

        var result = new ContinualLearningResult<T>(
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

        return result;
    }

    /// <summary>
    /// Computes the L2 norm of an output for sensitivity tracking.
    /// </summary>
    /// <param name="output">The model output.</param>
    /// <returns>The L2 norm of the output.</returns>
    private T ComputeOutputNorm(TOutput output)
    {
        var vector = ConvertToVector(output);
        T sum = NumOps.Zero;

        for (int i = 0; i < vector.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(vector[i], vector[i]));
        }

        return NumOps.Sqrt(sum);
    }

    /// <summary>
    /// Converts an output to a Vector for norm computation.
    /// </summary>
    /// <param name="output">The output to convert.</param>
    /// <returns>The output as a vector.</returns>
    private Vector<T> ConvertToVector(TOutput output)
    {
        if (output is Vector<T> vec)
            return vec;

        if (output is T scalar)
            return new Vector<T>(new[] { scalar });

        if (output is T[] arr)
            return new Vector<T>(arr);

        if (output is IEnumerable<T> enumerable)
            return new Vector<T>(enumerable.ToArray());

        // Fallback: single element vector
        return new Vector<T>(1);
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
