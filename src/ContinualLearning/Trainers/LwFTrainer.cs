using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.ContinualLearning.Results;
using AiDotNet.ContinualLearning.Strategies;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ContinualLearning.Trainers;

/// <summary>
/// Configuration options for the LwF trainer.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> These options control how the LwF trainer operates.
/// LwF uses knowledge distillation to preserve knowledge from previous tasks.</para>
/// </remarks>
public class LwFTrainerOptions<T>
{
    /// <summary>
    /// Weight for the distillation loss relative to task loss.
    /// Default: 1.0 (equal weighting).
    /// </summary>
    public double? DistillationWeight { get; set; }

    /// <summary>
    /// Temperature for softmax in distillation. Higher values produce softer probability distributions.
    /// Default: 2.0 (common choice for knowledge distillation).
    /// </summary>
    public double? Temperature { get; set; }

    /// <summary>
    /// Whether to use experience replay alongside distillation.
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
    /// Whether to include distillation loss in gradient updates (true) or only use it for monitoring (false).
    /// Default: true.
    /// </summary>
    public bool? IncludeDistillationInGradients { get; set; }

    /// <summary>
    /// Warmup epochs before applying full distillation weight.
    /// Default: 0 (no warmup).
    /// </summary>
    public int? DistillationWarmupEpochs { get; set; }
}

/// <summary>
/// Continual learning trainer using Learning without Forgetting (LwF).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This trainer implements continual learning using LwF,
/// which prevents catastrophic forgetting through knowledge distillation from a
/// "teacher" model (the model before training on the new task).</para>
///
/// <para><b>How LwF Works:</b></para>
/// <list type="number">
/// <item><description>Before learning a new task, save the current model as the "teacher"</description></item>
/// <item><description>During training, compute both task loss and distillation loss</description></item>
/// <item><description>Distillation loss measures how different the new model's outputs are from the teacher's</description></item>
/// <item><description>This ensures the model doesn't forget its previous knowledge</description></item>
/// </list>
///
/// <para><b>Key Advantage:</b> LwF doesn't need to store old task data - it only needs the
/// teacher model's outputs on current task data.</para>
///
/// <para><b>Distillation Loss:</b></para>
/// <code>
/// L_distill = -sum(softmax(teacher_output/T) * log(softmax(student_output/T))) * T²
/// </code>
/// <para>Where T is the temperature parameter.</para>
///
/// <para><b>Usage Example:</b></para>
/// <code>
/// var model = new MyNeuralNetwork();
/// var lossFunction = new CrossEntropyLoss&lt;double&gt;();
/// var config = new ContinualLearnerConfig&lt;double&gt;();
/// var lwfStrategy = new LearningWithoutForgetting&lt;double, Matrix, Vector&gt;(lossFunction, 2.0, 1.0);
/// var trainer = new LwFTrainer&lt;double, Matrix, Vector&gt;(model, lossFunction, config, lwfStrategy);
///
/// // Learn tasks sequentially
/// var task1Result = trainer.LearnTask(task1Data);
/// var task2Result = trainer.LearnTask(task2Data);
///
/// // Evaluate on all tasks
/// var evalResult = trainer.EvaluateAllTasks();
/// </code>
///
/// <para><b>Reference:</b> Li and Hoiem "Learning without Forgetting" (2017)</para>
/// </remarks>
public class LwFTrainer<T, TInput, TOutput> : ContinualLearnerBase<T, TInput, TOutput>
{
    [ThreadStatic]
    private static Random? _random;
    private static Random ThreadRandom => _random ??= RandomHelper.CreateSecureRandom();

    private readonly LwFTrainerOptions<T> _options;
    private readonly LearningWithoutForgetting<T, TInput, TOutput>? _lwfStrategy;

    /// <summary>
    /// Gets the LwF-specific strategy if available.
    /// </summary>
    public LearningWithoutForgetting<T, TInput, TOutput>? LwFStrategy => _lwfStrategy;

    /// <summary>
    /// Initializes a new LwF trainer with default options.
    /// </summary>
    /// <param name="model">The model to train.</param>
    /// <param name="lossFunction">The loss function for training.</param>
    /// <param name="config">Configuration for continual learning.</param>
    /// <param name="strategy">The LwF strategy (must be LearningWithoutForgetting or compatible).</param>
    public LwFTrainer(
        IFullModel<T, TInput, TOutput> model,
        ILossFunction<T> lossFunction,
        IContinualLearnerConfig<T> config,
        IContinualLearningStrategy<T, TInput, TOutput> strategy)
        : this(model, lossFunction, config, strategy, null)
    {
    }

    /// <summary>
    /// Initializes a new LwF trainer with custom options.
    /// </summary>
    /// <param name="model">The model to train.</param>
    /// <param name="lossFunction">The loss function for training.</param>
    /// <param name="config">Configuration for continual learning.</param>
    /// <param name="strategy">The LwF strategy (must be LearningWithoutForgetting or compatible).</param>
    /// <param name="options">Training options for LwF.</param>
    public LwFTrainer(
        IFullModel<T, TInput, TOutput> model,
        ILossFunction<T> lossFunction,
        IContinualLearnerConfig<T> config,
        IContinualLearningStrategy<T, TInput, TOutput> strategy,
        LwFTrainerOptions<T>? options)
        : base(model, lossFunction, config, strategy)
    {
        _options = options ?? new LwFTrainerOptions<T>();
        _lwfStrategy = strategy as LearningWithoutForgetting<T, TInput, TOutput>;
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
        var distillationLossHistory = new List<T>();
        var validationLossHistory = new List<T>();

        var useReplay = _options.UseExperienceReplay ?? true;
        var replayLrFactor = _options.ReplayLearningRateFactor ?? 0.5;
        var distillWeight = _options.DistillationWeight ?? 1.0;
        var temperature = _options.Temperature ?? 2.0;
        var gradientClip = _options.GradientClipThreshold;
        var computeValidation = _options.ComputeValidationMetrics ?? (validationData != null);
        var includeDistillGrads = _options.IncludeDistillationInGradients ?? true;
        var warmupEpochs = _options.DistillationWarmupEpochs ?? 0;
        var learningRate = Configuration.LearningRate ?? NumOps.FromDouble(0.001);

        // Get teacher model for distillation (only if we have previous tasks)
        var teacherModel = _lwfStrategy?.TeacherModel;
        var hasTeacher = teacherModel != null && _tasksLearned > 0;

        T bestValidationLoss = NumOps.FromDouble(double.MaxValue);
        int epochsWithoutImprovement = 0;
        long totalGradientUpdates = 0;

        for (int epoch = 0; epoch < epochsPerTask; epoch++)
        {
            T epochTaskLoss = NumOps.Zero;
            T epochDistillLoss = NumOps.Zero;
            int batchCount = 0;

            // Calculate current distillation weight (with warmup)
            double currentDistillWeight = distillWeight;
            if (epoch < warmupEpochs && warmupEpochs > 0)
            {
                currentDistillWeight = distillWeight * (epoch + 1) / warmupEpochs;
            }

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
                T batchTaskLoss = NumOps.Zero;
                T batchDistillLoss = NumOps.Zero;

                for (int i = batchStart; i < batchEnd; i++)
                {
                    int idx = indices[i];
                    var input = taskData.GetInput(idx);
                    var target = taskData.GetOutput(idx);

                    // Compute gradients for the task loss
                    var sampleGradients = Model.ComputeGradients(input, target, LossFunction);

                    // Accumulate task gradients
                    if (batchGradients == null)
                    {
                        batchGradients = CloneVector(sampleGradients);
                    }
                    else
                    {
                        AccumulateGradients(batchGradients, sampleGradients);
                    }

                    // Compute sample task loss
                    var prediction = Model.Predict(input);
                    var predictionVector = ConversionsHelper.ConvertToVector<T, TOutput>(prediction);
                    var targetVector = ConversionsHelper.ConvertToVector<T, TOutput>(target);
                    var sampleLoss = LossFunction.CalculateLoss(predictionVector, targetVector);
                    batchTaskLoss = NumOps.Add(batchTaskLoss, sampleLoss);

                    // Compute distillation loss if we have a teacher model
                    if (hasTeacher && _lwfStrategy != null)
                    {
                        var studentOutput = prediction;
                        var teacherOutput = teacherModel!.Predict(input);

                        // Convert outputs to vectors for distillation loss computation
                        var studentVector = ConvertToVector(studentOutput);
                        var teacherVector = ConvertToVector(teacherOutput);

                        if (studentVector != null && teacherVector != null)
                        {
                            var distillLoss = _lwfStrategy.ComputeDistillationLoss(teacherVector, studentVector);
                            distillLoss = NumOps.Multiply(distillLoss, NumOps.FromDouble(currentDistillWeight));
                            batchDistillLoss = NumOps.Add(batchDistillLoss, distillLoss);
                        }
                    }
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

                // Adjust gradients using strategy
                var adjustedGradients = Strategy.AdjustGradients(batchGradients);

                // Apply gradients to update model
                Model.ApplyGradients(adjustedGradients, learningRate);
                totalGradientUpdates++;

                // Track losses
                batchTaskLoss = NumOps.Divide(batchTaskLoss, NumOps.FromDouble(actualBatchSize));
                batchDistillLoss = NumOps.Divide(batchDistillLoss, NumOps.FromDouble(actualBatchSize));
                epochTaskLoss = NumOps.Add(epochTaskLoss, batchTaskLoss);
                epochDistillLoss = NumOps.Add(epochDistillLoss, batchDistillLoss);
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
                epochDistillLoss = NumOps.Divide(epochDistillLoss, NumOps.FromDouble(batchCount));
            }

            var totalEpochLoss = NumOps.Add(epochTaskLoss, epochDistillLoss);
            lossHistory.Add(totalEpochLoss);
            distillationLossHistory.Add(epochDistillLoss);

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
            regularizationLossHistory: new Vector<T>(distillationLossHistory.ToArray()))
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

    /// <summary>
    /// Converts an output to a Vector for distillation loss computation.
    /// </summary>
    private Vector<T>? ConvertToVector(TOutput output)
    {
        // Handle the case where TOutput is already Vector<T>
        if (output is Vector<T> vector)
            return vector;

        // Handle the case where TOutput is an array of T
        if (output is T[] array)
            return new Vector<T>(array);

        // Handle other numeric arrays by converting elements
        if (output is Array arr)
        {
            var length = arr.Length;
            var result = new T[length];
            for (int i = 0; i < length; i++)
            {
                var element = arr.GetValue(i);
                if (element != null)
                {
                    result[i] = NumOps.FromDouble(Convert.ToDouble(element));
                }
            }
            return new Vector<T>(result);
        }

        // If it's a single value, create a single-element vector
        if (output != null)
        {
            try
            {
                var value = NumOps.FromDouble(Convert.ToDouble(output));
                return new Vector<T>([value]);
            }
            catch
            {
                // Conversion failed
            }
        }

        return null;
    }

    #endregion
}
