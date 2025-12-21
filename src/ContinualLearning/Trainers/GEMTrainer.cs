using AiDotNet.ContinualLearning.Config;
using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.ContinualLearning.Results;
using AiDotNet.ContinualLearning.Strategies;
using AiDotNet.Interfaces;

namespace AiDotNet.ContinualLearning.Trainers;

/// <summary>
/// Continual learning trainer using Gradient Episodic Memory (GEM).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This trainer implements continual learning using GEM,
/// which prevents catastrophic forgetting by projecting gradients to avoid hurting
/// performance on previous tasks.</para>
///
/// <para><b>How GEM Works:</b>
/// 1. Store examples from each completed task in episodic memory
/// 2. When training on a new task, check if gradients would hurt old tasks
/// 3. If yes, project gradients to a direction that doesn't increase old task loss
/// 4. This ensures new learning never hurts previous performance
/// </para>
///
/// <para><b>Usage Example:</b>
/// <code>
/// var model = new MyNeuralNetwork();
/// var lossFunction = new CrossEntropyLoss();
/// var config = new ContinualLearnerConfig&lt;double&gt;();
/// var gemStrategy = new GradientEpisodicMemory&lt;double, Matrix, Vector&gt;(lossFunction, 256);
/// var trainer = new GEMTrainer(model, lossFunction, config, gemStrategy);
///
/// // Learn tasks sequentially
/// var task1Result = trainer.LearnTask(task1Data);
/// var task2Result = trainer.LearnTask(task2Data);
///
/// // Evaluate on all tasks
/// var evalResult = trainer.EvaluateAllTasks();
/// </code>
/// </para>
///
/// <para><b>Reference:</b> Lopez-Paz and Ranzato "Gradient Episodic Memory for Continual Learning" (2017)</para>
/// </remarks>
public class GEMTrainer<T, TInput, TOutput> : ContinualLearnerBase<T, TInput, TOutput>
{
    [ThreadStatic]
    private static Random? _random;
    private static Random ThreadRandom => _random ??= new Random();

    /// <summary>
    /// Initializes a new GEM trainer.
    /// </summary>
    public GEMTrainer(
        IFullModel<T, TInput, TOutput> model,
        ILossFunction<T> lossFunction,
        IContinualLearnerConfig<T> config,
        IContinualLearningStrategy<T, TInput, TOutput> strategy)
        : base(model, lossFunction, config, strategy)
    {
    }

    /// <inheritdoc/>
    public override ContinualLearningResult<T> LearnTask(IDataset<T, TInput, TOutput> taskData)
    {
        if (taskData == null)
            throw new ArgumentNullException(nameof(taskData));

        var startTime = System.Diagnostics.Stopwatch.StartNew();

        // Prepare strategy for the new task
        Strategy.PrepareForTask(Model, taskData);

        // Store test set for later evaluation
        _taskTestSets.Add(taskData);

        // Store initial accuracy on this task before training (for forward transfer calculation)
        var initialResult = EvaluateTaskInternal(taskData);
        _initialAccuracies[_tasksLearned] = initialResult.Accuracy;

        var lossHistory = new List<T>();
        var batchSize = Configuration.BatchSize;
        var numSamples = taskData.Count;

        // Accumulated gradient for computing task gradient (used by GEM)
        Vector<T>? accumulatedTaskGradient = null;
        int gradientSampleCount = 0;

        // Training loop
        for (int epoch = 0; epoch < Configuration.EpochsPerTask; epoch++)
        {
            T epochLoss = NumOps.Zero;
            int sampleCount = 0;

            // Shuffle indices for this epoch
            var indices = Enumerable.Range(0, numSamples).OrderBy(_ => ThreadRandom.Next()).ToList();

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
                        batchGradients = sampleGradients;
                    }
                    else
                    {
                        for (int j = 0; j < batchGradients.Length; j++)
                        {
                            batchGradients[j] = NumOps.Add(batchGradients[j], sampleGradients[j]);
                        }
                    }

                    // Also accumulate for task gradient (for GEM constraint storage)
                    if (accumulatedTaskGradient == null)
                    {
                        accumulatedTaskGradient = new Vector<T>(sampleGradients.Length);
                        for (int j = 0; j < sampleGradients.Length; j++)
                        {
                            accumulatedTaskGradient[j] = sampleGradients[j];
                        }
                    }
                    else
                    {
                        for (int j = 0; j < accumulatedTaskGradient.Length; j++)
                        {
                            accumulatedTaskGradient[j] = NumOps.Add(accumulatedTaskGradient[j], sampleGradients[j]);
                        }
                    }
                    gradientSampleCount++;

                    // Estimate sample loss from gradient norm (proxy for loss magnitude)
                    T gradientNorm = ComputeGradientNorm(sampleGradients);
                    batchLoss = NumOps.Add(batchLoss, gradientNorm);
                }

                if (batchGradients == null)
                    continue;

                // Average gradients over batch
                var batchSizeT = NumOps.FromDouble(actualBatchSize);
                for (int j = 0; j < batchGradients.Length; j++)
                {
                    batchGradients[j] = NumOps.Divide(batchGradients[j], batchSizeT);
                }

                // Adjust gradients using GEM strategy (project to avoid hurting previous tasks)
                var adjustedGradients = Strategy.AdjustGradients(batchGradients);

                // Apply gradients to update model
                Model.ApplyGradients(adjustedGradients, Configuration.LearningRate);

                // Track losses
                batchLoss = NumOps.Divide(batchLoss, batchSizeT);
                epochLoss = NumOps.Add(epochLoss, batchLoss);
                sampleCount++;
            }

            // Mix in experience replay samples (if we have previous task examples)
            if (MemoryBuffer.Count > 0)
            {
                int replaySamples = Math.Min(batchSize, MemoryBuffer.Count);
                var replayBatch = MemoryBuffer.SampleBatch(replaySamples);

                if (replayBatch.Count > 0)
                {
                    Vector<T>? replayGradients = null;

                    foreach (var dataPoint in replayBatch)
                    {
                        var sampleGradients = Model.ComputeGradients(dataPoint.Input, dataPoint.Output, LossFunction);

                        if (replayGradients == null)
                        {
                            replayGradients = sampleGradients;
                        }
                        else
                        {
                            for (int j = 0; j < replayGradients.Length; j++)
                            {
                                replayGradients[j] = NumOps.Add(replayGradients[j], sampleGradients[j]);
                            }
                        }
                    }

                    if (replayGradients != null)
                    {
                        // Average replay gradients
                        var replayBatchSizeT = NumOps.FromDouble(replayBatch.Count);
                        for (int j = 0; j < replayGradients.Length; j++)
                        {
                            replayGradients[j] = NumOps.Divide(replayGradients[j], replayBatchSizeT);
                        }

                        // Scale replay gradients (use half the learning rate for replay)
                        var replayLr = NumOps.Divide(Configuration.LearningRate, NumOps.FromDouble(2.0));
                        var adjustedReplayGradients = Strategy.AdjustGradients(replayGradients);
                        Model.ApplyGradients(adjustedReplayGradients, replayLr);
                    }
                }
            }

            // Average epoch losses
            if (sampleCount > 0)
            {
                var sampleCountT = NumOps.FromDouble(sampleCount);
                epochLoss = NumOps.Divide(epochLoss, sampleCountT);
            }

            lossHistory.Add(epochLoss);
        }

        // Store the average task gradient for GEM constraints
        if (accumulatedTaskGradient != null && gradientSampleCount > 0)
        {
            var gradientCountT = NumOps.FromDouble(gradientSampleCount);
            for (int j = 0; j < accumulatedTaskGradient.Length; j++)
            {
                accumulatedTaskGradient[j] = NumOps.Divide(accumulatedTaskGradient[j], gradientCountT);
            }

            // If strategy is GradientEpisodicMemory, store the computed gradient
            if (Strategy is GradientEpisodicMemory<T, TInput, TOutput> gemStrategy)
            {
                gemStrategy.StoreTaskGradient(accumulatedTaskGradient);
            }
        }

        // Finalize the task (stores placeholder gradient if not already stored)
        Strategy.FinalizeTask(Model);

        // Store examples in both the trainer's replay buffer and GEM's memory
        int samplesPerTask = Configuration.MemorySize / Math.Max(1, _tasksLearned + 1);
        MemoryBuffer.AddTaskExamples(taskData, samplesPerTask);

        // Also store in GEM's episodic memory if applicable
        if (Strategy is GradientEpisodicMemory<T, TInput, TOutput> gemMemory)
        {
            gemMemory.StoreTaskExamples(taskData);
        }

        // Evaluate final performance on this task
        var finalResult = EvaluateTaskInternal(taskData);
        var finalLoss = lossHistory.Count > 0 ? lossHistory[lossHistory.Count - 1] : NumOps.Zero;
        var finalAccuracy = finalResult.Accuracy;

        // Compute average accuracy on previous tasks
        T avgPrevAccuracy = NumOps.Zero;
        if (_tasksLearned > 0)
        {
            for (int i = 0; i < _tasksLearned; i++)
            {
                var result = EvaluateTask(i, _taskTestSets[i]);
                avgPrevAccuracy = NumOps.Add(avgPrevAccuracy, result.Accuracy);
            }
            avgPrevAccuracy = NumOps.Divide(avgPrevAccuracy, NumOps.FromDouble(_tasksLearned));
        }

        _tasksLearned++;
        startTime.Stop();

        // GEM doesn't use regularization loss in the same way as EWC
        var emptyRegLossHistory = new Vector<T>(new T[lossHistory.Count]);
        for (int i = 0; i < emptyRegLossHistory.Length; i++)
        {
            emptyRegLossHistory[i] = NumOps.Zero;
        }

        return new ContinualLearningResult<T>(
            taskId: _tasksLearned - 1,
            trainingLoss: finalLoss,
            trainingAccuracy: finalAccuracy,
            averagePreviousTaskAccuracy: avgPrevAccuracy,
            trainingTime: startTime.Elapsed,
            lossHistory: new Vector<T>(lossHistory.ToArray()),
            regularizationLossHistory: emptyRegLossHistory);
    }

    /// <summary>
    /// Computes the L2 norm of a gradient vector (used as proxy for loss magnitude).
    /// </summary>
    private T ComputeGradientNorm(Vector<T> gradients)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < gradients.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(gradients[i], gradients[i]));
        }
        return NumOps.Sqrt(sum);
    }

    /// <summary>
    /// Internal evaluation method that works on any dataset.
    /// </summary>
    private TaskEvaluationResult<T> EvaluateTaskInternal(IDataset<T, TInput, TOutput> testData)
    {
        if (testData.Count == 0)
        {
            return new TaskEvaluationResult<T>(0, NumOps.Zero, NumOps.Zero);
        }

        int correctCount = 0;
        T totalLoss = NumOps.Zero;

        for (int i = 0; i < testData.Count; i++)
        {
            var input = testData.GetInput(i);
            var target = testData.GetOutput(i);
            var prediction = Model.Predict(input);

            // Compute loss using gradient norm as proxy
            var gradients = Model.ComputeGradients(input, target, LossFunction);
            var sampleLoss = ComputeGradientNorm(gradients);
            totalLoss = NumOps.Add(totalLoss, sampleLoss);

            // Check if prediction is correct (using gradient magnitude as proxy - lower is better)
            // A correct prediction typically has very small gradients
            if (Convert.ToDouble(sampleLoss) < 0.1)
            {
                correctCount++;
            }
        }

        var accuracy = NumOps.FromDouble((double)correctCount / testData.Count);
        var avgLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(testData.Count));

        return new TaskEvaluationResult<T>(0, accuracy, avgLoss);
    }

    /// <inheritdoc/>
    public override TaskEvaluationResult<T> EvaluateTask(int taskId, IDataset<T, TInput, TOutput> testData)
    {
        if (taskId < 0 || taskId >= _tasksLearned)
            throw new ArgumentException($"Invalid task ID: {taskId}", nameof(taskId));
        if (testData == null)
            throw new ArgumentNullException(nameof(testData));

        var result = EvaluateTaskInternal(testData);
        return new TaskEvaluationResult<T>(taskId, result.Accuracy, result.Loss);
    }
}
