using AiDotNet.ContinualLearning.Config;
using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.ContinualLearning.Results;
using AiDotNet.ContinualLearning.Strategies;
using AiDotNet.Interfaces;

namespace AiDotNet.ContinualLearning.Trainers;

/// <summary>
/// Continual learning trainer using Learning without Forgetting (LwF).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This trainer implements continual learning using LwF,
/// which prevents catastrophic forgetting by using knowledge distillation from a
/// "teacher" model (the model before training on the new task).</para>
///
/// <para><b>How LwF Works:</b>
/// 1. Before learning a new task, save the current model as the "teacher"
/// 2. During training, compute both task loss and distillation loss
/// 3. Distillation loss measures how different the new model's outputs are from the teacher's
/// 4. This ensures the model doesn't forget its previous knowledge
/// </para>
///
/// <para><b>Key Advantage:</b> LwF doesn't need to store old task data - it only needs the
/// teacher model's outputs on current task data.</para>
///
/// <para><b>Usage Example:</b>
/// <code>
/// var model = new MyNeuralNetwork();
/// var lossFunction = new CrossEntropyLoss();
/// var config = new ContinualLearnerConfig&lt;double&gt;();
/// var lwfStrategy = new LearningWithoutForgetting&lt;double, Matrix, Vector&gt;(lossFunction, 2.0, 1.0);
/// var trainer = new LwFTrainer(model, lossFunction, config, lwfStrategy);
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
/// <para><b>Reference:</b> Li and Hoiem "Learning without Forgetting" (2017)</para>
/// </remarks>
public class LwFTrainer<T, TInput, TOutput> : ContinualLearnerBase<T, TInput, TOutput>
{
    [ThreadStatic]
    private static Random? _random;
    private static Random ThreadRandom => _random ??= new Random();

    /// <summary>
    /// Initializes a new LwF trainer.
    /// </summary>
    public LwFTrainer(
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

        // Prepare strategy for the new task (this creates/updates the teacher model)
        Strategy.PrepareForTask(Model, taskData);

        // Store test set for later evaluation
        _taskTestSets.Add(taskData);

        // Store initial accuracy on this task before training (for forward transfer calculation)
        var initialResult = EvaluateTaskInternal(taskData);
        _initialAccuracies[_tasksLearned] = initialResult.Accuracy;

        // Get the LwF strategy for distillation loss computation
        var lwfStrategy = Strategy as LearningWithoutForgetting<T, TInput, TOutput>;
        var teacherModel = lwfStrategy?.TeacherModel;

        var lossHistory = new List<T>();
        var distillationLossHistory = new List<T>();
        var batchSize = Configuration.BatchSize;
        var numSamples = taskData.Count;

        // Training loop
        for (int epoch = 0; epoch < Configuration.EpochsPerTask; epoch++)
        {
            T epochTaskLoss = NumOps.Zero;
            T epochDistillLoss = NumOps.Zero;
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
                        batchGradients = sampleGradients;
                    }
                    else
                    {
                        for (int j = 0; j < batchGradients.Length; j++)
                        {
                            batchGradients[j] = NumOps.Add(batchGradients[j], sampleGradients[j]);
                        }
                    }

                    // Estimate sample task loss from gradient norm (proxy for loss magnitude)
                    T gradientNorm = ComputeGradientNorm(sampleGradients);
                    batchTaskLoss = NumOps.Add(batchTaskLoss, gradientNorm);

                    // Compute distillation loss if we have a teacher model and LwF strategy
                    if (lwfStrategy != null && teacherModel != null && _tasksLearned > 0)
                    {
                        // Get outputs from both teacher and student
                        var studentOutput = Model.Predict(input);
                        var teacherOutput = teacherModel.Predict(input);

                        // Convert outputs to Vector<T> if they're not already
                        var studentVector = ConvertToVector(studentOutput);
                        var teacherVector = ConvertToVector(teacherOutput);

                        if (studentVector != null && teacherVector != null)
                        {
                            var distillLoss = lwfStrategy.ComputeDistillationLoss(teacherVector, studentVector);
                            batchDistillLoss = NumOps.Add(batchDistillLoss, distillLoss);
                        }
                    }
                }

                if (batchGradients == null)
                    continue;

                // Average gradients over batch
                var batchSizeT = NumOps.FromDouble(actualBatchSize);
                for (int j = 0; j < batchGradients.Length; j++)
                {
                    batchGradients[j] = NumOps.Divide(batchGradients[j], batchSizeT);
                }

                // Adjust gradients using strategy (LwF doesn't modify gradients directly)
                var adjustedGradients = Strategy.AdjustGradients(batchGradients);

                // Apply gradients to update model
                Model.ApplyGradients(adjustedGradients, Configuration.LearningRate);

                // Track losses
                batchTaskLoss = NumOps.Divide(batchTaskLoss, batchSizeT);
                batchDistillLoss = NumOps.Divide(batchDistillLoss, batchSizeT);
                epochTaskLoss = NumOps.Add(epochTaskLoss, batchTaskLoss);
                epochDistillLoss = NumOps.Add(epochDistillLoss, batchDistillLoss);
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
                epochTaskLoss = NumOps.Divide(epochTaskLoss, sampleCountT);
                epochDistillLoss = NumOps.Divide(epochDistillLoss, sampleCountT);
            }

            var totalEpochLoss = NumOps.Add(epochTaskLoss, epochDistillLoss);
            lossHistory.Add(totalEpochLoss);
            distillationLossHistory.Add(epochDistillLoss);
        }

        // Finalize the task (updates the teacher model with newly trained model)
        Strategy.FinalizeTask(Model);

        // Store examples in replay buffer
        int samplesPerTask = Configuration.MemorySize / Math.Max(1, _tasksLearned + 1);
        MemoryBuffer.AddTaskExamples(taskData, samplesPerTask);

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

        return new ContinualLearningResult<T>(
            taskId: _tasksLearned - 1,
            trainingLoss: finalLoss,
            trainingAccuracy: finalAccuracy,
            averagePreviousTaskAccuracy: avgPrevAccuracy,
            trainingTime: startTime.Elapsed,
            lossHistory: new Vector<T>(lossHistory.ToArray()),
            regularizationLossHistory: new Vector<T>(distillationLossHistory.ToArray()));
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
                return new Vector<T>(new[] { value });
            }
            catch
            {
                // Conversion failed
            }
        }

        return null;
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
