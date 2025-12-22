using System.Diagnostics;
using AiDotNet.Interfaces;
using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.ContinualLearning.Config;
using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.ContinualLearning.Memory;
using AiDotNet.ContinualLearning.Results;
using AiDotNet.Helpers;
using Newtonsoft.Json.Linq;

namespace AiDotNet.ContinualLearning.Trainers;

/// <summary>
/// Base class for continual learning trainers that provides common functionality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Algorithm Implementers:</b> To create a new continual learning algorithm:</para>
/// <list type="number">
/// <item><description>Extend this base class</description></item>
/// <item><description>Implement the abstract <see cref="TrainOnTask"/> method with your algorithm-specific training logic</description></item>
/// <item><description>Pass an <see cref="IContinualLearningStrategy{T,TInput,TOutput}"/> that handles algorithm-specific regularization</description></item>
/// <item><description>All shared functionality (evaluation, memory management, events) is handled automatically</description></item>
/// </list>
///
/// <para><b>Common Patterns:</b></para>
/// <list type="bullet">
/// <item><description><b>Regularization-based (EWC, SI):</b> Override TrainOnTask to add regularization loss</description></item>
/// <item><description><b>Replay-based (GEM, A-GEM):</b> Use MemoryBuffer and override gradient computation</description></item>
/// <item><description><b>Distillation-based (LwF):</b> Use the strategy's teacher model for knowledge transfer</description></item>
/// </list>
///
/// <para><b>Reference:</b> De Lange et al. "A Continual Learning Survey: Defying Forgetting" (2021)</para>
/// </remarks>
public abstract class ContinualLearnerBase<T, TInput, TOutput> : IContinualLearner<T, TInput, TOutput>
{
    #region Protected Fields

    /// <summary>
    /// The underlying model being trained.
    /// </summary>
    protected readonly IFullModel<T, TInput, TOutput> Model;

    /// <summary>
    /// The loss function used for training.
    /// </summary>
    protected readonly ILossFunction<T> LossFunction;

    /// <summary>
    /// Configuration for the continual learner.
    /// </summary>
    protected readonly IContinualLearnerConfig<T> Configuration;

    /// <summary>
    /// Strategy for preventing catastrophic forgetting.
    /// </summary>
    protected readonly IContinualLearningStrategy<T, TInput, TOutput> Strategy;

    /// <summary>
    /// Memory buffer for experience replay.
    /// </summary>
    protected readonly ExperienceReplayBuffer<T, TInput, TOutput> MemoryBuffer;

    /// <summary>
    /// Numeric operations for generic type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Number of tasks successfully learned.
    /// </summary>
    protected int _tasksLearned;

    /// <summary>
    /// Test sets for each learned task (for evaluation).
    /// </summary>
    protected readonly List<IDataset<T, TInput, TOutput>> _taskTestSets;

    /// <summary>
    /// Initial accuracy on each task before it was learned (for forward transfer).
    /// </summary>
    protected readonly Dictionary<int, T> _initialAccuracies;

    /// <summary>
    /// Accuracy on each task right after it was learned (for backward transfer).
    /// </summary>
    protected readonly Dictionary<int, T> _peakAccuracies;

    /// <summary>
    /// Training history for all tasks.
    /// </summary>
    protected readonly List<ContinualLearningResult<T>> _trainingHistory;

    /// <summary>
    /// Whether the learner is currently training.
    /// </summary>
    protected volatile bool _isTraining;

    /// <summary>
    /// Synchronization lock for thread safety.
    /// </summary>
    protected readonly object _lock = new object();

    #endregion

    #region IContinualLearner Properties

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> BaseModel => Model;

    /// <inheritdoc/>
    public IContinualLearnerConfig<T> Config => Configuration;

    /// <inheritdoc/>
    public int TasksLearned => _tasksLearned;

    /// <inheritdoc/>
    public bool IsTraining => _isTraining;

    /// <inheritdoc/>
    public long MemoryUsageBytes
    {
        get
        {
            long total = MemoryBuffer.EstimatedMemoryBytes;
            total += Strategy.MemoryUsageBytes;
            // Estimate model size (rough approximation)
            total += Model.GetParameters().Length * sizeof(double);
            return total;
        }
    }

    #endregion

    #region Events

    /// <inheritdoc/>
    public event EventHandler<TaskEventArgs>? TaskStarted;

    /// <inheritdoc/>
    public event EventHandler<TaskCompletedEventArgs<T>>? TaskCompleted;

    /// <inheritdoc/>
    public event EventHandler<EpochEventArgs<T>>? EpochCompleted;

    /// <summary>
    /// Raises the TaskStarted event.
    /// </summary>
    protected virtual void OnTaskStarted(int taskId, int sampleCount)
    {
        TaskStarted?.Invoke(this, new TaskEventArgs(taskId, sampleCount));
    }

    /// <summary>
    /// Raises the TaskCompleted event.
    /// </summary>
    protected virtual void OnTaskCompleted(int taskId, int sampleCount, ContinualLearningResult<T> result)
    {
        TaskCompleted?.Invoke(this, new TaskCompletedEventArgs<T>(taskId, sampleCount, result));
    }

    /// <summary>
    /// Raises the EpochCompleted event.
    /// </summary>
    protected virtual void OnEpochCompleted(int taskId, int epoch, int totalEpochs, T loss, T? validationLoss = default)
    {
        EpochCompleted?.Invoke(this, new EpochEventArgs<T>(taskId, epoch, totalEpochs, loss, validationLoss));
    }

    #endregion

    #region Constructor

    /// <summary>
    /// Initializes a new continual learner.
    /// </summary>
    /// <param name="model">The model to train.</param>
    /// <param name="lossFunction">The loss function for training.</param>
    /// <param name="config">Configuration for continual learning.</param>
    /// <param name="strategy">Strategy for preventing forgetting.</param>
    /// <exception cref="ArgumentNullException">Thrown if any parameter is null.</exception>
    /// <exception cref="ArgumentException">Thrown if configuration is invalid.</exception>
    protected ContinualLearnerBase(
        IFullModel<T, TInput, TOutput> model,
        ILossFunction<T> lossFunction,
        IContinualLearnerConfig<T> config,
        IContinualLearningStrategy<T, TInput, TOutput> strategy)
    {
        Model = model ?? throw new ArgumentNullException(nameof(model));
        LossFunction = lossFunction ?? throw new ArgumentNullException(nameof(lossFunction));
        Configuration = config ?? throw new ArgumentNullException(nameof(config));
        Strategy = strategy ?? throw new ArgumentNullException(nameof(strategy));

        if (!config.IsValid())
            throw new ArgumentException("Configuration validation failed", nameof(config));

        if (!strategy.IsCompatibleWith(model))
        {
            var reason = strategy.GetIncompatibilityReason(model) ?? "Unknown incompatibility";
            throw new ArgumentException($"Strategy is not compatible with model: {reason}", nameof(strategy));
        }

        MemoryBuffer = new ExperienceReplayBuffer<T, TInput, TOutput>(
            config.MemorySize,
            MemorySamplingStrategy.ClassBalanced,
            ReplaySamplingStrategy.TaskBalanced,
            config.RandomSeed);

        _tasksLearned = 0;
        _taskTestSets = new List<IDataset<T, TInput, TOutput>>();
        _initialAccuracies = new Dictionary<int, T>();
        _peakAccuracies = new Dictionary<int, T>();
        _trainingHistory = new List<ContinualLearningResult<T>>();
        _isTraining = false;
    }

    #endregion

    #region IContinualLearner Methods

    /// <inheritdoc/>
    public ContinualLearningResult<T> LearnTask(IDataset<T, TInput, TOutput> taskData)
    {
        return LearnTask(taskData, validationData: null, earlyStoppingPatience: null);
    }

    /// <inheritdoc/>
    public ContinualLearningResult<T> LearnTask(
        IDataset<T, TInput, TOutput> taskData,
        IDataset<T, TInput, TOutput>? validationData,
        int? earlyStoppingPatience = null)
    {
        if (taskData == null)
            throw new ArgumentNullException(nameof(taskData));

        lock (_lock)
        {
            if (_isTraining)
                throw new InvalidOperationException("Another training task is already in progress");
            _isTraining = true;
        }

        try
        {
            var stopwatch = Stopwatch.StartNew();
            int taskId = _tasksLearned;

            // Record initial accuracy (for forward transfer calculation)
            if (validationData != null)
            {
                var initialResult = EvaluateOnDataset(validationData);
                _initialAccuracies[taskId] = initialResult.Accuracy;
            }

            // Notify task started
            OnTaskStarted(taskId, taskData.Count);

            // Prepare strategy for new task
            Strategy.PrepareForTask(Model, taskData);

            // Perform training (implemented by derived classes)
            var result = TrainOnTask(
                taskData,
                validationData,
                earlyStoppingPatience ?? 5);

            // Finalize strategy after training
            Strategy.FinalizeTask(Model);

            // Store test set for future evaluation
            if (validationData != null)
            {
                _taskTestSets.Add(validationData);
            }
            else
            {
                _taskTestSets.Add(taskData); // Use training data if no validation
            }

            // Store examples in memory buffer
            var memorySize = Configuration.MemorySize;
            if (memorySize > 0)
            {
                int samplesPerTask = Math.Max(1, memorySize / (_tasksLearned + 1));
                MemoryBuffer.AddTaskExamples(taskData, taskId, samplesPerTask);
            }

            // Record peak accuracy
            if (validationData != null)
            {
                var peakResult = EvaluateOnDataset(validationData);
                _peakAccuracies[taskId] = peakResult.Accuracy;
            }

            stopwatch.Stop();

            // Create final result with additional metrics
            var finalResult = new ContinualLearningResult<T>(
                taskId: taskId,
                trainingLoss: result.TrainingLoss,
                trainingAccuracy: result.TrainingAccuracy,
                averagePreviousTaskAccuracy: ComputeAveragePreviousTaskAccuracy(),
                trainingTime: stopwatch.Elapsed,
                lossHistory: result.LossHistory,
                regularizationLossHistory: result.RegularizationLossHistory)
            {
                ValidationLoss = result.ValidationLoss,
                ValidationAccuracy = result.ValidationAccuracy,
                Forgetting = ComputeCurrentForgetting(),
                ForwardTransfer = _initialAccuracies.TryGetValue(taskId, out var initAcc) ? initAcc : default,
                SampleCount = taskData.Count,
                PeakMemoryBytes = MemoryUsageBytes,
                GradientUpdates = result.GradientUpdates,
                EffectiveLearningRate = result.EffectiveLearningRate,
                StrategyMetrics = Strategy.GetMetrics()
            };

            // Save to history
            _trainingHistory.Add(finalResult);
            _tasksLearned++;

            // Notify task completed
            OnTaskCompleted(taskId, taskData.Count, finalResult);

            return finalResult;
        }
        finally
        {
            lock (_lock)
            {
                _isTraining = false;
            }
        }
    }

    /// <inheritdoc/>
    public virtual ContinualEvaluationResult<T> EvaluateAllTasks()
    {
        if (_tasksLearned == 0)
            throw new InvalidOperationException("No tasks have been learned yet");

        var stopwatch = Stopwatch.StartNew();

        var perTaskResults = new List<TaskEvaluationResult<T>>();
        var accuracies = new List<T>();
        var losses = new List<T>();

        for (int i = 0; i < _tasksLearned; i++)
        {
            var result = EvaluateTask(i, _taskTestSets[i]);
            perTaskResults.Add(result);
            accuracies.Add(result.Accuracy);
            losses.Add(result.Loss);
        }

        stopwatch.Stop();

        var accVector = new Vector<T>(accuracies.ToArray());
        var lossVector = new Vector<T>(losses.ToArray());

        // Compute aggregate metrics
        var avgAccuracy = NumOps.Divide(
            accVector.Sum(),
            NumOps.FromDouble(_tasksLearned));

        var avgLoss = NumOps.Divide(
            lossVector.Sum(),
            NumOps.FromDouble(_tasksLearned));

        // Backward transfer: BWT = (1/T-1) * sum_{i=1}^{T-1} (R_{T,i} - R_{i,i})
        T backwardTransfer = NumOps.Zero;
        if (_tasksLearned > 1)
        {
            for (int i = 0; i < _tasksLearned - 1; i++)
            {
                if (_peakAccuracies.TryGetValue(i, out var peakAcc))
                {
                    backwardTransfer = NumOps.Add(
                        backwardTransfer,
                        NumOps.Subtract(accuracies[i], peakAcc));
                }
            }
            backwardTransfer = NumOps.Divide(
                backwardTransfer,
                NumOps.FromDouble(_tasksLearned - 1));
        }

        // Forward transfer: FWT = average initial accuracy
        T forwardTransfer = NumOps.Zero;
        if (_initialAccuracies.Count > 0)
        {
            foreach (var acc in _initialAccuracies.Values)
            {
                forwardTransfer = NumOps.Add(forwardTransfer, acc);
            }
            forwardTransfer = NumOps.Divide(
                forwardTransfer,
                NumOps.FromDouble(_initialAccuracies.Count));
        }

        // Max forgetting
        T? maxForgetting = default;
        if (_peakAccuracies.Count > 0)
        {
            var forgettingValues = new List<T>();
            for (int i = 0; i < accuracies.Count && i < _peakAccuracies.Count; i++)
            {
                if (_peakAccuracies.TryGetValue(i, out var peak))
                {
                    forgettingValues.Add(NumOps.Subtract(peak, accuracies[i]));
                }
            }
            if (forgettingValues.Count > 0)
            {
                maxForgetting = forgettingValues.OrderByDescending(v => NumOps.ToDouble(v)).First();
            }
        }

        return new ContinualEvaluationResult<T>(
            taskAccuracies: accVector,
            taskLosses: lossVector,
            averageAccuracy: avgAccuracy,
            averageLoss: avgLoss,
            backwardTransfer: backwardTransfer,
            forwardTransfer: forwardTransfer,
            evaluationTime: stopwatch.Elapsed)
        {
            MaxForgetting = maxForgetting,
            PerTaskResults = perTaskResults
        };
    }

    /// <inheritdoc/>
    public virtual TaskEvaluationResult<T> EvaluateTask(int taskId, IDataset<T, TInput, TOutput> testData)
    {
        if (testData == null)
            throw new ArgumentNullException(nameof(testData));
        if (taskId < 0)
            throw new ArgumentOutOfRangeException(nameof(taskId), "Task ID must be non-negative");

        var stopwatch = Stopwatch.StartNew();

        T totalLoss = NumOps.Zero;
        int correct = 0;
        var predictions = new List<(int Predicted, int Actual)>();

        for (int i = 0; i < testData.Count; i++)
        {
            var input = testData.GetInput(i);
            var expected = testData.GetOutput(i);
            var predicted = Model.Predict(input);

            var loss = LossFunction.CalculateLoss(ConversionsHelper.ConvertToVector<T, TOutput>(predicted), ConversionsHelper.ConvertToVector<T, TOutput>(expected));
            totalLoss = NumOps.Add(totalLoss, loss);

            // Check if prediction is correct (implementation may vary by output type)
            if (IsPredictionCorrect(expected, predicted))
            {
                correct++;
                predictions.Add((GetClassLabel(predicted), GetClassLabel(expected)));
            }
            else
            {
                predictions.Add((GetClassLabel(predicted), GetClassLabel(expected)));
            }
        }

        stopwatch.Stop();

        var avgLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(testData.Count));
        var accuracy = NumOps.FromDouble((double)correct / testData.Count);

        // Build confusion matrix
        var confusionMatrix = predictions
            .GroupBy(p => (p.Predicted, p.Actual))
            .ToDictionary(g => g.Key, g => g.Count());

        return new TaskEvaluationResult<T>(taskId, accuracy, avgLoss)
        {
            SampleCount = testData.Count,
            CorrectCount = correct,
            EvaluationTime = stopwatch.Elapsed,
            ConfusionMatrix = confusionMatrix
        };
    }

    /// <inheritdoc/>
    public ContinualLearningResult<T>? GetTaskHistory(int taskId)
    {
        if (taskId < 0 || taskId >= _trainingHistory.Count)
            return null;
        return _trainingHistory[taskId];
    }

    /// <inheritdoc/>
    public IReadOnlyList<ContinualLearningResult<T>> GetAllHistory()
    {
        return _trainingHistory.AsReadOnly();
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<int, T> ComputeForgetting()
    {
        var forgetting = new Dictionary<int, T>();

        for (int i = 0; i < _tasksLearned; i++)
        {
            if (_peakAccuracies.TryGetValue(i, out var peakAcc) && i < _taskTestSets.Count)
            {
                var currentResult = EvaluateOnDataset(_taskTestSets[i]);
                forgetting[i] = NumOps.Subtract(peakAcc, currentResult.Accuracy);
            }
        }

        return forgetting;
    }

    /// <inheritdoc/>
    public virtual void Save(string directoryPath)
    {
        if (string.IsNullOrWhiteSpace(directoryPath))
            throw new ArgumentException("Directory path cannot be null or empty", nameof(directoryPath));

        var normalizedDir = Path.GetFullPath(directoryPath);
        Directory.CreateDirectory(normalizedDir);

        // Save model
        var modelPath = Path.GetFullPath(Path.Combine(normalizedDir, "model.bin"));
        ValidatePathWithinDirectory(modelPath, normalizedDir);
        Model.SaveModel(modelPath);

        // Save strategy
        var strategyPath = Path.GetFullPath(Path.Combine(normalizedDir, "strategy.bin"));
        ValidatePathWithinDirectory(strategyPath, normalizedDir);
        Strategy.Save(strategyPath);

        // Save metadata
        var metadataPath = Path.GetFullPath(Path.Combine(normalizedDir, "metadata.json"));
        ValidatePathWithinDirectory(metadataPath, normalizedDir);

        var metadata = new JObject
        {
            ["TasksLearned"] = _tasksLearned,
            ["MemorySize"] = Configuration.MemorySize,
            ["StrategyName"] = Strategy.Name,
            ["MemoryBufferCount"] = MemoryBuffer.Count,
            ["InitialAccuracies"] = JObject.FromObject(_initialAccuracies.ToDictionary(
                kv => kv.Key.ToString(),
                kv => NumOps.ToDouble(kv.Value))),
            ["PeakAccuracies"] = JObject.FromObject(_peakAccuracies.ToDictionary(
                kv => kv.Key.ToString(),
                kv => NumOps.ToDouble(kv.Value)))
        };

        File.WriteAllText(metadataPath, metadata.ToString());
    }

    /// <inheritdoc/>
    public virtual void Load(string directoryPath)
    {
        if (string.IsNullOrWhiteSpace(directoryPath))
            throw new ArgumentException("Directory path cannot be null or empty", nameof(directoryPath));

        var normalizedDir = Path.GetFullPath(directoryPath);
        if (!Directory.Exists(normalizedDir))
            throw new DirectoryNotFoundException($"Directory not found: {normalizedDir}");

        // Load model
        var modelPath = Path.GetFullPath(Path.Combine(normalizedDir, "model.bin"));
        ValidatePathWithinDirectory(modelPath, normalizedDir);
        Model.LoadModel(modelPath);

        // Load strategy
        var strategyPath = Path.GetFullPath(Path.Combine(normalizedDir, "strategy.bin"));
        ValidatePathWithinDirectory(strategyPath, normalizedDir);
        if (File.Exists(strategyPath))
        {
            Strategy.Load(strategyPath);
        }

        // Load metadata
        var metadataPath = Path.GetFullPath(Path.Combine(normalizedDir, "metadata.json"));
        ValidatePathWithinDirectory(metadataPath, normalizedDir);
        if (File.Exists(metadataPath))
        {
            var json = File.ReadAllText(metadataPath);
            var metadata = JObject.Parse(json);

            if (metadata.TryGetValue("TasksLearned", out var tasksToken))
            {
                _tasksLearned = tasksToken.Value<int>();
            }

            if (metadata.TryGetValue("InitialAccuracies", out var initialAccToken) && initialAccToken is JObject initialObj)
            {
                foreach (var prop in initialObj.Properties())
                {
                    if (int.TryParse(prop.Name, out var taskId))
                    {
                        _initialAccuracies[taskId] = NumOps.FromDouble(prop.Value.Value<double>());
                    }
                }
            }

            if (metadata.TryGetValue("PeakAccuracies", out var peakAccToken) && peakAccToken is JObject peakObj)
            {
                foreach (var prop in peakObj.Properties())
                {
                    if (int.TryParse(prop.Name, out var taskId))
                    {
                        _peakAccuracies[taskId] = NumOps.FromDouble(prop.Value.Value<double>());
                    }
                }
            }
        }
    }

    /// <inheritdoc/>
    public virtual void Reset()
    {
        lock (_lock)
        {
            if (_isTraining)
                throw new InvalidOperationException("Cannot reset while training is in progress");

            _tasksLearned = 0;
            MemoryBuffer.Clear();
            _taskTestSets.Clear();
            _initialAccuracies.Clear();
            _peakAccuracies.Clear();
            _trainingHistory.Clear();
            Strategy.Reset();
        }
    }

    #endregion

    #region Abstract Methods

    /// <summary>
    /// Performs the actual training on a task. Implemented by derived classes.
    /// </summary>
    /// <param name="taskData">The training data for the task.</param>
    /// <param name="validationData">Optional validation data.</param>
    /// <param name="earlyStoppingPatience">Number of epochs without improvement before stopping.</param>
    /// <returns>Training result with metrics.</returns>
    /// <remarks>
    /// <para>Derived classes should:
    /// <list type="bullet">
    /// <item><description>Iterate through epochs</description></item>
    /// <item><description>Call <see cref="OnEpochCompleted"/> after each epoch</description></item>
    /// <item><description>Use <see cref="Strategy"/> for regularization</description></item>
    /// <item><description>Optionally replay from <see cref="MemoryBuffer"/></description></item>
    /// </list>
    /// </para>
    /// </remarks>
    protected abstract ContinualLearningResult<T> TrainOnTask(
        IDataset<T, TInput, TOutput> taskData,
        IDataset<T, TInput, TOutput>? validationData,
        int earlyStoppingPatience);

    #endregion

    #region Protected Helper Methods

    /// <summary>
    /// Evaluates the model on a dataset and returns simple metrics.
    /// </summary>
    protected virtual (T Accuracy, T Loss) EvaluateOnDataset(IDataset<T, TInput, TOutput> data)
    {
        T totalLoss = NumOps.Zero;
        int correct = 0;

        for (int i = 0; i < data.Count; i++)
        {
            var input = data.GetInput(i);
            var expected = data.GetOutput(i);
            var predicted = Model.Predict(input);

            totalLoss = NumOps.Add(totalLoss, LossFunction.CalculateLoss(ConversionsHelper.ConvertToVector<T, TOutput>(predicted), ConversionsHelper.ConvertToVector<T, TOutput>(expected)));

            if (IsPredictionCorrect(expected, predicted))
                correct++;
        }

        var avgLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(data.Count));
        var accuracy = NumOps.FromDouble((double)correct / data.Count);

        return (accuracy, avgLoss);
    }

    /// <summary>
    /// Determines if a prediction is correct. Override for custom comparison logic.
    /// </summary>
    protected virtual bool IsPredictionCorrect(TOutput expected, TOutput predicted)
    {
        // Default: use equality comparison
        return EqualityComparer<TOutput>.Default.Equals(expected, predicted);
    }

    /// <summary>
    /// Gets the class label from an output. Override for custom extraction logic.
    /// </summary>
    protected virtual int GetClassLabel(TOutput output)
    {
        // Default: use hash code as label (override for proper implementation)
        return output?.GetHashCode() ?? 0;
    }

    /// <summary>
    /// Computes the average accuracy on all previously learned tasks.
    /// </summary>
    protected T ComputeAveragePreviousTaskAccuracy()
    {
        if (_tasksLearned == 0)
            return NumOps.Zero;

        T sum = NumOps.Zero;
        int count = 0;

        for (int i = 0; i < _tasksLearned && i < _taskTestSets.Count; i++)
        {
            var result = EvaluateOnDataset(_taskTestSets[i]);
            sum = NumOps.Add(sum, result.Accuracy);
            count++;
        }

        return count > 0
            ? NumOps.Divide(sum, NumOps.FromDouble(count))
            : NumOps.Zero;
    }

    /// <summary>
    /// Computes the current forgetting averaged across all tasks.
    /// </summary>
    protected T? ComputeCurrentForgetting()
    {
        if (_peakAccuracies.Count == 0)
            return default;

        T sum = NumOps.Zero;
        int count = 0;

        foreach (var kvp in _peakAccuracies)
        {
            if (kvp.Key < _taskTestSets.Count)
            {
                var current = EvaluateOnDataset(_taskTestSets[kvp.Key]);
                sum = NumOps.Add(sum, NumOps.Subtract(kvp.Value, current.Accuracy));
                count++;
            }
        }

        return count > 0
            ? NumOps.Divide(sum, NumOps.FromDouble(count))
            : NumOps.Zero;
    }

    /// <summary>
    /// Validates that a path is within the expected directory (security measure).
    /// </summary>
    protected static void ValidatePathWithinDirectory(string path, string directory)
    {
        var fullPath = Path.GetFullPath(path);
        var fullDir = Path.GetFullPath(directory);

        if (!fullPath.StartsWith(fullDir, StringComparison.OrdinalIgnoreCase))
        {
            throw new UnauthorizedAccessException(
                $"Path traversal detected: {path} is outside {directory}");
        }
    }

    #endregion
}
