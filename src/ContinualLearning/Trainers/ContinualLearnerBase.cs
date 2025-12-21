using AiDotNet.ContinualLearning.Config;
using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.ContinualLearning.Memory;
using AiDotNet.ContinualLearning.Results;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using Newtonsoft.Json.Linq;

namespace AiDotNet.ContinualLearning.Trainers;

/// <summary>
/// Base class for continual learning trainers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Algorithm Implementers:</b> To create a new continual learning algorithm:
/// 1. Extend this base class
/// 2. Implement the LearnTask method with your algorithm-specific training logic
/// 3. Pass an IContinualLearningStrategy implementation that handles your algorithm's specific regularization logic
/// 4. All shared functionality (evaluation, memory management) is handled automatically
/// </para>
/// </remarks>
public abstract class ContinualLearnerBase<T, TInput, TOutput> : IContinualLearner<T, TInput, TOutput>
{
    protected readonly IFullModel<T, TInput, TOutput> Model;
    protected readonly ILossFunction<T> LossFunction;
    protected readonly IContinualLearnerConfig<T> Configuration;
    protected readonly IContinualLearningStrategy<T, TInput, TOutput> Strategy;
    protected readonly ExperienceReplayBuffer<T, TInput, TOutput> MemoryBuffer;
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    protected int _tasksLearned;
    protected readonly List<IDataset<T, TInput, TOutput>> _taskTestSets;
    protected readonly Dictionary<int, T> _initialAccuracies; // For forward transfer

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> BaseModel => Model;

    /// <inheritdoc/>
    public IContinualLearnerConfig<T> Config => Configuration;

    /// <inheritdoc/>
    public int TasksLearned => _tasksLearned;

    /// <summary>
    /// Initializes a new continual learner.
    /// </summary>
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

        MemoryBuffer = new ExperienceReplayBuffer<T, TInput, TOutput>(config.MemorySize);
        _tasksLearned = 0;
        _taskTestSets = new List<IDataset<T, TInput, TOutput>>();
        _initialAccuracies = new Dictionary<int, T>();
    }

    /// <inheritdoc/>
    public abstract ContinualLearningResult<T> LearnTask(IDataset<T, TInput, TOutput> taskData);

    /// <inheritdoc/>
    public virtual ContinualEvaluationResult<T> EvaluateAllTasks()
    {
        if (_tasksLearned == 0)
            throw new InvalidOperationException("No tasks have been learned yet");

        var startTime = System.Diagnostics.Stopwatch.StartNew();

        var accuracies = new List<T>();
        var losses = new List<T>();

        for (int i = 0; i < _tasksLearned; i++)
        {
            var result = EvaluateTask(i, _taskTestSets[i]);
            accuracies.Add(result.Accuracy);
            losses.Add(result.Loss);
        }

        startTime.Stop();

        var accVector = new Vector<T>(accuracies.ToArray());
        var lossVector = new Vector<T>(losses.ToArray());

        // Compute metrics
        var avgAccuracy = NumOps.Divide(
            accVector.Sum(),
            NumOps.FromDouble(_tasksLearned));

        var avgLoss = NumOps.Divide(
            lossVector.Sum(),
            NumOps.FromDouble(_tasksLearned));

        // Backward transfer: average change in accuracy on previous tasks
        T backwardTransfer = NumOps.Zero;
        if (_tasksLearned > 1)
        {
            for (int i = 0; i < _tasksLearned - 1; i++)
            {
                if (_initialAccuracies.TryGetValue(i, out var initialAcc))
                {
                    backwardTransfer = NumOps.Add(
                        backwardTransfer,
                        NumOps.Subtract(accuracies[i], initialAcc));
                }
            }
            backwardTransfer = NumOps.Divide(
                backwardTransfer,
                NumOps.FromDouble(_tasksLearned - 1));
        }

        // Forward transfer: average initial accuracy
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

        return new ContinualEvaluationResult<T>(
            taskAccuracies: accVector,
            taskLosses: lossVector,
            averageAccuracy: avgAccuracy,
            averageLoss: avgLoss,
            backwardTransfer: backwardTransfer,
            forwardTransfer: forwardTransfer,
            evaluationTime: startTime.Elapsed);
    }

    /// <inheritdoc/>
    public abstract TaskEvaluationResult<T> EvaluateTask(int taskId, IDataset<T, TInput, TOutput> testData);

    /// <inheritdoc/>
    public virtual void Save(string directoryPath)
    {
        if (string.IsNullOrWhiteSpace(directoryPath))
            throw new ArgumentException("Directory path cannot be null or empty", nameof(directoryPath));

        // Normalize the path to prevent path traversal attacks
        var normalizedDir = Path.GetFullPath(directoryPath);
        Directory.CreateDirectory(normalizedDir);

        // Save model - using hardcoded filename is safe
        var modelPath = Path.GetFullPath(Path.Combine(normalizedDir, "model.bin"));
        ValidatePathWithinDirectory(modelPath, normalizedDir);
        Model.SaveModel(modelPath);

        // Save metadata - using hardcoded filename is safe
        var metadataPath = Path.GetFullPath(Path.Combine(normalizedDir, "metadata.json"));
        ValidatePathWithinDirectory(metadataPath, normalizedDir);
        var metadata = new JObject
        {
            ["TasksLearned"] = _tasksLearned,
            ["MemorySize"] = Configuration.MemorySize
        };
        File.WriteAllText(metadataPath, metadata.ToString());
    }

    /// <inheritdoc/>
    public virtual void Load(string directoryPath)
    {
        if (string.IsNullOrWhiteSpace(directoryPath))
            throw new ArgumentException("Directory path cannot be null or empty", nameof(directoryPath));

        // Normalize the path to prevent path traversal attacks
        var normalizedDir = Path.GetFullPath(directoryPath);
        if (!Directory.Exists(normalizedDir))
            throw new DirectoryNotFoundException($"Directory not found: {normalizedDir}");

        var modelPath = Path.GetFullPath(Path.Combine(normalizedDir, "model.bin"));
        ValidatePathWithinDirectory(modelPath, normalizedDir);
        Model.LoadModel(modelPath);

        // Load metadata if needed
        var metadataPath = Path.GetFullPath(Path.Combine(normalizedDir, "metadata.json"));
        ValidatePathWithinDirectory(metadataPath, normalizedDir);
        if (File.Exists(metadataPath))
        {
            var json = File.ReadAllText(metadataPath);
            var metadata = JObject.Parse(json);

            if (metadata != null && metadata.TryGetValue("TasksLearned", out var tasksLearnedToken))
            {
                _tasksLearned = tasksLearnedToken.Value<int>();
            }
        }
    }

    /// <summary>
    /// Validates that the given path is within the expected directory to prevent path traversal attacks.
    /// </summary>
    private static void ValidatePathWithinDirectory(string path, string directory)
    {
        var fullPath = Path.GetFullPath(path);
        var fullDir = Path.GetFullPath(directory);

        if (!fullPath.StartsWith(fullDir, StringComparison.OrdinalIgnoreCase))
        {
            throw new UnauthorizedAccessException($"Path traversal detected: {path} is outside {directory}");
        }
    }

    /// <inheritdoc/>
    public virtual void Reset()
    {
        _tasksLearned = 0;
        MemoryBuffer.Clear();
        _taskTestSets.Clear();
        _initialAccuracies.Clear();
    }
}
