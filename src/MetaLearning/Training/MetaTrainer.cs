using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Data;
using Newtonsoft.Json;
using static Newtonsoft.Json.Formatting;

namespace AiDotNet.MetaLearning.Training;

/// <summary>
/// Trainer for meta-learning algorithms with checkpointing and logging support.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<T>, Tensor<T>).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The MetaTrainer orchestrates the meta-learning training process.
/// It handles:
/// - Training loop execution
/// - Checkpointing (saving progress)
/// - Logging metrics
/// - Early stopping
/// - Deterministic seeding for reproducibility
///
/// Think of it as a coach that manages the training process, keeps track of progress,
/// and can save/restore training state.
/// </para>
/// </remarks>
public class MetaTrainer<T, TInput, TOutput>
{
    private readonly IMetaLearningAlgorithm<T, TInput, TOutput> _algorithm;
    private readonly IEpisodicDataset<T, TInput, TOutput> _trainDataset;
    private readonly IEpisodicDataset<T, TInput, TOutput>? _valDataset;
    private readonly MetaTrainerOptions _options;
    private readonly List<TrainingMetrics<T>> _trainingHistory;
    private int _currentEpoch;
    private T _bestValLoss;
    private bool _hasBestValLoss;
    private int _epochsWithoutImprovement;

    /// <summary>
    /// Initializes a new instance of the MetaTrainer class.
    /// </summary>
    /// <param name="algorithm">The meta-learning algorithm to train.</param>
    /// <param name="trainDataset">The training dataset.</param>
    /// <param name="valDataset">The validation dataset (optional).</param>
    /// <param name="options">Training configuration options.</param>
    public MetaTrainer(
        IMetaLearningAlgorithm<T, TInput, TOutput> algorithm,
        IEpisodicDataset<T, TInput, TOutput> trainDataset,
        IEpisodicDataset<T, TInput, TOutput>? valDataset = null,
        MetaTrainerOptions? options = null)
    {
        _algorithm = algorithm ?? throw new ArgumentNullException(nameof(algorithm));
        _trainDataset = trainDataset ?? throw new ArgumentNullException(nameof(trainDataset));
        _valDataset = valDataset;
        _options = options ?? new MetaTrainerOptions();
        _trainingHistory = new List<TrainingMetrics<T>>();
        _currentEpoch = 0;
        _epochsWithoutImprovement = 0;
        _bestValLoss = default(T)!;
        _hasBestValLoss = false;

        // Set random seeds for reproducibility
        if (_options.RandomSeed.HasValue)
        {
            _trainDataset.SetRandomSeed(_options.RandomSeed.Value);
            _valDataset?.SetRandomSeed(_options.RandomSeed.Value + 1); // Different seed for val
        }
    }

    /// <summary>
    /// Trains the meta-learning algorithm.
    /// </summary>
    /// <returns>The training history.</returns>
    public List<TrainingMetrics<T>> Train()
    {
        Console.WriteLine($"Starting meta-training with {_algorithm.AlgorithmName}");
        Console.WriteLine($"Epochs: {_options.NumEpochs}, Tasks per epoch: {_options.TasksPerEpoch}");

        for (int epoch = 0; epoch < _options.NumEpochs; epoch++)
        {
            _currentEpoch = epoch;
            var epochMetrics = TrainEpoch();

            _trainingHistory.Add(epochMetrics);

            // Log progress
            if (epoch % _options.LogInterval == 0 || epoch == _options.NumEpochs - 1)
            {
                LogProgress(epochMetrics);
            }

            // Save checkpoint
            if (_options.CheckpointInterval > 0 && epoch % _options.CheckpointInterval == 0)
            {
                SaveCheckpoint();
            }

            // Check for early stopping
            if (_options.EarlyStoppingPatience > 0 && _valDataset != null && ShouldStopEarly(epochMetrics))
            {
                Console.WriteLine($"Early stopping triggered at epoch {epoch}");
                break;
            }
        }

        // Save final checkpoint
        if (_options.CheckpointInterval > 0)
        {
            SaveCheckpoint(isFinal: true);
        }

        Console.WriteLine("Meta-training completed!");
        return _trainingHistory;
    }

    /// <summary>
    /// Trains for one epoch.
    /// </summary>
    /// <returns>The metrics for this epoch.</returns>
    private TrainingMetrics<T> TrainEpoch()
    {
        double totalTrainLoss = 0.0;
        int numBatches = _options.TasksPerEpoch / _options.MetaBatchSize;

        for (int batch = 0; batch < numBatches; batch++)
        {
            // Sample a batch of tasks
            var tasks = _trainDataset.SampleTasks(
                _options.MetaBatchSize,
                _options.NumWays,
                _options.NumShots,
                _options.NumQueryPerClass
            );

            var taskBatch = new TaskBatch<T, TInput, TOutput>(tasks);

            // Perform meta-training step
            T batchLoss = _algorithm.MetaTrain(taskBatch);
            totalTrainLoss += Convert.ToDouble(batchLoss);
        }

        double avgTrainLoss = totalTrainLoss / numBatches;

        // Validation
        double? avgValLoss = null;
        if (_valDataset != null && _currentEpoch % _options.ValInterval == 0)
        {
            avgValLoss = Validate();
        }

        return new TrainingMetrics<T>
        {
            Epoch = _currentEpoch,
            TrainLoss = avgTrainLoss,
            ValLoss = avgValLoss,
            Timestamp = DateTimeOffset.UtcNow
        };
    }

    /// <summary>
    /// Validates the model on the validation set.
    /// </summary>
    /// <returns>The average validation loss.</returns>
    private double Validate()
    {
        if (_valDataset == null)
        {
            throw new InvalidOperationException("Validation dataset is not set.");
        }

        double totalValLoss = 0.0;
        int numBatches = _options.ValTasks / _options.MetaBatchSize;

        for (int batch = 0; batch < numBatches; batch++)
        {
            var tasks = _valDataset.SampleTasks(
                _options.MetaBatchSize,
                _options.NumWays,
                _options.NumShots,
                _options.NumQueryPerClass
            );

            var taskBatch = new TaskBatch<T, TInput, TOutput>(tasks);
            T batchLoss = _algorithm.Evaluate(taskBatch);
            totalValLoss += Convert.ToDouble(batchLoss);
        }

        return totalValLoss / numBatches;
    }

    /// <summary>
    /// Logs training progress.
    /// </summary>
    /// <param name="metrics">The metrics to log.</param>
    private void LogProgress(TrainingMetrics<T> metrics)
    {
        Console.WriteLine($"Epoch {metrics.Epoch}/{_options.NumEpochs} - " +
                         $"Train Loss: {metrics.TrainLoss:F4}" +
                         (metrics.ValLoss.HasValue ? $", Val Loss: {metrics.ValLoss.Value:F4}" : ""));
    }

    /// <summary>
    /// Saves a checkpoint of the current training state.
    /// </summary>
    /// <param name="isFinal">Whether this is the final checkpoint.</param>
    private void SaveCheckpoint(bool isFinal = false)
    {
        if (string.IsNullOrEmpty(_options.CheckpointDir))
        {
            return;
        }

        try
        {
            Directory.CreateDirectory(_options.CheckpointDir);

            string checkpointName = isFinal ? "final" : $"epoch_{_currentEpoch}";
            string checkpointPath = Path.Combine(_options.CheckpointDir, $"{checkpointName}_checkpoint.json");

            var checkpoint = new MetaLearningCheckpoint
            {
                Epoch = _currentEpoch,
                AlgorithmName = _algorithm.AlgorithmName,
                BestValLoss = _hasBestValLoss ? Convert.ToDouble(_bestValLoss) : (double?)null,
                EpochsWithoutImprovement = _epochsWithoutImprovement,
                TrainingHistory = _trainingHistory,
                Timestamp = DateTimeOffset.UtcNow
            };

            string json = JsonConvert.SerializeObject(checkpoint, Formatting.Indented);

            File.WriteAllText(checkpointPath, json);

            // Save model parameters
            var model = _algorithm.GetMetaModel();
            string modelPath = Path.Combine(_options.CheckpointDir, $"{checkpointName}_model.bin");
            model.SaveModel(modelPath);

            if (_options.Verbose)
            {
                Console.WriteLine($"Checkpoint saved: {checkpointPath}");
            }
        }
        catch (IOException ex)
        {
            Console.WriteLine($"Warning: Failed to save checkpoint (I/O error): {ex.Message}");
        }
        catch (UnauthorizedAccessException ex)
        {
            Console.WriteLine($"Warning: Failed to save checkpoint (unauthorized): {ex.Message}");
        }
        catch (JsonException ex)
        {
            Console.WriteLine($"Warning: Failed to save checkpoint (serialization error): {ex.Message}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Failed to save checkpoint: {ex.Message}");
        }
    }

    /// <summary>
    /// Checks if early stopping criteria are met.
    /// </summary>
    /// <param name="metrics">The current epoch metrics.</param>
    /// <returns>True if training should stop early.</returns>
    private bool ShouldStopEarly(TrainingMetrics<T> metrics)
    {
        if (!metrics.ValLoss.HasValue)
        {
            return false;
        }

        double valLossDouble = metrics.ValLoss.Value;
        T currentValLoss = (T)(object)valLossDouble; // Safe cast via boxing for numeric types

        // Initialize best validation loss on first validation
        if (!_hasBestValLoss)
        {
            _bestValLoss = currentValLoss;
            _hasBestValLoss = true;
            _epochsWithoutImprovement = 0;
            return false;
        }

        // Check if validation loss improved
        if (Convert.ToDouble(currentValLoss) < Convert.ToDouble(_bestValLoss))
        {
            _bestValLoss = currentValLoss;
            _epochsWithoutImprovement = 0;
            return false;
        }

        // No improvement
        _epochsWithoutImprovement++;
        return _epochsWithoutImprovement >= _options.EarlyStoppingPatience;
    }

    /// <summary>
    /// Gets the training history.
    /// </summary>
    public List<TrainingMetrics<T>> TrainingHistory => _trainingHistory;

    /// <summary>
    /// Gets the current epoch.
    /// </summary>
    public int CurrentEpoch => _currentEpoch;
}

/// <summary>
/// Configuration options for MetaTrainer.
/// </summary>
public class MetaTrainerOptions
{
    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    public int NumEpochs { get; set; } = 100;

    /// <summary>
    /// Gets or sets the number of tasks to train on per epoch.
    /// </summary>
    public int TasksPerEpoch { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the meta-batch size (number of tasks per meta-update).
    /// </summary>
    public int MetaBatchSize { get; set; } = 4;

    /// <summary>
    /// Gets or sets the number of ways (classes per task).
    /// </summary>
    public int NumWays { get; set; } = 5;

    /// <summary>
    /// Gets or sets the number of shots (examples per class in support set).
    /// </summary>
    public int NumShots { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of query examples per class.
    /// </summary>
    public int NumQueryPerClass { get; set; } = 15;

    /// <summary>
    /// Gets or sets the validation interval (validate every N epochs).
    /// </summary>
    public int ValInterval { get; set; } = 5;

    /// <summary>
    /// Gets or sets the number of validation tasks.
    /// </summary>
    public int ValTasks { get; set; } = 100;

    /// <summary>
    /// Gets or sets the logging interval (log every N epochs).
    /// </summary>
    public int LogInterval { get; set; } = 1;

    /// <summary>
    /// Gets or sets the checkpoint interval (save checkpoint every N epochs, 0 to disable).
    /// </summary>
    public int CheckpointInterval { get; set; } = 10;

    /// <summary>
    /// Gets or sets the checkpoint directory.
    /// </summary>
    public string CheckpointDir { get; set; } = "./checkpoints";

    /// <summary>
    /// Gets or sets the early stopping patience (number of epochs without improvement).
    /// </summary>
    public int EarlyStoppingPatience { get; set; } = 20;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Gets or sets whether to print verbose output.
    /// </summary>
    public bool Verbose { get; set; } = true;
}

/// <summary>
/// Represents training metrics for a single epoch.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TrainingMetrics<T>
{
    public int Epoch { get; set; }
    public double TrainLoss { get; set; }
    public double? ValLoss { get; set; }
    public DateTimeOffset Timestamp { get; set; }
}

/// <summary>
/// Represents a meta-learning checkpoint.
/// </summary>
public class MetaLearningCheckpoint
{
    public int Epoch { get; set; }
    public string AlgorithmName { get; set; } = string.Empty;
    public double? BestValLoss { get; set; }
    public int EpochsWithoutImprovement { get; set; }
    public object? TrainingHistory { get; set; }
    public DateTimeOffset Timestamp { get; set; }
}
