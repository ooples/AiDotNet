using System.Diagnostics;
using AiDotNet.Configuration;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Factories;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Training.Configuration;
using AiDotNet.Training.Factories;

namespace AiDotNet.Training;

/// <summary>
/// Abstract base class for all trainers, providing shared infrastructure for
/// configuration-driven training pipelines.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Think of TrainerBase as a template for running experiments.
/// It handles all the boilerplate—loading data, setting up the model/optimizer/loss,
/// timing the run, and collecting results—so that concrete trainers only need to
/// implement the actual training strategy in <see cref="TrainEpoch"/>.
/// </para>
/// <para>
/// The inheritance pattern follows the project's architecture requirements:
/// <c>ITrainer&lt;T&gt;</c> (interface) → <c>TrainerBase&lt;T&gt;</c> (base) → <c>Trainer&lt;T&gt;</c> (concrete).
/// To create a custom training strategy, inherit from this base class and override
/// <see cref="TrainEpoch"/>.
/// </para>
/// </remarks>
public abstract class TrainerBase<T> : ITrainer<T>
{
    private readonly IFullModel<T, Matrix<T>, Vector<T>> _model;
    private readonly IOptimizer<T, Matrix<T>, Vector<T>>? _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly int _epochs;
    private readonly bool _enableLogging;
    private readonly int? _seed;
    private readonly CsvDataLoader<T>? _csvLoader;
    private Matrix<T>? _features;
    private Vector<T>? _labels;

    /// <inheritdoc/>
    public TrainingRecipeConfig Config { get; }

    /// <summary>
    /// Gets the model created from the configuration.
    /// </summary>
    protected IFullModel<T, Matrix<T>, Vector<T>> Model => _model;

    /// <summary>
    /// Gets the optimizer created from the configuration, if one was specified.
    /// </summary>
    internal IOptimizer<T, Matrix<T>, Vector<T>>? Optimizer => _optimizer;

    /// <summary>
    /// Gets the loss function created from the configuration.
    /// </summary>
    protected ILossFunction<T> LossFunction => _lossFunction;

    /// <summary>
    /// Gets or sets the action used for logging training messages.
    /// Defaults to <see cref="Console.WriteLine(string)"/>.
    /// Set to a custom delegate to redirect logs to your preferred logging framework.
    /// </summary>
    public Action<string> LogAction { get; set; } = Console.WriteLine;

    /// <summary>
    /// Creates a trainer base from a YAML configuration file.
    /// </summary>
    /// <param name="yamlFilePath">Path to the YAML training recipe file.</param>
    /// <exception cref="ArgumentException">Thrown when the file path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the YAML file does not exist.</exception>
    protected TrainerBase(string yamlFilePath)
        : this(YamlConfigLoader.LoadFromFile<TrainingRecipeConfig>(yamlFilePath))
    {
    }

    /// <summary>
    /// Creates a trainer base from a <see cref="TrainingRecipeConfig"/> object.
    /// </summary>
    /// <param name="config">The training recipe configuration.</param>
    /// <exception cref="ArgumentNullException">Thrown when config is null.</exception>
    /// <exception cref="ArgumentException">Thrown when required config sections are missing.</exception>
    protected TrainerBase(TrainingRecipeConfig config)
    {
        Config = config ?? throw new ArgumentNullException(nameof(config));

        if (config.Model is null || string.IsNullOrWhiteSpace(config.Model.Name))
        {
            throw new ArgumentException("Training recipe must specify a model with a name.", nameof(config));
        }

        // Create model
        _model = ModelFactory<T, Matrix<T>, Vector<T>>.Create(config.Model);

        // Create optimizer (if specified)
        if (config.Optimizer is not null && !string.IsNullOrWhiteSpace(config.Optimizer.Name))
        {
            if (!Enum.TryParse<OptimizerType>(config.Optimizer.Name, ignoreCase: true, out var optimizerType))
            {
                throw new ArgumentException(
                    $"Unknown optimizer name: '{config.Optimizer.Name}'. " +
                    $"Valid names are: {string.Join(", ", Enum.GetNames(typeof(OptimizerType)))}",
                    nameof(config));
            }

            _optimizer = OptimizerFactory<T, Matrix<T>, Vector<T>>.CreateOptimizer(optimizerType);
            _optimizer.SetModel(_model);

            // Apply learning rate from config to the optimizer's options
            if (config.Optimizer.LearningRate > 0)
            {
                var optimizerOptions = _optimizer.GetOptions();
                optimizerOptions.InitialLearningRate = config.Optimizer.LearningRate;
            }
        }

        // Create loss function (default to model's DefaultLossFunction if not specified)
        _lossFunction = config.LossFunction is not null && !string.IsNullOrWhiteSpace(config.LossFunction.Name)
            ? LossFunctionFactory<T>.Create(config.LossFunction.Name, config.LossFunction.Params)
            : _model.DefaultLossFunction;

        // Apply trainer settings
        _epochs = config.Trainer?.Epochs ?? 10;
        _enableLogging = config.Trainer?.EnableLogging ?? true;
        _seed = config.Trainer?.Seed;

        // Create data loader if path is specified
        _csvLoader = DatasetFactory<T>.Create(config.Dataset);
    }

    /// <summary>
    /// Sets in-memory feature and label data for training, bypassing CSV loading.
    /// </summary>
    /// <param name="features">The input feature matrix.</param>
    /// <param name="labels">The output label vector.</param>
    /// <exception cref="ArgumentNullException">Thrown when features or labels is null.</exception>
    public void SetData(Matrix<T> features, Vector<T> labels)
    {
        _features = features ?? throw new ArgumentNullException(nameof(features));
        _labels = labels ?? throw new ArgumentNullException(nameof(labels));
    }

    /// <inheritdoc/>
    public async Task<TrainingResult<T>> RunAsync()
    {
        return await Task.Run(() => Run()).ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public TrainingResult<T> Run()
    {
        var stopwatch = Stopwatch.StartNew();
        var epochLosses = new List<T>();

        // Load data if needed
        var (features, labels) = ResolveData();

        // Set seed for reproducibility if specified
        if (_seed.HasValue)
        {
            RandomHelper.CreateSeededRandom(_seed.Value);
        }

        if (_enableLogging)
        {
            LogAction($"Training {Config.Model?.Name} for {_epochs} epochs...");
            if (_optimizer is not null)
            {
                LogAction($"  Optimizer: {Config.Optimizer?.Name}");
            }
            LogAction($"  Loss Function: {_lossFunction.GetType().Name}");
        }

        // Training loop - delegates to subclass for the actual training strategy
        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            var loss = TrainEpoch(features, labels, epoch);
            epochLosses.Add(loss);

            if (_enableLogging)
            {
                LogAction($"  Epoch {epoch + 1}/{_epochs} - Loss: {loss}");
            }

            // Check for early stopping via optimizer
            if (_optimizer is not null && _optimizer.ShouldEarlyStop())
            {
                if (_enableLogging)
                {
                    LogAction($"  Early stopping triggered at epoch {epoch + 1}");
                }
                break;
            }
        }

        stopwatch.Stop();

        if (_enableLogging)
        {
            LogAction($"Training completed in {stopwatch.Elapsed.TotalSeconds:F2}s");
        }

        return new TrainingResult<T>
        {
            TrainedModel = _model,
            EpochLosses = epochLosses,
            TotalEpochs = epochLosses.Count,
            TrainingDuration = stopwatch.Elapsed,
            Completed = true
        };
    }

    /// <summary>
    /// Executes a single training epoch and returns the computed loss.
    /// </summary>
    /// <param name="features">The input feature matrix for this epoch.</param>
    /// <param name="labels">The output label vector for this epoch.</param>
    /// <param name="epoch">The zero-based epoch index.</param>
    /// <returns>The loss value computed after training for this epoch.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the method that concrete trainers override to define
    /// their training strategy. For example, a time series trainer calls <c>model.Train()</c>,
    /// while a gradient-based trainer would iterate over batches performing forward/backward passes.
    /// </para>
    /// </remarks>
    protected abstract T TrainEpoch(Matrix<T> features, Vector<T> labels, int epoch);

    /// <summary>
    /// Resolves the feature/label data from either in-memory data or the CSV loader.
    /// </summary>
    protected (Matrix<T> Features, Vector<T> Labels) ResolveData()
    {
        // Prefer in-memory data if set
        if (_features is not null && _labels is not null)
        {
            return (_features, _labels);
        }

        // Load from CSV
        if (_csvLoader is not null)
        {
            _csvLoader.LoadAsync().ConfigureAwait(false).GetAwaiter().GetResult();
            return (_csvLoader.Features, _csvLoader.Labels);
        }

        throw new InvalidOperationException(
            "No training data available. Either specify a dataset path in the configuration " +
            "or call SetData() with in-memory data before calling Run().");
    }
}
