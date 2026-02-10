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
/// Executes a complete training pipeline from a YAML configuration or <see cref="TrainingRecipeConfig"/>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Trainer is the central piece that brings everything together.
/// Give it a YAML file or a configuration object, and it will:
/// 1. Create the model, optimizer, and loss function
/// 2. Load data from CSV (if specified)
/// 3. Run the training loop for the specified number of epochs
/// 4. Return the trained model with loss history
/// </para>
/// <para>
/// <b>Example usage from YAML:</b>
/// <code>
/// var trainer = new Trainer&lt;double&gt;("config/my-experiment.yaml");
/// var result = trainer.Run();
/// // result.TrainedModel is ready for predictions
/// // result.EpochLosses shows the training progress
/// </code>
/// </para>
/// <para>
/// <b>Example usage with in-memory data:</b>
/// <code>
/// var config = new TrainingRecipeConfig { ... };
/// var trainer = new Trainer&lt;double&gt;(config);
/// trainer.SetData(features, labels);
/// var result = trainer.Run();
/// </code>
/// </para>
/// </remarks>
public class Trainer<T> : ITrainer<T>
{
    private readonly ITimeSeriesModel<T> _model;
    private readonly IOptimizer<T, Matrix<T>, Vector<T>>? _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly int _epochs;
    private readonly bool _enableLogging;
    private readonly int? _seed;
    private CsvDataLoader<T>? _csvLoader;
    private Matrix<T>? _features;
    private Vector<T>? _labels;

    /// <inheritdoc/>
    public TrainingRecipeConfig Config { get; }

    /// <summary>
    /// Gets the optimizer created from the configuration, if one was specified.
    /// </summary>
    public IOptimizer<T, Matrix<T>, Vector<T>>? Optimizer => _optimizer;

    /// <summary>
    /// Creates a trainer from a YAML configuration file.
    /// </summary>
    /// <param name="yamlFilePath">Path to the YAML training recipe file.</param>
    /// <exception cref="ArgumentException">Thrown when the file path is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the YAML file does not exist.</exception>
    public Trainer(string yamlFilePath)
        : this(YamlConfigLoader.LoadFromFile<TrainingRecipeConfig>(yamlFilePath))
    {
    }

    /// <summary>
    /// Creates a trainer from a <see cref="TrainingRecipeConfig"/> object.
    /// </summary>
    /// <param name="config">The training recipe configuration.</param>
    /// <exception cref="ArgumentNullException">Thrown when config is null.</exception>
    /// <exception cref="ArgumentException">Thrown when required config sections are missing.</exception>
    public Trainer(TrainingRecipeConfig config)
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
        }

        // Create loss function (default to model's DefaultLossFunction if not specified)
        if (config.LossFunction is not null && !string.IsNullOrWhiteSpace(config.LossFunction.Name))
        {
            _lossFunction = LossFunctionFactory<T>.Create(config.LossFunction.Name, config.LossFunction.Params);
        }
        else
        {
            _lossFunction = _model.DefaultLossFunction;
        }

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
            Console.WriteLine($"Training {Config.Model?.Name} for {_epochs} epochs...");
            if (_optimizer is not null)
            {
                Console.WriteLine($"  Optimizer: {Config.Optimizer?.Name}");
            }
            Console.WriteLine($"  Loss Function: {_lossFunction.GetType().Name}");
        }

        // Training loop
        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            // Train the model on the data
            _model.Train(features, labels);

            // Compute loss
            var predictions = _model.Predict(features);
            var loss = _lossFunction.CalculateLoss(predictions, labels);
            epochLosses.Add(loss);

            if (_enableLogging)
            {
                Console.WriteLine($"  Epoch {epoch + 1}/{_epochs} - Loss: {loss}");
            }

            // Check for early stopping via optimizer
            if (_optimizer is not null && _optimizer.ShouldEarlyStop())
            {
                if (_enableLogging)
                {
                    Console.WriteLine($"  Early stopping triggered at epoch {epoch + 1}");
                }
                break;
            }
        }

        stopwatch.Stop();

        if (_enableLogging)
        {
            Console.WriteLine($"Training completed in {stopwatch.Elapsed.TotalSeconds:F2}s");
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
    /// Resolves the feature/label data from either in-memory data or the CSV loader.
    /// </summary>
    private (Matrix<T> Features, Vector<T> Labels) ResolveData()
    {
        // Prefer in-memory data if set
        if (_features is not null && _labels is not null)
        {
            return (_features, _labels);
        }

        // Load from CSV
        if (_csvLoader is not null)
        {
            _csvLoader.LoadAsync().GetAwaiter().GetResult();
            return (_csvLoader.Features, _csvLoader.Labels);
        }

        throw new InvalidOperationException(
            "No training data available. Either specify a dataset path in the configuration " +
            "or call SetData() with in-memory data before calling Run().");
    }
}
