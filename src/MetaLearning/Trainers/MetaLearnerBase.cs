using AiDotNet.Data.Abstractions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Trainers;

/// <summary>
/// Base class providing shared functionality for all meta-learning algorithms.
/// </summary>
/// <typeparam name="T">The numeric data type (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <remarks>
/// <para>
/// This base class implements the common infrastructure needed by all meta-learning algorithms
/// including MAML, Reptile, and SEAL. It provides:
/// - Configuration management and validation
/// - Loss and accuracy computation
/// - Model saving/loading
/// - Training loop orchestration
/// - Evaluation on multiple tasks
/// </para>
/// <para><b>For Algorithm Implementers:</b>
/// To create a new meta-learning algorithm:
/// 1. Extend this base class
/// 2. Implement MetaTrainStep() with your algorithm's meta-update logic
/// 3. Optionally override AdaptAndEvaluate() if your algorithm needs custom adaptation
/// 4. All shared functionality (metrics, saving, evaluation) is handled automatically
/// </para>
/// </remarks>
public abstract class MetaLearnerBase<T, TInput, TOutput> : IMetaLearner<T, TInput, TOutput>
{
    /// <summary>
    /// The model being meta-trained.
    /// </summary>
    protected readonly IFullModel<T, TInput, TOutput> MetaModel;

    /// <summary>
    /// The loss function used to evaluate task performance.
    /// </summary>
    protected readonly ILossFunction<T> LossFunction;

    /// <summary>
    /// Configuration containing learning rates and training parameters.
    /// </summary>
    protected readonly IMetaLearnerConfig<T> Configuration;

    /// <summary>
    /// Episodic data loader for sampling meta-learning tasks.
    /// </summary>
    protected readonly IEpisodicDataLoader<T, TInput, TOutput> DataLoader;

    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Current meta-training iteration count.
    /// </summary>
    protected int _currentIteration;

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> BaseModel => MetaModel;

    /// <inheritdoc/>
    public IMetaLearnerConfig<T> Config => Configuration;

    /// <inheritdoc/>
    public int CurrentIteration => _currentIteration;

    /// <summary>
    /// Initializes a new instance of the MetaLearnerBase with a configuration object.
    /// </summary>
    /// <param name="metaModel">The model to meta-train.</param>
    /// <param name="lossFunction">Loss function for evaluation.</param>
    /// <param name="dataLoader">Episodic data loader for sampling meta-learning tasks.</param>
    /// <param name="config">Configuration object with all hyperparameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel, lossFunction, or dataLoader is null.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration validation fails.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up the meta-learning trainer with your model and settings.
    ///
    /// <b>Parameters explained:</b>
    /// - <b>metaModel:</b> The neural network or model you want to train for fast adaptation
    /// - <b>lossFunction:</b> How to measure prediction errors (e.g., MSE for regression, CrossEntropy for classification)
    /// - <b>dataLoader:</b> Provides different tasks for meta-training (N-way K-shot episodic sampling)
    /// - <b>config:</b> Learning rates and steps - can be left as default for standard meta-learning
    ///
    /// The configuration controls two learning processes:
    /// - Inner loop: How the model adapts to each individual task
    /// - Outer loop: How the meta-parameters improve across tasks
    /// </para>
    /// </remarks>
    protected MetaLearnerBase(
        IFullModel<T, TInput, TOutput> metaModel,
        ILossFunction<T> lossFunction,
        IEpisodicDataLoader<T, TInput, TOutput> dataLoader,
        IMetaLearnerConfig<T> config)
    {
        if (metaModel == null)
            throw new ArgumentNullException(nameof(metaModel), "Meta-model cannot be null");
        if (lossFunction == null)
            throw new ArgumentNullException(nameof(lossFunction), "Loss function cannot be null");
        if (dataLoader == null)
            throw new ArgumentNullException(nameof(dataLoader), "Episodic data loader cannot be null");
        if (config == null)
            throw new ArgumentNullException(nameof(config), "Configuration cannot be null");

        if (!config.IsValid())
            throw new ArgumentException("Configuration validation failed", nameof(config));

        MetaModel = metaModel;
        LossFunction = lossFunction;
        DataLoader = dataLoader;
        Configuration = config;
        _currentIteration = 0;
    }

    /// <inheritdoc/>
    public abstract MetaTrainingStepResult<T> MetaTrainStep(int batchSize);

    /// <inheritdoc/>
    public virtual MetaEvaluationResult<T> Evaluate(int numTasks)
    {
        if (numTasks < 1)
            throw new ArgumentException("Number of tasks must be at least 1", nameof(numTasks));

        var startTime = System.Diagnostics.Stopwatch.StartNew();

        // Collect per-task results with generic T
        var accuracyValues = new List<T>();
        var lossValues = new List<T>();

        for (int i = 0; i < numTasks; i++)
        {
            var task = DataLoader.GetNextTask();
            var adaptResult = AdaptAndEvaluate(task);

            accuracyValues.Add(adaptResult.QueryAccuracy);
            lossValues.Add(adaptResult.QueryLoss);
        }

        startTime.Stop();

        // Convert to vectors for statistical analysis using existing infrastructure
        var accuracyVector = new Vector<T>(accuracyValues.ToArray());
        var lossVector = new Vector<T>(lossValues.ToArray());

        // Return new Result type using our established pattern
        return new MetaEvaluationResult<T>(
            taskAccuracies: accuracyVector,
            taskLosses: lossVector,
            evaluationTime: startTime.Elapsed);
    }

    /// <inheritdoc/>
    public abstract MetaAdaptationResult<T> AdaptAndEvaluate(MetaLearningTask<T, TInput, TOutput> task);

    /// <inheritdoc/>
    public virtual MetaTrainingResult<T> Train()
    {
        // Use configuration values - all parameters specified during construction
        int numMetaIterations = Configuration.NumMetaIterations;
        int batchSize = Configuration.MetaBatchSize;

        // Collect history with generic T
        var lossValues = new List<T>();
        var accuracyValues = new List<T>();
        var startTime = System.Diagnostics.Stopwatch.StartNew();

        for (int iteration = 0; iteration < numMetaIterations; iteration++)
        {
            var stepResult = MetaTrainStep(batchSize);
            lossValues.Add(stepResult.MetaLoss);
            accuracyValues.Add(stepResult.Accuracy);
        }

        startTime.Stop();

        // Convert to vectors for Result type
        var lossHistory = new Vector<T>(lossValues.ToArray());
        var accuracyHistory = new Vector<T>(accuracyValues.ToArray());

        // Return new Result type following established pattern
        return new MetaTrainingResult<T>(
            lossHistory: lossHistory,
            accuracyHistory: accuracyHistory,
            trainingTime: startTime.Elapsed);
    }

    /// <inheritdoc/>
    public virtual void Save(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be null or empty", nameof(filePath));

        MetaModel.SaveModel(filePath);
    }

    /// <inheritdoc/>
    public virtual void Load(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be null or empty", nameof(filePath));
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Model file not found: {filePath}");

        MetaModel.LoadModel(filePath);
    }

    /// <inheritdoc/>
    public virtual void Reset()
    {
        _currentIteration = 0;
        // Note: Model parameter reset should be handled by derived classes if needed
    }

    /// <summary>
    /// Computes the loss on a given dataset.
    /// </summary>
    /// <param name="model">Model to evaluate.</param>
    /// <param name="inputX">Input features.</param>
    /// <param name="targetY">Target labels.</param>
    /// <returns>Loss value.</returns>
    protected T ComputeLoss(IFullModel<T, TInput, TOutput> model, TInput inputX, TOutput targetY)
    {
        var predictions = model.Predict(inputX);

        // Convert predictions and targets to vectors for loss calculation
        var predictedVector = ConvertToVector(predictions);
        var targetVector = ConvertToVector(targetY);

        return LossFunction.CalculateLoss(predictedVector, targetVector);
    }

    /// <summary>
    /// Computes accuracy for classification tasks.
    /// </summary>
    /// <param name="model">Model to evaluate.</param>
    /// <param name="inputX">Input features.</param>
    /// <param name="targetY">Target labels.</param>
    /// <returns>Accuracy as generic T (between 0 and 1).</returns>
    protected T ComputeAccuracy(IFullModel<T, TInput, TOutput> model, TInput inputX, TOutput targetY)
    {
        var predictions = model.Predict(inputX);
        var predictedVector = ConvertToVector(predictions);
        var targetVector = ConvertToVector(targetY);

        int correct = 0;
        int total = Math.Min(predictedVector.Length, targetVector.Length);

        for (int i = 0; i < total; i++)
        {
            // For classification: check if predictions match targets
            // This works for both hard labels and argmax of softmax outputs
            var predValue = Convert.ToDouble(predictedVector[i]);
            var targetValue = Convert.ToDouble(targetVector[i]);

            if (Math.Abs(predValue - targetValue) < 0.5) // Tolerance for classification
                correct++;
        }

        // Return accuracy as generic T
        return NumOps.FromDouble(total > 0 ? (double)correct / total : 0.0);
    }

    /// <summary>
    /// Converts output type to Vector for loss calculation.
    /// </summary>
    private Vector<T> ConvertToVector(TOutput output)
    {
        return output switch
        {
            Vector<T> v => v,
            Tensor<T> tensor => tensor.ToVector(),
            T[] array => new Vector<T>(array),
            _ => throw new NotSupportedException($"Output type {typeof(TOutput)} not supported for conversion to Vector<T>")
        };
    }
}
