using AiDotNet.Data.Abstractions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Config;
using AiDotNet.Models.Results;

namespace AiDotNet.MetaLearning.Trainers;

/// <summary>
/// Production-ready base implementation for Reptile meta-learning trainers.
/// </summary>
/// <typeparam name="T">The numeric data type (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <remarks>
/// <para>
/// Reptile is a first-order meta-learning algorithm (Nichol et al., 2018) that provides
/// a simple yet effective approach to few-shot learning. It works by repeatedly moving
/// model parameters toward task-adapted parameters, causing them to converge to an
/// initialization that enables rapid adaptation.
/// </para>
/// <para><b>Key Advantages:</b>
/// - Simpler than MAML (no second-order derivatives)
/// - Computationally efficient
/// - Works with any gradient-based model
/// - Strong empirical performance on few-shot tasks
/// </para>
/// <para><b>Algorithm Overview:</b>
/// <code>
/// Initialize θ (meta-parameters)
/// for each meta-iteration:
///     Sample batch of B tasks
///     for each task i in batch:
///         θ_i = θ (clone parameters)
///         for k = 1 to K (inner steps):
///             θ_i = θ_i - α∇L(θ_i, support_set_i)
///         Δθ_i = θ_i - θ
///     θ = θ + ε * Average(Δθ_i) (meta-update)
/// return θ
/// </code>
/// </para>
/// </remarks>
public abstract class ReptileTrainerBase<T, TInput, TOutput> : IMetaLearner<T, TInput, TOutput>
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
    protected readonly IEpisodicDataLoader<T> DataLoader;

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
    /// Initializes a new instance of the ReptileTrainerBase with a configuration object.
    /// </summary>
    /// <param name="metaModel">The model to meta-train.</param>
    /// <param name="lossFunction">Loss function for evaluation.</param>
    /// <param name="dataLoader">Episodic data loader for sampling meta-learning tasks.</param>
    /// <param name="config">Configuration object with all hyperparameters. If null, uses default ReptileTrainerConfig.</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel, lossFunction, or dataLoader is null.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration validation fails.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up the Reptile meta-learning trainer with your model and settings.
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
    protected ReptileTrainerBase(
        IFullModel<T, TInput, TOutput> metaModel,
        ILossFunction<T> lossFunction,
        IEpisodicDataLoader<T> dataLoader,
        IMetaLearnerConfig<T>? config = null)
    {
        if (metaModel == null)
            throw new ArgumentNullException(nameof(metaModel), "Meta-model cannot be null");
        if (lossFunction == null)
            throw new ArgumentNullException(nameof(lossFunction), "Loss function cannot be null");
        if (dataLoader == null)
            throw new ArgumentNullException(nameof(dataLoader), "Episodic data loader cannot be null");

        var configuration = config ?? new ReptileTrainerConfig<T>();

        if (!configuration.IsValid())
            throw new ArgumentException("Configuration validation failed", nameof(config));

        MetaModel = metaModel;
        LossFunction = lossFunction;
        DataLoader = dataLoader;
        Configuration = configuration;
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
    public abstract MetaAdaptationResult<T> AdaptAndEvaluate(MetaLearningTask<T> task);

    /// <inheritdoc/>
    public virtual MetaTrainingResult<T> Train(int numMetaIterations, int batchSize = 1)
    {
        if (numMetaIterations < 1)
            throw new ArgumentException("Number of meta-iterations must be at least 1", nameof(numMetaIterations));
        if (batchSize < 1)
            throw new ArgumentException("Batch size must be at least 1", nameof(batchSize));

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

        MetaModel.Save(filePath);
    }

    /// <inheritdoc/>
    public virtual void Load(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be null or empty", nameof(filePath));
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Model file not found: {filePath}");

        MetaModel.Load(filePath);
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
            var predValue = NumOps.ToDouble(predictedVector[i]);
            var targetValue = NumOps.ToDouble(targetVector[i]);

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
            Tensor<T> tensor => tensor.Flatten(),
            T[] array => new Vector<T>(array),
            _ => throw new NotSupportedException($"Output type {typeof(TOutput)} not supported for conversion to Vector<T>")
        };
    }
}
