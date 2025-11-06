using AiDotNet.Data.Abstractions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Config;
using AiDotNet.MetaLearning.Metrics;

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
    /// Initializes a new instance with individual parameters (backwards compatibility).
    /// </summary>
    /// <param name="metaModel">The model to meta-train.</param>
    /// <param name="lossFunction">Loss function for evaluation.</param>
    /// <param name="innerSteps">Number of gradient steps per task.</param>
    /// <param name="innerLearningRate">Learning rate for inner loop.</param>
    /// <param name="metaLearningRate">Learning rate for outer loop.</param>
    protected ReptileTrainerBase(
        IFullModel<T, TInput, TOutput> metaModel,
        ILossFunction<T> lossFunction,
        int innerSteps = 5,
        double innerLearningRate = 0.01,
        double metaLearningRate = 0.001)
    {
        ValidateConstructorParameters(metaModel, lossFunction, innerSteps, innerLearningRate, metaLearningRate);

        MetaModel = metaModel;
        LossFunction = lossFunction;
        Configuration = new ReptileTrainerConfig<T>(innerLearningRate, metaLearningRate, innerSteps);
        _currentIteration = 0;
    }

    /// <summary>
    /// Initializes a new instance with configuration object (recommended for production).
    /// </summary>
    /// <param name="metaModel">The model to meta-train.</param>
    /// <param name="lossFunction">Loss function for evaluation.</param>
    /// <param name="config">Configuration object with all hyperparameters.</param>
    protected ReptileTrainerBase(
        IFullModel<T, TInput, TOutput> metaModel,
        ILossFunction<T> lossFunction,
        IMetaLearnerConfig<T> config)
    {
        if (metaModel == null)
            throw new ArgumentNullException(nameof(metaModel), "Meta-model cannot be null");
        if (lossFunction == null)
            throw new ArgumentNullException(nameof(lossFunction), "Loss function cannot be null");
        if (config == null)
            throw new ArgumentNullException(nameof(config), "Configuration cannot be null");
        if (!config.IsValid())
            throw new ArgumentException("Configuration validation failed", nameof(config));

        MetaModel = metaModel;
        LossFunction = lossFunction;
        Configuration = config;
        _currentIteration = 0;
    }

    /// <inheritdoc/>
    public abstract MetaTrainingMetrics MetaTrainStep(IEpisodicDataLoader<T> dataLoader, int batchSize);

    /// <inheritdoc/>
    public virtual MetaEvaluationMetrics Evaluate(IEpisodicDataLoader<T> dataLoader, int numTasks)
    {
        if (dataLoader == null)
            throw new ArgumentNullException(nameof(dataLoader));
        if (numTasks < 1)
            throw new ArgumentException("Number of tasks must be at least 1", nameof(numTasks));

        var accuracies = new List<double>();
        var losses = new List<double>();
        var adaptationTimes = new List<double>();

        for (int i = 0; i < numTasks; i++)
        {
            var task = dataLoader.GetNextTask();
            var metrics = AdaptAndEvaluate(task);

            accuracies.Add(metrics.QueryAccuracy);
            losses.Add(metrics.QueryLoss);
            adaptationTimes.Add(metrics.AdaptationTimeMs);
        }

        // Calculate statistics
        double meanAcc = accuracies.Average();
        double stdAcc = CalculateStandardDeviation(accuracies);
        double meanLoss = losses.Average();
        double stdLoss = CalculateStandardDeviation(losses);
        double meanTime = adaptationTimes.Average();

        // Calculate 95% confidence interval
        double marginOfError = 1.96 * stdAcc / Math.Sqrt(numTasks);
        var confidenceInterval = (meanAcc - marginOfError, meanAcc + marginOfError);

        return new MetaEvaluationMetrics
        {
            Accuracy = meanAcc,
            AccuracyStd = stdAcc,
            ConfidenceInterval = confidenceInterval,
            Loss = meanLoss,
            LossStd = stdLoss,
            NumTasks = numTasks,
            MeanAdaptationTimeMs = meanTime,
            PerTaskAccuracies = accuracies
        };
    }

    /// <inheritdoc/>
    public abstract AdaptationMetrics AdaptAndEvaluate(MetaLearningTask<T> task);

    /// <summary>
    /// High-level training method that runs multiple meta-training iterations.
    /// </summary>
    /// <param name="dataLoader">Data loader for sampling tasks.</param>
    /// <param name="numMetaIterations">Number of meta-training iterations to run.</param>
    /// <param name="tasksPerIteration">Number of tasks to sample per iteration (defaults to MetaBatchSize).</param>
    /// <returns>Metadata containing training history and statistics.</returns>
    /// <remarks>
    /// <para>
    /// This is a convenience method that wraps MetaTrainStep() for complete training runs.
    /// It tracks loss and accuracy over time, making it ideal for experiments and benchmarks.
    /// </para>
    /// <para><b>Example Usage:</b>
    /// <code>
    /// var trainer = new ReptileTrainer(model, lossFunction);
    /// var metadata = trainer.Train(dataLoader, numMetaIterations: 1000);
    /// Console.WriteLine($"Final Loss: {metadata.FinalLoss}");
    /// </code>
    /// </para>
    /// </remarks>
    public virtual MetaTrainingMetadata Train(
        IEpisodicDataLoader<T> dataLoader,
        int numMetaIterations,
        int? tasksPerIteration = null)
    {
        if (dataLoader == null)
            throw new ArgumentNullException(nameof(dataLoader));
        if (numMetaIterations < 1)
            throw new ArgumentException("Number of meta-iterations must be at least 1", nameof(numMetaIterations));

        int batchSize = tasksPerIteration ?? Configuration.MetaBatchSize;

        var lossHistory = new List<double>();
        var accuracyHistory = new List<double>();
        var startTime = System.Diagnostics.Stopwatch.StartNew();

        for (int iteration = 0; iteration < numMetaIterations; iteration++)
        {
            var metrics = MetaTrainStep(dataLoader, batchSize);
            lossHistory.Add(metrics.MetaLoss);
            accuracyHistory.Add(metrics.Accuracy);
        }

        startTime.Stop();

        return new MetaTrainingMetadata
        {
            Iterations = numMetaIterations,
            LossHistory = lossHistory,
            AccuracyHistory = accuracyHistory,
            TrainingTime = startTime.Elapsed
        };
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
    /// Validates constructor parameters.
    /// </summary>
    private void ValidateConstructorParameters(
        IFullModel<T, TInput, TOutput> metaModel,
        ILossFunction<T> lossFunction,
        int innerSteps,
        double innerLearningRate,
        double metaLearningRate)
    {
        if (metaModel == null)
            throw new ArgumentNullException(nameof(metaModel), "Meta-model cannot be null");
        if (lossFunction == null)
            throw new ArgumentNullException(nameof(lossFunction), "Loss function cannot be null");
        if (innerSteps < 1)
            throw new ArgumentException("Inner steps must be at least 1", nameof(innerSteps));
        if (innerLearningRate <= 0)
            throw new ArgumentException("Inner learning rate must be positive", nameof(innerLearningRate));
        if (metaLearningRate <= 0)
            throw new ArgumentException("Meta learning rate must be positive", nameof(metaLearningRate));
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
    /// <returns>Accuracy between 0 and 1.</returns>
    protected double ComputeAccuracy(IFullModel<T, TInput, TOutput> model, TInput inputX, TOutput targetY)
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

        return total > 0 ? (double)correct / total : 0.0;
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

    /// <summary>
    /// Calculates standard deviation of a list of values.
    /// </summary>
    private double CalculateStandardDeviation(List<double> values)
    {
        if (values.Count < 2)
            return 0.0;

        double mean = values.Average();
        double sumSquaredDiffs = values.Sum(v => Math.Pow(v - mean, 2));
        return Math.Sqrt(sumSquaredDiffs / (values.Count - 1));
    }
}
