using AiDotNet.Data.Abstractions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Config;
using AiDotNet.MetaLearning.Metrics;
using System.Diagnostics;

namespace AiDotNet.MetaLearning.Trainers;

/// <summary>
/// Production-ready implementation of the Reptile meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <remarks>
/// <para>
/// Reptile (Nichol et al., 2018) is a simple and effective first-order meta-learning algorithm.
/// Unlike MAML which requires second-order derivatives, Reptile simply moves meta-parameters
/// toward task-adapted parameters, making it computationally efficient while maintaining
/// strong few-shot learning performance.
/// </para>
/// <para><b>Algorithm - Reptile with Batch Meta-Training:</b>
/// <code>
/// Initialize: θ (meta-parameters)
///
/// for iteration = 1 to N:
///     # Sample batch of tasks for this meta-update
///     tasks = SampleTasks(batch_size)
///
///     # Collect parameter updates from all tasks
///     updates = []
///     for each task in tasks:
///         # Clone current meta-parameters
///         θ_i = Clone(θ)
///
///         # Inner loop: Adapt to this task
///         for step = 1 to K:
///             θ_i = θ_i - α * ∇L(θ_i, support_set)
///
///         # Record the parameter change
///         Δθ_i = θ_i - θ
///         updates.append(Δθ_i)
///
///     # Outer loop: Meta-update averages all task adaptations
///     Δθ_avg = Average(updates)
///     θ = θ + ε * Δθ_avg
///
/// return θ
/// </code>
/// </para>
/// <para><b>Why This Works:</b>
///
/// By repeatedly moving toward task-adapted parameters, θ naturally converges to a region
/// of parameter space where:
/// 1. Many tasks can be solved with few gradient steps
/// 2. The loss surface is smooth and easy to navigate
/// 3. Small parameter changes lead to effective task-specific solutions
///
/// This creates an initialization that is "pre-adapted" for rapid fine-tuning.
/// </para>
/// <para><b>Production Features:</b>
/// - Batch meta-training for stable gradient estimates
/// - Progress monitoring with detailed metrics
/// - Memory-efficient parameter cloning
/// - Support for any model implementing IFullModel
/// - Comprehensive error handling and validation
/// </para>
/// </remarks>
public class ReptileTrainer<T, TInput, TOutput> : ReptileTrainerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of the ReptileTrainer with a configuration object.
    /// </summary>
    /// <param name="metaModel">The model to meta-train.</param>
    /// <param name="lossFunction">Loss function for evaluating task performance.</param>
    /// <param name="config">Configuration object containing all hyperparameters. If null, uses default ReptileTrainerConfig with industry-standard values.</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel or lossFunction is null.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration validation fails.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a Reptile trainer ready for meta-learning.
    ///
    /// Reptile is simpler than MAML but equally effective for few-shot learning. It works by:
    /// 1. Cloning the meta-model for each task
    /// 2. Training the clone on a few examples (inner loop)
    /// 3. Moving the meta-model toward the trained clone (outer loop)
    /// 4. Repeating for many tasks
    ///
    /// After meta-training, your model can quickly adapt to new tasks with very few examples.
    ///
    /// <b>Default configuration (if null):</b>
    /// - Inner learning rate: 0.01 (how fast the model adapts to each task)
    /// - Meta learning rate: 0.001 (how fast meta-parameters update)
    /// - Inner steps: 5 (gradient steps per task)
    /// - Meta batch size: 1 (tasks processed per meta-update)
    /// </para>
    /// </remarks>
    public ReptileTrainer(
        IFullModel<T, TInput, TOutput> metaModel,
        ILossFunction<T> lossFunction,
        IMetaLearnerConfig<T>? config = null)
        : base(metaModel, lossFunction, config)
    {
    }

    /// <inheritdoc/>
    public override MetaTrainingMetrics MetaTrainStep(IEpisodicDataLoader<T> dataLoader, int batchSize)
    {
        if (dataLoader == null)
            throw new ArgumentNullException(nameof(dataLoader));
        if (batchSize < 1)
            throw new ArgumentException("Batch size must be at least 1", nameof(batchSize));

        var startTime = Stopwatch.StartNew();

        // Save original meta-parameters
        Vector<T> originalParameters = MetaModel.GetParameters();

        // Collect parameter updates from all tasks in batch
        var parameterUpdates = new List<Vector<T>>();
        var taskLosses = new List<double>();
        var taskAccuracies = new List<double>();

        // Process each task in the batch
        for (int taskIdx = 0; taskIdx < batchSize; taskIdx++)
        {
            // Sample a task
            MetaLearningTask<T> task = dataLoader.GetNextTask();

            // Reset model to original meta-parameters for this task
            MetaModel.SetParameters(originalParameters.Copy());

            // Inner loop: Adapt to this task using support set
            for (int step = 0; step < Configuration.InnerSteps; step++)
            {
                MetaModel.Train(task.SupportSetX, task.SupportSetY);
            }

            // Get adapted parameters after inner loop
            Vector<T> adaptedParameters = MetaModel.GetParameters();

            // Compute parameter update: Δθ = θ_adapted - θ_original
            Vector<T> parameterUpdate = adaptedParameters.Subtract(originalParameters);
            parameterUpdates.Add(parameterUpdate);

            // Evaluate on query set to measure adaptation quality
            double queryLoss = NumOps.ToDouble(ComputeLoss(MetaModel, task.QuerySetX, task.QuerySetY));
            double queryAccuracy = ComputeAccuracy(MetaModel, task.QuerySetX, task.QuerySetY);

            taskLosses.Add(queryLoss);
            taskAccuracies.Add(queryAccuracy);
        }

        // Outer loop: Meta-update by averaging parameter updates
        Vector<T> averageUpdate = AverageVectors(parameterUpdates);

        // Scale by meta-learning rate: ε * Δθ_avg
        Vector<T> scaledUpdate = averageUpdate.Multiply(Configuration.MetaLearningRate);

        // Update meta-parameters: θ = θ + ε * Δθ_avg
        Vector<T> newMetaParameters = originalParameters.Add(scaledUpdate);
        MetaModel.SetParameters(newMetaParameters);

        // Increment iteration counter
        _currentIteration++;

        startTime.Stop();

        // Return comprehensive metrics
        return new MetaTrainingMetrics
        {
            MetaLoss = taskLosses.Average(),
            TaskLoss = taskLosses.Average(),
            Accuracy = taskAccuracies.Average(),
            NumTasks = batchSize,
            Iteration = _currentIteration,
            TimeMs = startTime.Elapsed.TotalMilliseconds,
            AdditionalMetrics = new Dictionary<string, double>
            {
                ["accuracy_std"] = CalculateStdDev(taskAccuracies),
                ["loss_std"] = CalculateStdDev(taskLosses),
                ["min_accuracy"] = taskAccuracies.Min(),
                ["max_accuracy"] = taskAccuracies.Max(),
                ["min_loss"] = taskLosses.Min(),
                ["max_loss"] = taskLosses.Max()
            }
        };
    }

    /// <inheritdoc/>
    public override AdaptationMetrics AdaptAndEvaluate(MetaLearningTask<T> task)
    {
        if (task == null)
            throw new ArgumentNullException(nameof(task));

        var startTime = Stopwatch.StartNew();

        // Save original meta-parameters
        Vector<T> originalParameters = MetaModel.GetParameters();

        // Evaluate before adaptation (baseline)
        double initialQueryLoss = NumOps.ToDouble(ComputeLoss(MetaModel, task.QuerySetX, task.QuerySetY));

        // Inner loop: Adapt to task using support set
        for (int step = 0; step < Configuration.InnerSteps; step++)
        {
            MetaModel.Train(task.SupportSetX, task.SupportSetY);
        }

        // Evaluate after adaptation
        double queryLoss = NumOps.ToDouble(ComputeLoss(MetaModel, task.QuerySetX, task.QuerySetY));
        double queryAccuracy = ComputeAccuracy(MetaModel, task.QuerySetX, task.QuerySetY);

        double supportLoss = NumOps.ToDouble(ComputeLoss(MetaModel, task.SupportSetX, task.SupportSetY));
        double supportAccuracy = ComputeAccuracy(MetaModel, task.SupportSetX, task.SupportSetY);

        startTime.Stop();

        // Restore original meta-parameters (don't modify meta-model during evaluation)
        MetaModel.SetParameters(originalParameters);

        return new AdaptationMetrics
        {
            QueryAccuracy = queryAccuracy,
            QueryLoss = queryLoss,
            SupportAccuracy = supportAccuracy,
            SupportLoss = supportLoss,
            AdaptationSteps = Configuration.InnerSteps,
            AdaptationTimeMs = startTime.Elapsed.TotalMilliseconds,
            TaskId = task.TaskId ?? "unknown",
            AdditionalMetrics = new Dictionary<string, double>
            {
                ["initial_query_loss"] = initialQueryLoss,
                ["loss_improvement"] = initialQueryLoss - queryLoss,
                ["loss_improvement_ratio"] = initialQueryLoss > 0 ? (initialQueryLoss - queryLoss) / initialQueryLoss : 0,
                ["support_query_accuracy_gap"] = supportAccuracy - queryAccuracy
            }
        };
    }

    /// <summary>
    /// Averages a list of vectors element-wise.
    /// </summary>
    private Vector<T> AverageVectors(List<Vector<T>> vectors)
    {
        if (vectors.Count == 0)
            throw new ArgumentException("Cannot average empty list of vectors");

        int dimension = vectors[0].Length;
        var result = new Vector<T>(dimension);

        // Sum all vectors
        foreach (var vector in vectors)
        {
            if (vector.Length != dimension)
                throw new ArgumentException("All vectors must have the same dimension");

            result = result.Add(vector);
        }

        // Divide by count to get average
        T divisor = NumOps.FromDouble(vectors.Count);
        result = result.Divide(divisor);

        return result;
    }

    /// <summary>
    /// Calculates standard deviation of a list of values.
    /// </summary>
    private double CalculateStdDev(List<double> values)
    {
        if (values.Count < 2)
            return 0.0;

        double mean = values.Average();
        double sumSquaredDiffs = values.Sum(v => Math.Pow(v - mean, 2));
        return Math.Sqrt(sumSquaredDiffs / (values.Count - 1));
    }
}
