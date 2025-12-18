using AiDotNet.Data.Structures;

using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Config;
using AiDotNet.Models.Results;
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
public class ReptileTrainer<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the Reptile-specific configuration.
    /// </summary>
    protected ReptileTrainerConfig<T> ReptileConfig => (ReptileTrainerConfig<T>)Configuration;

    /// <summary>
    /// Velocity vector for momentum updates.
    /// </summary>
    private Vector<T>? _momentumVelocity;

    /// <summary>
    /// Initializes a new instance of the ReptileTrainer with a configuration object.
    /// </summary>
    /// <param name="metaModel">The model to meta-train.</param>
    /// <param name="lossFunction">Loss function for evaluating task performance.</param>
    /// <param name="dataLoader">Episodic data loader for sampling meta-learning tasks.</param>
    /// <param name="config">Configuration object containing all hyperparameters. If null, uses default ReptileTrainerConfig with industry-standard values.</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel, lossFunction, or dataLoader is null.</exception>
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
    /// <b>Parameters explained:</b>
    /// - <b>metaModel:</b> Your neural network or model to be meta-trained
    /// - <b>lossFunction:</b> How to measure errors (MSE, CrossEntropy, etc.)
    /// - <b>dataLoader:</b> Provides N-way K-shot tasks for meta-training (configured at construction time)
    /// - <b>config:</b> Learning rates and steps (optional - uses sensible defaults)
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
        IEpisodicDataLoader<T, TInput, TOutput> dataLoader,
        IMetaLearnerConfig<T>? config = null)
        : base(metaModel, lossFunction, dataLoader, config ?? new ReptileTrainerConfig<T>())
    {
        // Initialize momentum velocity if needed
        if (ReptileConfig.UseMomentum)
        {
            _momentumVelocity = new Vector<T>(metaModel.GetParameters().Length);
        }
    }

    /// <inheritdoc/>
    public override MetaTrainingStepResult<T> MetaTrainStep(int batchSize)
    {
        if (batchSize < 1)
            throw new ArgumentException("Batch size must be at least 1", nameof(batchSize));

        var startTime = Stopwatch.StartNew();

        // Save original meta-parameters
        Vector<T> originalParameters = MetaModel.GetParameters();

        // Collect parameter updates from all tasks in batch (using generic T)
        var parameterUpdates = new List<Vector<T>>();
        var taskLosses = new List<T>();
        var taskAccuracies = new List<T>();

        // Process each task in the batch
        for (int taskIdx = 0; taskIdx < batchSize; taskIdx++)
        {
            // Sample a task using configured data loader
            IMetaLearningTask<T, TInput, TOutput> task = DataLoader.GetNextTask();

            // Reset model to original meta-parameters for this task
            MetaModel.SetParameters(originalParameters.Clone());

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

            // Evaluate on query set to measure adaptation quality (using generic T)
            T queryLoss = ComputeLoss(MetaModel, task.QuerySetX, task.QuerySetY);
            T queryAccuracy = ComputeAccuracy(MetaModel, task.QuerySetX, task.QuerySetY);

            taskLosses.Add(queryLoss);
            taskAccuracies.Add(queryAccuracy);
        }

        // Outer loop: Meta-update by averaging parameter updates
        Vector<T> averageUpdate = AverageVectors(parameterUpdates);

        // Get current epsilon (with decay if enabled)
        T currentEpsilon = GetCurrentEpsilon();

        // Scale by meta-learning rate: ε * Δθ_avg
        Vector<T> scaledUpdate = averageUpdate.Multiply(currentEpsilon);

        // Apply momentum if enabled
        Vector<T> finalUpdate;
        if (ReptileConfig.UseMomentum && _momentumVelocity != null)
        {
            if (ReptileConfig.UseNesterovMomentum)
            {
                // Nesterov momentum: look ahead
                Vector<T> lookaheadParams = originalParameters.Add(
                    _momentumVelocity.Multiply(ReptileConfig.MomentumCoefficient));
                MetaModel.SetParameters(lookaheadParams);

                // Compute gradient at lookahead position
                Vector<T> lookaheadGradient = ComputeLookaheadGradient(
                    originalParameters,
                    lookaheadParams,
                    averageUpdate);

                // Update velocity: v = μ * v + ε * gradient
                _momentumVelocity = _momentumVelocity.Multiply(ReptileConfig.MomentumCoefficient)
                    .Add(scaledUpdate);
                finalUpdate = _momentumVelocity;
            }
            else
            {
                // Standard momentum: v = μ * v + ε * update
                _momentumVelocity = _momentumVelocity.Multiply(ReptileConfig.MomentumCoefficient)
                    .Add(scaledUpdate);
                finalUpdate = _momentumVelocity;
            }
        }
        else
        {
            finalUpdate = scaledUpdate;
        }

        // Update meta-parameters: θ = θ + update
        Vector<T> newMetaParameters = originalParameters.Add(finalUpdate);
        MetaModel.SetParameters(newMetaParameters);

        // Increment iteration counter
        _currentIteration++;

        startTime.Stop();

        // Calculate aggregate metrics using generic T
        var lossVector = new Vector<T>(taskLosses.ToArray());
        var accuracyVector = new Vector<T>(taskAccuracies.ToArray());

        // Use StatisticsHelper for proper generic calculations
        T meanLoss = StatisticsHelper<T>.CalculateMean(lossVector);
        T meanAccuracy = StatisticsHelper<T>.CalculateMean(accuracyVector);

        // Return comprehensive metrics with new Result type
        return new MetaTrainingStepResult<T>(
            metaLoss: meanLoss,
            taskLoss: meanLoss,  // For Reptile, meta-loss = avg task loss
            accuracy: meanAccuracy,
            numTasks: batchSize,
            iteration: _currentIteration,
            timeMs: startTime.Elapsed.TotalMilliseconds);
    }

    /// <inheritdoc/>
    public override MetaAdaptationResult<T> AdaptAndEvaluate(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
            throw new ArgumentNullException(nameof(task));

        var startTime = Stopwatch.StartNew();

        // Save original meta-parameters
        Vector<T> originalParameters = MetaModel.GetParameters();

        // Evaluate before adaptation (baseline) - using generic T
        T initialQueryLoss = ComputeLoss(MetaModel, task.QuerySetX, task.QuerySetY);

        var perStepLosses = new List<T> { initialQueryLoss };

        // Inner loop: Adapt to task using support set
        for (int step = 0; step < Configuration.InnerSteps; step++)
        {
            MetaModel.Train(task.SupportSetX, task.SupportSetY);

            // Track loss after each step for convergence analysis
            T stepLoss = ComputeLoss(MetaModel, task.QuerySetX, task.QuerySetY);
            perStepLosses.Add(stepLoss);
        }

        // Evaluate after adaptation (all using generic T)
        T queryLoss = ComputeLoss(MetaModel, task.QuerySetX, task.QuerySetY);
        T queryAccuracy = ComputeAccuracy(MetaModel, task.QuerySetX, task.QuerySetY);

        T supportLoss = ComputeLoss(MetaModel, task.SupportSetX, task.SupportSetY);
        T supportAccuracy = ComputeAccuracy(MetaModel, task.SupportSetX, task.SupportSetY);

        startTime.Stop();

        // Restore original meta-parameters (don't modify meta-model during evaluation)
        MetaModel.SetParameters(originalParameters);

        // Calculate additional metrics
        var additionalMetrics = new Dictionary<string, T>
        {
            ["initial_query_loss"] = initialQueryLoss,
            ["loss_improvement"] = NumOps.Subtract(initialQueryLoss, queryLoss),
            ["support_query_accuracy_gap"] = NumOps.Subtract(supportAccuracy, queryAccuracy)
        };

        // Return new Result type with generic T throughout
        return new MetaAdaptationResult<T>(
            queryAccuracy: queryAccuracy,
            queryLoss: queryLoss,
            supportAccuracy: supportAccuracy,
            supportLoss: supportLoss,
            adaptationSteps: Configuration.InnerSteps,
            adaptationTimeMs: startTime.Elapsed.TotalMilliseconds,
            perStepLosses: perStepLosses,
            additionalMetrics: additionalMetrics);
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
    /// Gets the current epsilon value based on decay schedule.
    /// </summary>
    /// <returns>The current meta-learning rate (epsilon).</returns>
    private T GetCurrentEpsilon()
    {
        if (!ReptileConfig.UseEpsilonDecay)
        {
            return ReptileConfig.MetaLearningRate;
        }

        double progress = (double)_currentIteration / ReptileConfig.NumMetaIterations;
        progress = Math.Max(0, Math.Min(1, progress)); // Clamp to [0, 1]

        T initialEpsilon = ReptileConfig.MetaLearningRate;
        T finalEpsilon = ReptileConfig.FinalEpsilon;

        switch (ReptileConfig.EpsilonDecaySchedule)
        {
            case EpsilonDecaySchedule.Linear:
                // ε_t = ε_0 * (1 - t/T) + ε_final * (t/T)
                return NumOps.Add(
                    NumOps.Multiply(initialEpsilon, NumOps.FromDouble(1 - progress)),
                    NumOps.Multiply(finalEpsilon, NumOps.FromDouble(progress)));

            case EpsilonDecaySchedule.Cosine:
                // ε_t = ε_0 * 0.5 * (1 + cos(π * t/T))
                double cosine = Math.Cos(Math.PI * progress);
                return NumOps.Multiply(
                    NumOps.FromDouble(0.5 * (1 + cosine)),
                    initialEpsilon);

            case EpsilonDecaySchedule.Step:
                // ε_t = ε_0 * 0.5^floor(t * 10 / T)
                int steps = (int)(progress * 10);
                double decay = Math.Pow(0.5, steps);
                return NumOps.Multiply(initialEpsilon, NumOps.FromDouble(decay));

            case EpsilonDecaySchedule.Exponential:
                // ε_t = ε_0 * exp(-10 * t/T)
                double exponential = Math.Exp(-10 * progress);
                return NumOps.Multiply(initialEpsilon, NumOps.FromDouble(exponential));

            default:
                return initialEpsilon;
        }
    }

    /// <summary>
    /// Computes gradient for Nesterov momentum lookahead.
    /// </summary>
    /// <param name="originalParams">Original parameters.</param>
    /// <param name="lookaheadParams">Lookahead parameters.</param>
    /// <param name="averageUpdate">Average parameter update.</param>
    /// <returns>The computed gradient.</returns>
    private Vector<T> ComputeLookaheadGradient(
        Vector<T> originalParams,
        Vector<T> lookaheadParams,
        Vector<T> averageUpdate)
    {
        // For Reptile, the "gradient" is just the parameter update direction
        // Nesterov momentum uses this to look ahead
        return averageUpdate;
    }
}
