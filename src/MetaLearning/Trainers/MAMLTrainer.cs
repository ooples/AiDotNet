using AiDotNet.Data.Abstractions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Config;
using AiDotNet.Models.Results;
using System.Diagnostics;

namespace AiDotNet.MetaLearning.Trainers;

/// <summary>
/// Production-ready implementation of the MAML (Model-Agnostic Meta-Learning) algorithm with full second-order support.
/// </summary>
/// <typeparam name="T">The numeric data type (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <remarks>
/// <para>
/// <b>MAML (Finn et al., 2017)</b> is a second-order meta-learning algorithm that learns optimal parameter
/// initializations for rapid adaptation to new tasks with minimal data.
/// </para>
/// <para><b>Key Innovation:</b>
/// Unlike Reptile which moves toward adapted parameters, MAML explicitly optimizes for adaptability
/// by computing gradients of post-adaptation performance with respect to initial parameters.
/// This requires computing gradients through the adaptation process (second-order derivatives).
/// </para>
/// <para><b>Algorithm - Full MAML (Second-Order):</b>
/// <code>
/// Initialize: θ (meta-parameters)
///
/// for iteration = 1 to N:
///     # Sample batch of tasks
///     tasks = SampleTasks(batch_size)
///     meta_gradients = []
///
///     for each task in tasks:
///         # Inner loop: Adapt on support set
///         φ = θ  # Clone meta-parameters
///         for k = 1 to K:
///             φ = φ - α∇L_support(φ)  # Task adaptation
///
///         # Compute meta-gradient by backpropagating through adaptation
///         # This is the key difference from Reptile!
///         ∇θ = ∂L_query(φ)/∂θ  # Gradient w.r.t. ORIGINAL parameters
///         meta_gradients.append(∇θ)
///
///     # Outer loop: Meta-update
///     θ = θ - β * Average(meta_gradients)
///
/// return θ
/// </code>
/// </para>
/// <para><b>First-Order MAML (FOMAML):</b>
/// Approximates full MAML by ignoring second-order terms:
/// ∇θ ≈ ∂L_query(φ)/∂φ  (gradient w.r.t. adapted parameters)
///
/// This is computationally cheaper and often performs similarly to full MAML.
/// The original paper (Finn et al., 2017) showed only minor performance differences.
/// </para>
/// <para><b>Production Features:</b>
/// - Full second-order MAML when model implements IGradientComputable
/// - FOMAML (first-order) approximation for efficiency
/// - Automatic fallback to Reptile-style for models without gradient support
/// - Gradient clipping for training stability
/// - Adam meta-optimizer for faster convergence
/// - Per-parameter adaptive learning rates
/// - Comprehensive error handling and validation
/// - Detailed performance metrics and monitoring
/// </para>
/// <para><b>When to Use MAML vs Reptile:</b>
/// - <b>Use MAML:</b> When you need maximum few-shot performance and can afford computation
/// - <b>Use Reptile:</b> When you need simpler implementation and faster training
/// - <b>Use FOMAML:</b> Best of both worlds - nearly MAML performance at Reptile speed
/// </para>
/// </remarks>
public class MAMLTrainer<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the MAML-specific configuration.
    /// </summary>
    protected MAMLTrainerConfig<T> MAMLConfig => (MAMLTrainerConfig<T>)Configuration;

    /// <summary>
    /// Indicates whether the model supports explicit gradient computation.
    /// </summary>
    private readonly bool _supportsGradientComputation;

    /// <summary>
    /// Optional meta-optimizer for outer loop updates. If null, uses built-in Adam implementation.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, TInput, TOutput>? _metaOptimizer;

    /// <summary>
    /// Optional inner optimizer for task adaptation. If null, uses SGD with inner learning rate.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, TInput, TOutput>? _innerOptimizer;

    // Adam optimizer state for meta-updates (used only when _metaOptimizer is null)
    private Vector<T>? _adamFirstMoment;  // m_t in Adam
    private Vector<T>? _adamSecondMoment; // v_t in Adam
    private int _adamTimeStep = 0;

    /// <summary>
    /// Initializes a new instance of the MAMLTrainer with a configuration object.
    /// </summary>
    /// <param name="metaModel">The model to meta-train.</param>
    /// <param name="lossFunction">Loss function for evaluating task performance.</param>
    /// <param name="dataLoader">Episodic data loader for sampling meta-learning tasks.</param>
    /// <param name="config">Configuration object containing all hyperparameters. If null, uses default MAMLTrainerConfig.</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel, lossFunction, or dataLoader is null.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration validation fails.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a MAML trainer ready for few-shot meta-learning.
    ///
    /// MAML learns optimal starting points by explicitly optimizing for rapid adaptation.
    /// After meta-training, your model can adapt to new tasks with just a few examples.
    ///
    /// <b>How MAML works:</b>
    /// 1. Start with meta-parameters (θ)
    /// 2. For each task: adapt θ to get task-specific parameters (φ)
    /// 3. Measure how well φ performs on held-out query data
    /// 4. Update θ to make future adaptations better
    /// 5. The key: compute gradients that account for the adaptation process
    ///
    /// <b>Parameters explained:</b>
    /// - <b>metaModel:</b> Your neural network or model to be meta-trained
    /// - <b>lossFunction:</b> How to measure errors (MSE, CrossEntropy, etc.)
    /// - <b>dataLoader:</b> Provides N-way K-shot tasks for meta-training
    /// - <b>config:</b> Learning rates, steps, and optimization settings
    ///
    /// <b>Default configuration (if null):</b>
    /// - Inner learning rate: 0.01 (task adaptation speed)
    /// - Meta learning rate: 0.001 (meta-parameter update speed)
    /// - Inner steps: 5 (adaptation steps per task)
    /// - Meta batch size: 4 (tasks per meta-update)
    /// - Use FOMAML: true (first-order approximation for speed)
    /// - Gradient clipping: 10.0 (for stability)
    /// - Adam meta-optimizer: enabled (for faster convergence)
    /// </para>
    /// </remarks>
    public MAMLTrainer(
        IFullModel<T, TInput, TOutput> metaModel,
        ILossFunction<T> lossFunction,
        IEpisodicDataLoader<T, TInput, TOutput> dataLoader,
        IMetaLearnerConfig<T>? config = null)
        : base(metaModel, lossFunction, dataLoader, config ?? new MAMLTrainerConfig<T>())
    {
        // Validate that config is actually a MAMLTrainerConfig
        if (Configuration is not MAMLTrainerConfig<T>)
        {
            throw new ArgumentException(
                $"Configuration must be of type MAMLTrainerConfig<T>, but was {Configuration.GetType().Name}",
                nameof(config));
        }

        // Check if model supports gradient computation
        _supportsGradientComputation = metaModel is IGradientComputable<T, TInput, TOutput>;

        // Initialize optimizers as null (will use built-in implementation)
        _metaOptimizer = null;
        _innerOptimizer = null;
    }

    /// <summary>
    /// Initializes a new instance of the MAMLTrainer with custom optimizers.
    /// </summary>
    /// <param name="metaModel">The model to meta-train.</param>
    /// <param name="lossFunction">Loss function for evaluating task performance.</param>
    /// <param name="dataLoader">Episodic data loader for sampling meta-learning tasks.</param>
    /// <param name="metaOptimizer">Optional custom meta-optimizer for outer loop updates. If null, uses built-in Adam.</param>
    /// <param name="innerOptimizer">Optional custom inner optimizer for task adaptation. If null, uses SGD with inner learning rate.</param>
    /// <param name="config">Configuration object containing all hyperparameters. If null, uses default MAMLTrainerConfig.</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel, lossFunction, or dataLoader is null.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration validation fails.</exception>
    /// <remarks>
    /// This constructor allows you to inject custom optimizers instead of relying on the built-in Adam implementation.
    /// When metaOptimizer is provided, the Adam-related settings in config will be ignored.
    /// When innerOptimizer is provided, the InnerLearningRate in config will be managed by the optimizer.
    /// </remarks>
    public MAMLTrainer(
        IFullModel<T, TInput, TOutput> metaModel,
        ILossFunction<T> lossFunction,
        IEpisodicDataLoader<T, TInput, TOutput> dataLoader,
        IGradientBasedOptimizer<T, TInput, TOutput>? metaOptimizer,
        IGradientBasedOptimizer<T, TInput, TOutput>? innerOptimizer = null,
        IMetaLearnerConfig<T>? config = null)
        : base(metaModel, lossFunction, dataLoader, config ?? new MAMLTrainerConfig<T>())
    {
        // Validate that config is actually a MAMLTrainerConfig
        if (Configuration is not MAMLTrainerConfig<T>)
        {
            throw new ArgumentException(
                $"Configuration must be of type MAMLTrainerConfig<T>, but was {Configuration.GetType().Name}",
                nameof(config));
        }

        // Check if model supports gradient computation
        _supportsGradientComputation = metaModel is IGradientComputable<T, TInput, TOutput>;

        // Store provided optimizers
        _metaOptimizer = metaOptimizer;
        _innerOptimizer = innerOptimizer;
    }

    /// <inheritdoc/>
    public override MetaTrainingStepResult<T> MetaTrainStep(int batchSize)
    {
        if (batchSize < 1)
            throw new ArgumentException("Batch size must be at least 1", nameof(batchSize));

        var startTime = Stopwatch.StartNew();

        // Save original meta-parameters
        Vector<T> originalParameters = MetaModel.GetParameters();
        int paramCount = originalParameters.Length;

        // Initialize Adam optimizer state if needed
        if (MAMLConfig.UseAdaptiveMetaOptimizer && _adamFirstMoment == null)
        {
            _adamFirstMoment = new Vector<T>(new T[paramCount]);
            _adamSecondMoment = new Vector<T>(new T[paramCount]);
        }

        // Collect meta-gradients from all tasks in batch
        var metaGradients = new List<Vector<T>>();
        var taskLosses = new List<T>();
        var taskAccuracies = new List<T>();

        // Process each task in the batch
        for (int taskIdx = 0; taskIdx < batchSize; taskIdx++)
        {
            // Sample a task using configured data loader
            MetaLearningTask<T, TInput, TOutput> task = DataLoader.GetNextTask();

            // Compute meta-gradient for this task
            Vector<T> metaGradient = ComputeTaskMetaGradient(task, originalParameters, out T queryLoss, out T queryAccuracy);

            metaGradients.Add(metaGradient);
            taskLosses.Add(queryLoss);
            taskAccuracies.Add(queryAccuracy);
        }

        // Average meta-gradients across tasks
        Vector<T> averageMetaGradient = AverageVectors(metaGradients);

        // Apply gradient clipping if enabled
        if (Convert.ToDouble(MAMLConfig.MaxGradientNorm) > 0)
        {
            averageMetaGradient = ClipGradientByNorm(averageMetaGradient, MAMLConfig.MaxGradientNorm);
        }

        // Apply meta-update using appropriate optimizer
        Vector<T> newMetaParameters;
        if (_metaOptimizer != null)
        {
            // Use provided meta-optimizer
            newMetaParameters = _metaOptimizer.UpdateParameters(originalParameters, averageMetaGradient);
        }
        else if (MAMLConfig.UseAdaptiveMetaOptimizer)
        {
            // Use built-in Adam implementation
            newMetaParameters = AdamMetaUpdate(originalParameters, averageMetaGradient);
        }
        else
        {
            // Vanilla SGD: θ = θ - β * ∇θ
            Vector<T> scaledGradient = averageMetaGradient.Multiply(Configuration.MetaLearningRate);
            newMetaParameters = originalParameters.Subtract(scaledGradient);
        }

        MetaModel.SetParameters(newMetaParameters);

        // Increment iteration counter
        _currentIteration++;

        startTime.Stop();

        // Calculate aggregate metrics
        var lossVector = new Vector<T>(taskLosses.ToArray());
        var accuracyVector = new Vector<T>(taskAccuracies.ToArray());

        T meanLoss = StatisticsHelper<T>.CalculateMean(lossVector);
        T meanAccuracy = StatisticsHelper<T>.CalculateMean(accuracyVector);

        // Return comprehensive metrics
        return new MetaTrainingStepResult<T>(
            metaLoss: meanLoss,
            taskLoss: meanLoss,  // For MAML, meta-loss = avg query loss
            accuracy: meanAccuracy,
            numTasks: batchSize,
            iteration: _currentIteration,
            timeMs: startTime.Elapsed.TotalMilliseconds);
    }

    /// <summary>
    /// Computes the meta-gradient for a single task using appropriate method based on configuration and model capabilities.
    /// </summary>
    private Vector<T> ComputeTaskMetaGradient(
        MetaLearningTask<T, TInput, TOutput> task,
        Vector<T> originalParameters,
        out T queryLoss,
        out T queryAccuracy)
    {
        // Check if we can use true MAML with gradient computation
        if (_supportsGradientComputable && !MAMLConfig.UseFirstOrderApproximation)
        {
            // Full second-order MAML
            return ComputeSecondOrderMetaGradient(task, originalParameters, out queryLoss, out queryAccuracy);
        }
        else if (_supportsGradientComputable && MAMLConfig.UseFirstOrderApproximation)
        {
            // FOMAML (First-Order MAML)
            return ComputeFirstOrderMetaGradient(task, originalParameters, out queryLoss, out queryAccuracy);
        }
        else
        {
            // Fallback to Reptile-style approximation for models without gradient support
            return ComputeReptileStyleApproximation(task, originalParameters, out queryLoss, out queryAccuracy);
        }
    }

    /// <summary>
    /// Computes full second-order MAML meta-gradient by backpropagating through adaptation.
    /// </summary>
    private Vector<T> ComputeSecondOrderMetaGradient(
        MetaLearningTask<T, TInput, TOutput> task,
        Vector<T> originalParameters,
        out T queryLoss,
        out T queryAccuracy)
    {
        var gradientModel = (IGradientComputable<T, TInput, TOutput>)MetaModel;

        // Reset to original parameters
        MetaModel.SetParameters(originalParameters.Clone());

        // Build adaptation trajectory for backpropagation
        var adaptationSteps = new List<(TInput input, TOutput target)>();
        adaptationSteps.Add((task.SupportSetX, task.SupportSetY));

        // Compute second-order gradient through adaptation
        Vector<T> metaGradient = gradientModel.ComputeSecondOrderGradients(
            adaptationSteps,
            task.QuerySetX,
            task.QuerySetY,
            LossFunction,
            Configuration.InnerLearningRate);

        // Evaluate for metrics (after adaptation)
        for (int step = 0; step < Configuration.InnerSteps; step++)
        {
            MetaModel.Train(task.SupportSetX, task.SupportSetY);
        }

        queryLoss = ComputeLoss(MetaModel, task.QuerySetX, task.QuerySetY);
        queryAccuracy = ComputeAccuracy(MetaModel, task.QuerySetX, task.QuerySetY);

        // Reset to original parameters
        MetaModel.SetParameters(originalParameters);

        return metaGradient;
    }

    /// <summary>
    /// Computes first-order MAML (FOMAML) meta-gradient.
    /// </summary>
    private Vector<T> ComputeFirstOrderMetaGradient(
        MetaLearningTask<T, TInput, TOutput> task,
        Vector<T> originalParameters,
        out T queryLoss,
        out T queryAccuracy)
    {
        var gradientModel = (IGradientComputable<T, TInput, TOutput>)MetaModel;

        // Reset to original parameters
        MetaModel.SetParameters(originalParameters.Clone());

        // Inner loop: Adapt on support set
        for (int step = 0; step < Configuration.InnerSteps; step++)
        {
            MetaModel.Train(task.SupportSetX, task.SupportSetY);
        }

        // FOMAML: Compute gradient of query loss w.r.t. adapted parameters
        // (ignoring the derivative through the adaptation process)
        Vector<T> metaGradient = gradientModel.ComputeGradients(
            task.QuerySetX,
            task.QuerySetY,
            LossFunction);

        // Evaluate for metrics
        queryLoss = ComputeLoss(MetaModel, task.QuerySetX, task.QuerySetY);
        queryAccuracy = ComputeAccuracy(MetaModel, task.QuerySetX, task.QuerySetY);

        // Reset to original parameters
        MetaModel.SetParameters(originalParameters);

        return metaGradient;
    }

    /// <summary>
    /// Computes Reptile-style approximation for models without explicit gradient support.
    /// </summary>
    private Vector<T> ComputeReptileStyleApproximation(
        MetaLearningTask<T, TInput, TOutput> task,
        Vector<T> originalParameters,
        out T queryLoss,
        out T queryAccuracy)
    {
        // Reset to original parameters
        MetaModel.SetParameters(originalParameters.Clone());

        // Inner loop: Adapt on support set
        for (int step = 0; step < Configuration.InnerSteps; step++)
        {
            MetaModel.Train(task.SupportSetX, task.SupportSetY);
        }

        // Get adapted parameters
        Vector<T> adaptedParameters = MetaModel.GetParameters();

        // Evaluate on query set
        queryLoss = ComputeLoss(MetaModel, task.QuerySetX, task.QuerySetY);
        queryAccuracy = ComputeAccuracy(MetaModel, task.QuerySetX, task.QuerySetY);

        // Approximate gradient as parameter difference / learning rate
        // This is the Reptile update, used as fallback
        Vector<T> parameterDifference = adaptedParameters.Subtract(originalParameters);
        Vector<T> approximateGradient = parameterDifference.Divide(Configuration.InnerLearningRate);

        // Reset to original parameters
        MetaModel.SetParameters(originalParameters);

        return approximateGradient;
    }

    /// <summary>
    /// Applies Adam meta-optimization update.
    /// </summary>
    private Vector<T> AdamMetaUpdate(Vector<T> parameters, Vector<T> gradient)
    {
        _adamTimeStep++;

        // Update biased first moment estimate: m_t = β1 * m_{t-1} + (1 - β1) * g_t
        _adamFirstMoment = _adamFirstMoment!.Multiply(MAMLConfig.AdamBeta1)
            .Add(gradient.Multiply(NumOps.Subtract(NumOps.FromDouble(1.0), MAMLConfig.AdamBeta1)));

        // Update biased second moment estimate: v_t = β2 * v_{t-1} + (1 - β2) * g_t²
        Vector<T> gradientSquared = ElementwiseMultiply(gradient, gradient);
        _adamSecondMoment = _adamSecondMoment!.Multiply(MAMLConfig.AdamBeta2)
            .Add(gradientSquared.Multiply(NumOps.Subtract(NumOps.FromDouble(1.0), MAMLConfig.AdamBeta2)));

        // Compute bias-corrected first moment: m̂_t = m_t / (1 - β1^t)
        T beta1Power = NumOps.Power(MAMLConfig.AdamBeta1, NumOps.FromDouble(_adamTimeStep));
        Vector<T> mHat = _adamFirstMoment.Divide(NumOps.Subtract(NumOps.FromDouble(1.0), beta1Power));

        // Compute bias-corrected second moment: v̂_t = v_t / (1 - β2^t)
        T beta2Power = NumOps.Power(MAMLConfig.AdamBeta2, NumOps.FromDouble(_adamTimeStep));
        Vector<T> vHat = _adamSecondMoment.Divide(NumOps.Subtract(NumOps.FromDouble(1.0), beta2Power));

        // Compute Adam update: θ = θ - α * m̂_t / (√v̂_t + ε)
        Vector<T> vHatSqrt = ElementwiseSqrt(vHat);
        T[] epsilonArray = new T[parameters.Length];
        for (int i = 0; i < parameters.Length; i++)
        {
            epsilonArray[i] = MAMLConfig.AdamEpsilon;
        }
        Vector<T> epsilonVector = new Vector<T>(epsilonArray);
        Vector<T> denominator = vHatSqrt.Add(epsilonVector);
        Vector<T> adamUpdate = ElementwiseDivide(mHat, denominator);
        Vector<T> scaledUpdate = adamUpdate.Multiply(Configuration.MetaLearningRate);

        return parameters.Subtract(scaledUpdate);
    }

    /// <summary>
    /// Clips gradient by norm for training stability.
    /// </summary>
    private Vector<T> ClipGradientByNorm(Vector<T> gradient, T maxNorm)
    {
        T gradientNorm = gradient.Norm();
        T maxNormValue = maxNorm;

        if (NumOps.GreaterThan(gradientNorm, maxNormValue))
        {
            // Scale gradient: g_clipped = g * (max_norm / ||g||)
            T scale = NumOps.Divide(maxNormValue, gradientNorm);
            return gradient.Multiply(scale);
        }

        return gradient;
    }

    /// <summary>
    /// Element-wise multiplication of two vectors.
    /// </summary>
    private Vector<T> ElementwiseMultiply(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have the same length");

        var result = new T[a.Length];
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = NumOps.Multiply(a[i], b[i]);
        }
        return new Vector<T>(result);
    }

    /// <summary>
    /// Element-wise division of two vectors.
    /// </summary>
    private Vector<T> ElementwiseDivide(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have the same length");

        var result = new T[a.Length];
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = NumOps.Divide(a[i], b[i]);
        }
        return new Vector<T>(result);
    }

    /// <summary>
    /// Element-wise square root of a vector.
    /// </summary>
    private Vector<T> ElementwiseSqrt(Vector<T> v)
    {
        var result = new T[v.Length];
        for (int i = 0; i < v.Length; i++)
        {
            result[i] = NumOps.Sqrt(v[i]);
        }
        return new Vector<T>(result);
    }

    /// <inheritdoc/>
    public override MetaAdaptationResult<T> AdaptAndEvaluate(MetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
            throw new ArgumentNullException(nameof(task));

        var startTime = Stopwatch.StartNew();

        // Save original meta-parameters
        Vector<T> originalParameters = MetaModel.GetParameters();

        // Evaluate before adaptation (baseline)
        T initialQueryLoss = ComputeLoss(MetaModel, task.QuerySetX, task.QuerySetY);

        var perStepLosses = new List<T> { initialQueryLoss };

        // Inner loop: Adapt to task using support set
        if (_innerOptimizer != null && _supportsGradientComputation)
        {
            // Use provided inner optimizer with gradient computation
            var gradientModel = (IGradientComputable<T, TInput, TOutput>)MetaModel;

            for (int step = 0; step < Configuration.InnerSteps; step++)
            {
                // Compute gradients on support set
                var gradients = gradientModel.ComputeGradients(task.SupportSetX, task.SupportSetY, LossFunction);

                // Update parameters using inner optimizer
                var currentParams = MetaModel.GetParameters();
                var newParams = _innerOptimizer.UpdateParameters(currentParams, gradients);
                MetaModel.SetParameters(newParams);

                // Track loss after each step for convergence analysis
                T stepLoss = ComputeLoss(MetaModel, task.QuerySetX, task.QuerySetY);
                perStepLosses.Add(stepLoss);
            }
        }
        else
        {
            // Use model's built-in Train method (default behavior)
            for (int step = 0; step < Configuration.InnerSteps; step++)
            {
                MetaModel.Train(task.SupportSetX, task.SupportSetY);

                // Track loss after each step for convergence analysis
                T stepLoss = ComputeLoss(MetaModel, task.QuerySetX, task.QuerySetY);
                perStepLosses.Add(stepLoss);
            }
        }

        // Evaluate after adaptation
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
            ["support_query_accuracy_gap"] = NumOps.Subtract(supportAccuracy, queryAccuracy),
            ["uses_second_order"] = MAMLConfig.UseFirstOrderApproximation ? NumOps.FromDouble(0) : NumOps.FromDouble(1),
            ["gradient_computation_supported"] = _supportsGradientComputation ? NumOps.FromDouble(1) : NumOps.FromDouble(0)
        };

        // Return comprehensive adaptation results
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

    private bool _supportsGradientComputable => _supportsGradientComputation;
}
