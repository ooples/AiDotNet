using AiDotNet.Data.Structures;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.MetaLearning;

/// <summary>
/// Unified base class for all meta-learning algorithms, providing both training infrastructure
/// and shared algorithm utilities.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// This base class follows the TimeSeriesModelBase pattern where the base class does heavy lifting
/// and concrete algorithm implementations override with their custom logic. It provides:
/// </para>
/// <list type="bullet">
/// <item>Configuration management and validation</item>
/// <item>Loss and accuracy computation</item>
/// <item>Model saving/loading with serialization</item>
/// <item>Training loop orchestration</item>
/// <item>Evaluation on multiple tasks</item>
/// <item>Gradient computation, clipping, and application utilities</item>
/// <item>Built-in inner/outer optimizer support with Adam defaults</item>
/// </list>
/// <para><b>For Algorithm Implementers:</b>
/// To create a new meta-learning algorithm:
/// 1. Extend this base class
/// 2. Set AlgorithmType in constructor
/// 3. Override MetaTrainCore() with your algorithm's meta-update logic
/// 4. Override AdaptCore() with your task adaptation strategy
/// 5. All shared functionality (metrics, saving, evaluation) is handled automatically
/// </para>
/// </remarks>
public abstract class MetaLearnerBase<T, TInput, TOutput> : IMetaLearner<T, TInput, TOutput>, IConfigurableModel<T>
{
    #region Fields

    /// <summary>
    /// The model being meta-trained.
    /// </summary>
    protected IFullModel<T, TInput, TOutput> MetaModel;

    /// <summary>
    /// The loss function used to evaluate task performance.
    /// </summary>
    protected readonly ILossFunction<T> LossFunction;

    /// <summary>
    /// Configuration options for meta-learning.
    /// </summary>
    protected readonly IMetaLearnerOptions<T> _options;

    /// <summary>
    /// Episodic data loader for sampling meta-learning tasks.
    /// </summary>
    protected readonly IEpisodicDataLoader<T, TInput, TOutput>? DataLoader;

    /// <summary>
    /// Numeric operations for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Random number generator for stochastic operations.
    /// </summary>
    protected Random RandomGenerator;

    /// <summary>
    /// Optimizer for meta-parameter updates (outer loop).
    /// </summary>
    protected readonly IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer;

    /// <summary>
    /// Optimizer for task adaptation (inner loop).
    /// </summary>
    protected readonly IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer;

    /// <summary>
    /// Current meta-training iteration count.
    /// </summary>
    protected int _currentIteration;

    /// <summary>
    /// Tracks whether the expensive gradient fallback warning has been emitted.
    /// </summary>
    private bool _gradientFallbackWarningEmitted;

    #endregion

    #region Properties

    /// <summary>
    /// Gets the global execution engine for vectorized operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The execution engine provides CPU/GPU-accelerated vector and matrix operations.
    /// All vectorized computations should use this engine rather than manual loops
    /// for optimal performance and hardware acceleration.
    /// </para>
    /// <para><b>For Beginners:</b> This engine automatically chooses the best way to
    /// run math operations - on your CPU with SIMD optimization, or on your GPU
    /// for massive parallelism. You don't need to change your code to switch between them.
    /// </para>
    /// </remarks>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> BaseModel => MetaModel;

    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Options => _options;

    /// <inheritdoc/>
    public virtual ModelOptions GetOptions() => (ModelOptions)_options;

    /// <inheritdoc/>
    public int CurrentIteration => _currentIteration;

    /// <inheritdoc/>
    public abstract MetaLearningAlgorithmType AlgorithmType { get; }

    /// <inheritdoc/>
    public int AdaptationSteps => _options.AdaptationSteps;

    /// <inheritdoc/>
    public double InnerLearningRate => _options.InnerLearningRate;

    /// <inheritdoc/>
    public double OuterLearningRate => _options.OuterLearningRate;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the MetaLearnerBase class.
    /// </summary>
    /// <param name="metaModel">The model to meta-train.</param>
    /// <param name="lossFunction">Loss function for evaluation.</param>
    /// <param name="options">Configuration options with all hyperparameters.</param>
    /// <param name="dataLoader">Optional episodic data loader for sampling tasks.</param>
    /// <param name="metaOptimizer">Optional optimizer for meta-updates. If null, gradient updates use manual SGD with OuterLearningRate.</param>
    /// <param name="innerOptimizer">Optional optimizer for inner-loop. If null, gradient updates use manual SGD with InnerLearningRate.</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel, lossFunction, or options is null.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration validation fails.</exception>
    protected MetaLearnerBase(
        IFullModel<T, TInput, TOutput> metaModel,
        ILossFunction<T> lossFunction,
        IMetaLearnerOptions<T> options,
        IEpisodicDataLoader<T, TInput, TOutput>? dataLoader = null,
        IGradientBasedOptimizer<T, TInput, TOutput>? metaOptimizer = null,
        IGradientBasedOptimizer<T, TInput, TOutput>? innerOptimizer = null)
    {
        MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel));
        LossFunction = lossFunction ?? throw new ArgumentNullException(nameof(lossFunction));
        _options = options ?? throw new ArgumentNullException(nameof(options));

        if (!options.IsValid())
        {
            throw new ArgumentException("Invalid meta-learner options configuration.", nameof(options));
        }

        DataLoader = dataLoader;
        MetaOptimizer = metaOptimizer;
        InnerOptimizer = innerOptimizer;

        // Initialize random generator
        RandomGenerator = options.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(options.RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();
    }


    #endregion

    #region IMetaLearner Implementation

    /// <inheritdoc/>
    public abstract T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch);

    /// <inheritdoc/>
    public abstract IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task);

    /// <inheritdoc/>
    public virtual T Evaluate(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null)
        {
            throw new ArgumentNullException(nameof(taskBatch));
        }

        if (taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be empty.", nameof(taskBatch));
        }

        T totalLoss = NumOps.Zero;
        int taskCount = 0;

        foreach (var task in taskBatch.Tasks)
        {
            // Adapt to the task using support set
            var adaptedModel = Adapt(task);

            // Evaluate on query set
            var queryPredictions = adaptedModel.Predict(task.QueryInput);
            var queryLoss = ComputeLossFromOutput(queryPredictions, task.QueryOutput);

            totalLoss = NumOps.Add(totalLoss, queryLoss);
            taskCount++;
        }

        // Return average loss
        return taskCount > 0
            ? NumOps.Divide(totalLoss, NumOps.FromDouble(taskCount))
            : NumOps.Zero;
    }

    /// <inheritdoc/>
    public virtual MetaTrainingStepResult<T> MetaTrainStep(int batchSize)
    {
        if (batchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize), batchSize, "Batch size must be a positive integer.");
        }

        if (DataLoader == null)
        {
            throw new InvalidOperationException("Cannot perform MetaTrainStep without an episodic data loader.");
        }

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        // Sample task batch
        var tasks = DataLoader.GetTaskBatch(batchSize);
        var taskBatch = CreateTaskBatch(tasks);

        // Perform meta-training
        var metaLoss = MetaTrain(taskBatch);
        _currentIteration++;

        stopwatch.Stop();

        // MetaTrainingStepResult constructor: metaLoss, taskLoss, accuracy, numTasks, iteration, timeMs
        return new MetaTrainingStepResult<T>(
            metaLoss: metaLoss,
            taskLoss: metaLoss, // Use meta-loss as task loss for simplicity
            accuracy: NumOps.Zero, // Accuracy computed separately if needed
            numTasks: batchSize,
            iteration: _currentIteration,
            timeMs: stopwatch.Elapsed.TotalMilliseconds
        );
    }

    /// <inheritdoc/>
    public virtual MetaTrainingResult<T> Train()
    {
        if (DataLoader == null)
        {
            throw new InvalidOperationException("Cannot train without an episodic data loader.");
        }

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var lossHistory = new List<T>();
        var accuracyHistory = new List<T>();

        for (int iter = 0; iter < _options.NumMetaIterations; iter++)
        {
            var stepResult = MetaTrainStep(_options.MetaBatchSize);
            lossHistory.Add(stepResult.MetaLoss);
            accuracyHistory.Add(stepResult.Accuracy);

            // Checkpointing
            if (_options.EnableCheckpointing && _options.CheckpointFrequency > 0 &&
                iter > 0 && iter % _options.CheckpointFrequency == 0)
            {
                // Save checkpoint to current directory (override Save method for custom paths)
                Save($"checkpoint_{iter}.bin");
            }
        }

        stopwatch.Stop();

        // MetaTrainingResult constructor: lossHistory, accuracyHistory, trainingTime
        return new MetaTrainingResult<T>(
            lossHistory: new Vector<T>(lossHistory.ToArray()),
            accuracyHistory: new Vector<T>(accuracyHistory.ToArray()),
            trainingTime: stopwatch.Elapsed
        );
    }

    /// <inheritdoc/>
    public virtual MetaEvaluationResult<T> Evaluate(int numTasks)
    {
        if (numTasks <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numTasks), numTasks, "Number of tasks must be a positive integer.");
        }

        if (DataLoader == null)
        {
            throw new InvalidOperationException("Cannot evaluate without an episodic data loader.");
        }

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var losses = new List<T>();
        var accuracies = new List<T>();

        for (int i = 0; i < numTasks; i++)
        {
            var task = DataLoader.GetNextTask();

            // Adapt to task
            var adaptedModel = Adapt(ToMetaLearningTask(task));

            // Evaluate on query set
            var predictions = adaptedModel.Predict(task.QuerySetX);
            var loss = ComputeLossFromOutput(predictions, task.QuerySetY);
            losses.Add(loss);

            // Compute accuracy if applicable
            var accuracy = ComputeAccuracy(predictions, task.QuerySetY);
            accuracies.Add(NumOps.FromDouble(accuracy));
        }

        stopwatch.Stop();

        // MetaEvaluationResult constructor: taskAccuracies, taskLosses, evaluationTime
        return new MetaEvaluationResult<T>(
            taskAccuracies: new Vector<T>(accuracies.ToArray()),
            taskLosses: new Vector<T>(losses.ToArray()),
            evaluationTime: stopwatch.Elapsed
        );
    }

    /// <inheritdoc/>
    public virtual MetaAdaptationResult<T> AdaptAndEvaluate(MetaLearningTask<T, TInput, TOutput> task)
    {
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        // Convert to IMetaLearningTask and adapt
        var metaTask = ToMetaLearningTask(task);
        var adaptedModel = Adapt(metaTask);

        // Evaluate on query set
        var queryPredictions = adaptedModel.Predict(task.QuerySetX);
        var queryLoss = ComputeLossFromOutput(queryPredictions, task.QuerySetY);
        var queryAccuracy = ComputeAccuracy(queryPredictions, task.QuerySetY);

        // Evaluate on support set
        var supportPredictions = adaptedModel.Predict(task.SupportSetX);
        var supportLoss = ComputeLossFromOutput(supportPredictions, task.SupportSetY);
        var supportAccuracy = ComputeAccuracy(supportPredictions, task.SupportSetY);

        stopwatch.Stop();

        // MetaAdaptationResult constructor: queryAccuracy, queryLoss, supportAccuracy, supportLoss, adaptationSteps, adaptationTimeMs
        return new MetaAdaptationResult<T>(
            queryAccuracy: NumOps.FromDouble(queryAccuracy),
            queryLoss: queryLoss,
            supportAccuracy: NumOps.FromDouble(supportAccuracy),
            supportLoss: supportLoss,
            adaptationSteps: _options.AdaptationSteps,
            adaptationTimeMs: stopwatch.Elapsed.TotalMilliseconds
        );
    }

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> GetMetaModel() => MetaModel;

    /// <inheritdoc/>
    public void SetMetaModel(IFullModel<T, TInput, TOutput> model)
    {
        MetaModel = model ?? throw new ArgumentNullException(nameof(model));

        // Reset optimizer state when the model changes to avoid stale momentum/velocity vectors
        // from the previous model's parameter space
        if (MetaOptimizer is IResettable resettableMetaOptimizer)
        {
            resettableMetaOptimizer.Reset();
        }

        if (InnerOptimizer is IResettable resettableInnerOptimizer)
        {
            resettableInnerOptimizer.Reset();
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// This method saves the meta-model parameters. Note that optimizer state (momentum, velocity)
    /// is not persisted and will be reset when loading. For production use, consider implementing
    /// a custom serialization scheme that saves:
    /// </para>
    /// <list type="bullet">
    /// <item>Model parameters (saved here)</item>
    /// <item>Optimizer state (momentum, velocity vectors)</item>
    /// <item>Current iteration count</item>
    /// <item>Learning rate schedules</item>
    /// </list>
    /// </remarks>
    public virtual void Save(string filePath)
    {
        // Basic serialization - saves model parameters
        // Note: Optimizer state is not persisted (will be reset on load)
        var serializer = MetaModel as IModelSerializer;
        if (serializer != null)
        {
            serializer.SaveModel(filePath);

            // Save iteration count in a companion file for training resumption
            var metaPath = filePath + ".meta";
            try
            {
                System.IO.File.WriteAllText(metaPath, _currentIteration.ToString());
            }
            catch
            {
                // Non-critical: continue even if metadata save fails
            }
        }
        else
        {
            throw new NotSupportedException("The meta-model does not support serialization.");
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// This method loads the meta-model parameters. Note that optimizer state is not restored
    /// and will be reset after loading. Call <see cref="Reset"/> after loading if you want
    /// to ensure a clean training state.
    /// </para>
    /// </remarks>
    public virtual void Load(string filePath)
    {
        var serializer = MetaModel as IModelSerializer;
        if (serializer != null)
        {
            serializer.LoadModel(filePath);

            // Try to restore iteration count from companion file
            var metaPath = filePath + ".meta";
            if (System.IO.File.Exists(metaPath))
            {
                try
                {
                    var content = System.IO.File.ReadAllText(metaPath);
                    if (int.TryParse(content, out int iteration))
                    {
                        _currentIteration = iteration;
                    }
                }
                catch
                {
                    // Non-critical: continue even if metadata load fails
                }
            }

            // Reset optimizer state since it's not persisted
            if (MetaOptimizer is IResettable resettableMetaOptimizer)
            {
                resettableMetaOptimizer.Reset();
            }

            if (InnerOptimizer is IResettable resettableInnerOptimizer)
            {
                resettableInnerOptimizer.Reset();
            }
        }
        else
        {
            throw new NotSupportedException("The meta-model does not support serialization.");
        }
    }

    /// <inheritdoc/>
    public virtual void Reset()
    {
        _currentIteration = 0;
        RandomGenerator = _options.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    #endregion

    #region Gradient Utilities

    /// <summary>
    /// Computes gradients of the loss with respect to model parameters using the model's
    /// built-in gradient computation via <see cref="IGradientComputable{T, TInput, TOutput}"/>.
    /// </summary>
    /// <param name="model">The model to compute gradients for.</param>
    /// <param name="input">Input data.</param>
    /// <param name="expectedOutput">Expected output.</param>
    /// <returns>Gradient vector with respect to all model parameters.</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the model does not implement <see cref="IGradientComputable{T, TInput, TOutput}"/>
    /// and no fallback computation is possible.
    /// </exception>
    /// <remarks>
    /// <para>
    /// This method leverages the model's own gradient computation capabilities, which typically use:
    /// - Automatic differentiation (backpropagation for neural networks)
    /// - GPU acceleration via the Engine system when available
    /// - Optimized JIT-compiled gradient kernels
    /// </para>
    /// <para>
    /// <b>Production Implementation:</b> Models that implement <see cref="IGradientComputable{T, TInput, TOutput}"/>
    /// provide their own efficient gradient computation. This is the preferred path for production use.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Gradients tell us how to adjust each parameter to reduce the loss.
    /// This method asks the model to compute those gradients using its internal backpropagation
    /// mechanism, which is much more accurate and efficient than numerical approximations.
    /// </para>
    /// </remarks>
    protected virtual Vector<T> ComputeGradients(
        IFullModel<T, TInput, TOutput> model,
        TInput input,
        TOutput expectedOutput)
    {
        // PRODUCTION PATH: Use model's built-in gradient computation (IGradientComputable)
        // This is the preferred method as it uses proper backpropagation with GPU acceleration
        if (model is IGradientComputable<T, TInput, TOutput> gradientModel)
        {
            return gradientModel.ComputeGradients(input, expectedOutput, LossFunction);
        }

        // FALLBACK PATH: Manual gradient computation using loss function derivative
        // This should only be used for simple models that don't implement IGradientComputable
        // WARNING: This is less efficient and may not work correctly for all model architectures
        return ComputeGradientsFallback(model, input, expectedOutput);
    }

    /// <summary>
    /// Fallback gradient computation for models that don't implement <see cref="IGradientComputable{T, TInput, TOutput}"/>.
    /// Uses numerical differentiation or loss derivative propagation.
    /// </summary>
    /// <param name="model">The model to compute gradients for.</param>
    /// <param name="input">Input data.</param>
    /// <param name="expectedOutput">Expected output.</param>
    /// <returns>Gradient vector (may be less accurate than model's native gradient computation).</returns>
    /// <remarks>
    /// <para>
    /// This fallback method uses finite differences to approximate gradients when the model
    /// doesn't support automatic differentiation. This is computationally expensive O(n) where
    /// n is the number of parameters, compared to O(1) backpropagation.
    /// </para>
    /// <para>
    /// <b>WARNING:</b> For production use, ensure your model implements
    /// <see cref="IGradientComputable{T, TInput, TOutput}"/> for efficient and accurate gradients.
    /// </para>
    /// </remarks>
    private Vector<T> ComputeGradientsFallback(
        IFullModel<T, TInput, TOutput> model,
        TInput input,
        TOutput expectedOutput)
    {
        // Emit a warning once about the expensive fallback being used
        if (!_gradientFallbackWarningEmitted)
        {
            _gradientFallbackWarningEmitted = true;
            System.Diagnostics.Debug.WriteLine(
                $"[MetaLearning Warning] Model {model.GetType().Name} does not implement IGradientComputable<T, TInput, TOutput>. " +
                $"Using expensive O(n) finite difference gradient approximation. " +
                $"For production use, implement IGradientComputable for efficient backpropagation.");
        }

        var parameters = model.GetParameters();
        var gradients = new Vector<T>(parameters.Length);

        // Compute baseline loss
        var predictions = model.Predict(input);
        var predVector = ConvertToVector(predictions);
        var expectedVector = ConvertToVector(expectedOutput);

        if (predVector == null || expectedVector == null)
        {
            throw new InvalidOperationException(
                $"Cannot compute gradients: unable to convert predictions or expected output to Vector<T>. " +
                $"Prediction type: {typeof(TOutput).Name}. Ensure the model implements IGradientComputable<T, TInput, TOutput> " +
                $"for proper gradient computation.");
        }

        T baseLoss = LossFunction.CalculateLoss(predVector, expectedVector);

        // Use finite differences for gradient approximation
        // This is expensive O(n) but works for any differentiable model
        T epsilon = NumOps.FromDouble(1e-7);

        for (int i = 0; i < parameters.Length; i++)
        {
            // Perturb parameter positively
            var perturbedParams = new Vector<T>(parameters.Length);
            for (int j = 0; j < parameters.Length; j++)
            {
                perturbedParams[j] = j == i
                    ? NumOps.Add(parameters[j], epsilon)
                    : parameters[j];
            }

            // Compute perturbed loss
            model.SetParameters(perturbedParams);
            var perturbedPredictions = model.Predict(input);
            var perturbedVector = ConvertToVector(perturbedPredictions);

            if (perturbedVector == null)
            {
                gradients[i] = NumOps.Zero;
                continue;
            }

            T perturbedLoss = LossFunction.CalculateLoss(perturbedVector, expectedVector);

            // Compute gradient using finite difference: (f(x+h) - f(x)) / h
            gradients[i] = NumOps.Divide(
                NumOps.Subtract(perturbedLoss, baseLoss),
                epsilon);
        }

        // Restore original parameters
        model.SetParameters(parameters);

        return gradients;
    }

    /// <summary>
    /// Computes second-order gradients for full MAML by backpropagating through the adaptation process.
    /// </summary>
    /// <param name="model">The model to compute second-order gradients for.</param>
    /// <param name="adaptationSteps">Sequence of (input, target) pairs used during inner loop adaptation.</param>
    /// <param name="queryInput">Query input for meta-gradient computation.</param>
    /// <param name="queryTarget">Query target for meta-gradient computation.</param>
    /// <param name="innerLearningRate">Learning rate used during inner loop adaptation.</param>
    /// <returns>Meta-gradients that account for how changing initial parameters affects learning trajectory.</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the model doesn't implement <see cref="ISecondOrderGradientComputable{T, TInput, TOutput}"/>
    /// and second-order gradients are required.
    /// </exception>
    /// <remarks>
    /// <para>
    /// Second-order gradients are required for full MAML (Model-Agnostic Meta-Learning).
    /// They capture how changes to initial parameters affect the entire adaptation trajectory,
    /// not just the final adapted parameters.
    /// </para>
    /// <para>
    /// <b>Computational Cost:</b> O(K * P^2) where K is adaptation steps and P is parameter count.
    /// For large models, consider using first-order MAML (FOMAML) approximation instead.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Normal gradients ask "how should I change parameters to improve?"
    /// Second-order gradients ask "how should I change my starting point so that learning
    /// itself becomes more effective?"
    /// </para>
    /// </remarks>
    protected virtual Vector<T> ComputeSecondOrderGradients(
        IFullModel<T, TInput, TOutput> model,
        List<(TInput input, TOutput target)> adaptationSteps,
        TInput queryInput,
        TOutput queryTarget,
        T innerLearningRate)
    {
        // Use model's built-in second-order gradient computation if available
        if (model is ISecondOrderGradientComputable<T, TInput, TOutput> secondOrderModel)
        {
            return secondOrderModel.ComputeSecondOrderGradients(
                adaptationSteps,
                queryInput,
                queryTarget,
                LossFunction,
                innerLearningRate);
        }

        // Fallback: First-order approximation (FOMAML)
        // This ignores the gradient through the adaptation process but is much cheaper
        // and performs nearly as well in practice (Finn et al., 2017)

        // First, perform adaptation
        var parameters = model.GetParameters();
        foreach (var (input, target) in adaptationSteps)
        {
            var gradients = ComputeGradients(model, input, target);
            parameters = ApplyGradients(parameters, gradients, NumOps.ToDouble(innerLearningRate));
            model.SetParameters(parameters);
        }

        // Compute gradients on query set (first-order approximation)
        return ComputeGradients(model, queryInput, queryTarget);
    }

    /// <summary>
    /// Applies gradients to update model parameters using simple SGD.
    /// </summary>
    /// <param name="parameters">Current parameters.</param>
    /// <param name="gradients">Gradients to apply.</param>
    /// <param name="learningRate">Learning rate.</param>
    /// <returns>Updated parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method performs vanilla SGD: θ = θ - lr * gradients.
    /// It does NOT use the stored <see cref="MetaOptimizer"/> or <see cref="InnerOptimizer"/>
    /// because those optimizers expect to work with the full model, not just parameter vectors.
    /// </para>
    /// <para>
    /// <b>For Advanced Optimizers:</b> If you need momentum, Adam, or other adaptive methods,
    /// override this method in your algorithm implementation or use the stored optimizers directly
    /// in your <see cref="MetaTrain"/> and <see cref="Adapt"/> implementations.
    /// </para>
    /// </remarks>
    protected virtual Vector<T> ApplyGradients(
        Vector<T> parameters,
        Vector<T> gradients,
        double learningRate)
    {
        // Use IEngine for GPU/CPU-accelerated vectorized operations
        // Vanilla SGD: θ = θ - lr * gradients
        var lr = NumOps.FromDouble(learningRate);
        var scaledGradients = Engine.Multiply(gradients, lr);
        return Engine.Subtract(parameters, scaledGradients);
    }

    /// <summary>
    /// Clips gradients to prevent exploding gradients.
    /// </summary>
    /// <param name="gradients">Gradients to clip.</param>
    /// <param name="threshold">Maximum gradient norm. If null, uses options value.</param>
    /// <returns>Clipped gradients.</returns>
    protected virtual Vector<T> ClipGradients(Vector<T> gradients, double? threshold = null)
    {
        var clipThreshold = threshold ?? _options.GradientClipThreshold;

        if (!clipThreshold.HasValue || clipThreshold.Value <= 0)
        {
            return gradients;
        }

        // Compute L2 norm using IEngine for vectorized element-wise multiplication
        // gradients² = gradients * gradients (element-wise)
        var squaredGradients = Engine.Multiply(gradients, gradients);

        // Sum all squared elements to get ||gradients||²
        T sumSquared = NumOps.Zero;
        for (int i = 0; i < squaredGradients.Length; i++)
        {
            sumSquared = NumOps.Add(sumSquared, squaredGradients[i]);
        }

        double norm = Math.Sqrt(NumOps.ToDouble(sumSquared));

        if (norm <= clipThreshold.Value)
        {
            return gradients;
        }

        // Scale gradients using IEngine for vectorized multiplication
        // clipped = gradients * (threshold / norm)
        var scale = NumOps.FromDouble(clipThreshold.Value / norm);
        return Engine.Multiply(gradients, scale);
    }

    /// <summary>
    /// Updates auxiliary parameters using multi-sample SPSA (Simultaneous Perturbation Stochastic
    /// Approximation) with a caller-provided loss function. The loss delegate must use the
    /// auxiliary parameters (which are perturbed in-place via the <paramref name="auxParams"/> ref)
    /// so that the finite-difference gradient estimate reflects their actual effect on the loss.
    /// </summary>
    /// <param name="taskBatch">The current task batch for loss evaluation.</param>
    /// <param name="auxParams">The auxiliary parameters to update (modified in place).</param>
    /// <param name="learningRate">Learning rate for the update step.</param>
    /// <param name="computeLoss">
    /// Delegate that computes the average loss over the task batch using the current state of
    /// <paramref name="auxParams"/>. The delegate is called with params already perturbed,
    /// so it should read the auxiliary param field directly (which is the same ref).
    /// </param>
    /// <param name="numSamples">Number of perturbation directions to average (default 3).</param>
    /// <param name="epsilon">Perturbation magnitude for finite differences (default 1e-5).</param>
    /// <remarks>
    /// <para>
    /// SPSA (Spall, 1992) estimates gradients using only 2 function evaluations per sample,
    /// regardless of the number of parameters. Multi-sample averaging reduces variance of the
    /// gradient estimate at the cost of additional evaluations (2 * numSamples total).
    /// </para>
    /// <para><b>For Beginners:</b> When we can't compute exact gradients for auxiliary parameters
    /// (because they're not part of the main neural network), we estimate them by:
    /// 1. Randomly perturbing all parameters simultaneously
    /// 2. Measuring how the loss changes via a delegate that actually USES the perturbed params
    /// 3. Inferring the gradient direction from the change
    /// Doing this multiple times and averaging gives a more reliable gradient estimate.
    /// </para>
    /// </remarks>
    protected void UpdateAuxiliaryParamsSPSA(
        TaskBatch<T, TInput, TOutput> taskBatch,
        ref Vector<T> auxParams,
        double learningRate,
        Func<TaskBatch<T, TInput, TOutput>, double> computeLoss,
        int numSamples = 3,
        double epsilon = 1e-5)
    {
        if (auxParams.Length == 0 || taskBatch.Tasks.Length == 0)
            return;

        // Compute base loss once (shared across all samples)
        // The delegate reads the current (unperturbed) aux params
        double baseLoss = computeLoss(taskBatch);

        // Accumulate gradient estimates across multiple perturbation directions
        var gradientAccum = new double[auxParams.Length];

        for (int s = 0; s < numSamples; s++)
        {
            // Generate random Rademacher direction (+1/-1 per component)
            var direction = new double[auxParams.Length];
            for (int i = 0; i < auxParams.Length; i++)
                direction[i] = RandomGenerator.NextDouble() > 0.5 ? 1.0 : -1.0;

            // Perturb parameters: auxParams += epsilon * direction
            for (int i = 0; i < auxParams.Length; i++)
                auxParams[i] = NumOps.Add(auxParams[i], NumOps.FromDouble(epsilon * direction[i]));

            // Compute perturbed loss — delegate reads the now-perturbed aux params
            double perturbedLoss = computeLoss(taskBatch);

            // Restore parameters: auxParams -= epsilon * direction
            for (int i = 0; i < auxParams.Length; i++)
                auxParams[i] = NumOps.Subtract(auxParams[i], NumOps.FromDouble(epsilon * direction[i]));

            // Accumulate per-component gradient estimate: g_i = (loss+ - loss0) / (epsilon * direction_i)
            double directionalGrad = (perturbedLoss - baseLoss) / epsilon;
            for (int i = 0; i < auxParams.Length; i++)
                gradientAccum[i] += directionalGrad * direction[i];
        }

        // Average gradient estimates and apply update
        double scale = learningRate / numSamples;
        for (int i = 0; i < auxParams.Length; i++)
            auxParams[i] = NumOps.Subtract(auxParams[i], NumOps.FromDouble(scale * gradientAccum[i]));
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Computes the element-wise average of a list of vectors.
    /// </summary>
    /// <param name="vectors">The vectors to average (must all be the same length).</param>
    /// <returns>The element-wise average vector, or an empty vector if the list is empty.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Given multiple gradient vectors (one per task), this computes
    /// the average gradient by summing all vectors element-wise and dividing by the count.
    /// This is used to aggregate gradients across a batch of meta-learning tasks.</para>
    /// </remarks>
    protected Vector<T> AverageVectors(List<Vector<T>> vectors)
    {
        if (vectors.Count == 0) return new Vector<T>(0);
        var result = new Vector<T>(vectors[0].Length);
        foreach (var v in vectors)
            for (int i = 0; i < result.Length; i++)
                result[i] = NumOps.Add(result[i], v[i]);
        var scale = NumOps.FromDouble(1.0 / vectors.Count);
        for (int i = 0; i < result.Length; i++)
            result[i] = NumOps.Multiply(result[i], scale);
        return result;
    }

    /// <summary>
    /// Creates a TaskBatch from a list of MetaLearningTasks.
    /// </summary>
    protected TaskBatch<T, TInput, TOutput> CreateTaskBatch(
        IReadOnlyList<MetaLearningTask<T, TInput, TOutput>> tasks)
    {
        var taskArray = tasks.Select(t => (IMetaLearningTask<T, TInput, TOutput>)new TaskWrapper<T, TInput, TOutput>(t)).ToArray();
        return new TaskBatch<T, TInput, TOutput>(taskArray);
    }

    /// <summary>
    /// Converts a MetaLearningTask to IMetaLearningTask.
    /// </summary>
    protected IMetaLearningTask<T, TInput, TOutput> ToMetaLearningTask(MetaLearningTask<T, TInput, TOutput> task)
    {
        // Use TaskWrapper for unified task adaptation (consolidates former duplicate wrappers)
        return new TaskWrapper<T, TInput, TOutput>(task);
    }

    /// <summary>
    /// Computes accuracy for classification tasks.
    /// </summary>
    /// <param name="predictions">Model predictions (logits or probabilities per class).</param>
    /// <param name="labels">Ground truth labels (one-hot encoded or class indices).</param>
    /// <returns>Accuracy as a value between 0.0 and 1.0, or 0.0 if computation is not possible.</returns>
    /// <remarks>
    /// <para>
    /// This method computes classification accuracy by comparing predicted class indices
    /// (argmax of predictions) against true class indices (argmax of labels for one-hot,
    /// or direct value for class indices).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Accuracy is simply the fraction of samples where the model's
    /// predicted class matches the true class. For example, 0.85 means 85% correct predictions.
    /// </para>
    /// </remarks>
    protected virtual double ComputeAccuracy(TOutput predictions, TOutput labels)
    {
        // Convert to vectors for comparison
        var predVector = ConvertToVector(predictions);
        var labelVector = ConvertToVector(labels);

        if (predVector == null || labelVector == null)
        {
            return 0.0;
        }

        // Handle different data layouts:
        // 1. Batch of logits: [batch_size * num_classes] vs [batch_size * num_classes] (one-hot)
        // 2. Single sample: [num_classes] vs [num_classes] (one-hot)
        // 3. Class indices: [batch_size] vs [batch_size]

        // If lengths match and are small (likely single sample classification)
        if (predVector.Length == labelVector.Length && predVector.Length > 0)
        {
            // Check if it's one-hot encoded (values are 0 or 1) or class indices
            bool isOneHot = true;
            for (int i = 0; i < labelVector.Length && isOneHot; i++)
            {
                double val = NumOps.ToDouble(labelVector[i]);
                if (val != 0.0 && val != 1.0)
                {
                    isOneHot = false;
                }
            }

            if (isOneHot && predVector.Length > 1)
            {
                // One-hot encoded: find argmax of both and compare
                int predClass = FindArgmax(predVector);
                int labelClass = FindArgmax(labelVector);
                return predClass == labelClass ? 1.0 : 0.0;
            }
            else
            {
                // Likely class indices or regression - compare directly
                int correct = 0;
                for (int i = 0; i < predVector.Length; i++)
                {
                    int predClass = (int)Math.Round(NumOps.ToDouble(predVector[i]));
                    int labelClass = (int)Math.Round(NumOps.ToDouble(labelVector[i]));
                    if (predClass == labelClass)
                    {
                        correct++;
                    }
                }
                return (double)correct / predVector.Length;
            }
        }

        // Cannot compute accuracy - different layouts or unsupported format
        return 0.0;
    }

    /// <summary>
    /// Finds the index of the maximum value in a vector (argmax).
    /// </summary>
    private int FindArgmax(Vector<T> vector)
    {
        if (vector.Length == 0)
        {
            return -1;
        }

        int maxIndex = 0;
        T maxValue = vector[0];

        for (int i = 1; i < vector.Length; i++)
        {
            if (NumOps.ToDouble(vector[i]) > NumOps.ToDouble(maxValue))
            {
                maxValue = vector[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    /// <summary>
    /// Computes loss from TOutput by converting to Vector&lt;T&gt; if needed.
    /// </summary>
    /// <param name="predictions">The predictions from the model.</param>
    /// <param name="expectedOutput">The expected output.</param>
    /// <returns>The computed loss value.</returns>
    protected virtual T ComputeLossFromOutput(TOutput predictions, TOutput expectedOutput)
    {
        // Try to convert TOutput to Vector<T> for loss computation
        Vector<T>? predictedVector = ConvertToVector(predictions);
        Vector<T>? expectedVector = ConvertToVector(expectedOutput);

        if (predictedVector != null && expectedVector != null)
        {
            return LossFunction.CalculateLoss(predictedVector, expectedVector);
        }

        // If conversion fails, return zero (subclasses should override for specific types)
        return NumOps.Zero;
    }

    /// <summary>
    /// Converts TOutput to Vector&lt;T&gt; if possible.
    /// </summary>
    /// <param name="output">The output to convert.</param>
    /// <returns>The converted vector, or null if conversion is not possible.</returns>
    protected virtual Vector<T>? ConvertToVector(TOutput output)
    {
        // Direct Vector<T> type
        if (output is Vector<T> vector)
        {
            return vector;
        }

        // Tensor<T> type - convert to vector
        if (output is Tensor<T> tensor)
        {
            return tensor.ToVector();
        }

        // Array type
        if (output is T[] array)
        {
            return new Vector<T>(array);
        }

        // Cannot convert
        return null;
    }

    /// <summary>
    /// Computes mean of a list of values.
    /// </summary>
    protected T ComputeMean(List<T> values)
    {
        if (values.Count == 0)
        {
            return NumOps.Zero;
        }

        T sum = NumOps.Zero;
        foreach (var value in values)
        {
            sum = NumOps.Add(sum, value);
        }

        return NumOps.Divide(sum, NumOps.FromDouble(values.Count));
    }

    /// <summary>
    /// Clones the meta-model for task-specific adaptation.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the model does not implement ICloneable. Meta-learning requires
    /// model cloning to prevent parameter corruption during parallel task adaptation.
    /// </exception>
    protected virtual IFullModel<T, TInput, TOutput> CloneModel()
    {
        // Try to clone if the model supports it
        if (MetaModel is ICloneable cloneable)
        {
            return (IFullModel<T, TInput, TOutput>)cloneable.Clone();
        }

        // If model doesn't implement ICloneable, throw to prevent silent parameter corruption
        throw new InvalidOperationException(
            $"Cannot clone model of type {MetaModel.GetType().Name}. " +
            $"Meta-learning algorithms require models that implement ICloneable " +
            $"to prevent parameter corruption during parallel task adaptation.");
    }

    #endregion
}

#region Helper Classes

/// <summary>
/// Wraps a MetaLearningTask as an IMetaLearningTask.
/// </summary>
internal class TaskWrapper<T, TInput, TOutput> : IMetaLearningTask<T, TInput, TOutput>
{
    private readonly MetaLearningTask<T, TInput, TOutput> _task;

    public TaskWrapper(MetaLearningTask<T, TInput, TOutput> task)
    {
        _task = task;
    }

    public TInput SupportInput => _task.SupportSetX;
    public TOutput SupportOutput => _task.SupportSetY;
    public TInput QueryInput => _task.QuerySetX;
    public TOutput QueryOutput => _task.QuerySetY;
    public int NumWays => _task.NumWays;
    public int NumShots => _task.NumShots;
    public int NumQueryPerClass => _task.NumQueryPerClass;
    public string? Name => _task.Name;
    public Dictionary<string, object>? Metadata => _task.Metadata;
    public int? TaskId { get; set; }

    // Alias properties for compatibility
    public TInput QuerySetX => QueryInput;
    public TOutput QuerySetY => QueryOutput;
    public TInput SupportSetX => SupportInput;
    public TOutput SupportSetY => SupportOutput;
}

#endregion
