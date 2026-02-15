using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Meta-SGD (Meta Stochastic Gradient Descent) algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// Meta-SGD learns per-parameter learning rates for meta-learning. Instead of
/// learning just initialization parameters like MAML, it learns the learning
/// rate, momentum, and direction for each parameter individually, which can be
/// seen as learning a custom optimizer for each parameter.
/// </para>
/// <para>
/// <b>For Beginners:</b> Meta-SGD learns how to update each parameter individually:
/// </para>
/// <para>
/// In regular training, you use one learning rate for all weights. But different
/// parts of a neural network benefit from different learning rates. Meta-SGD
/// figures this out automatically by learning:
/// - <b>α_i:</b> The optimal learning rate for parameter i
/// - <b>β_i:</b> The optimal momentum for parameter i (optional)
/// - <b>d_i:</b> The optimal update direction/sign for parameter i (optional)
/// </para>
/// <para>
/// <b>Algorithm - Meta-SGD:</b>
/// <code>
/// # Learn per-parameter optimizers
/// for each parameter θ_i:
///     learning_rate_i = learnable_parameter
///     momentum_i = learnable_parameter (optional)
///     direction_i = learnable_parameter (optional)
///
/// # Meta-training episode
/// for each task in task_batch:
///     # Inner loop: adapt to task
///     adapted_params = initial_params.copy()
///     for step = 1 to K_inner:
///         gradients = compute_gradients(adapted_params, support_set)
///         for i in range(num_params):
///             # Per-parameter update rule
///             adapted_params[i] = update_rule_i(
///                 adapted_params[i],
///                 gradients[i],
///                 learning_rate_i,
///                 momentum_i,
///                 direction_i
///             )
///
///     # Evaluate on query set
///     query_loss = evaluate(adapted_params, query_set)
///
///     # Meta-update: optimize per-parameter coefficients
///     meta_gradients = compute_meta_gradients(query_loss)
///     update_per_parameter_optimizers(meta_gradients)
/// </code>
/// </para>
/// <para>
/// <b>Key Insights:</b>
/// 1. <b>Per-Parameter Optimization:</b> Each parameter gets its own learned
///    optimizer configuration, allowing heterogeneous learning rates across layers.
/// 2. <b>First-Order Method:</b> No Hessian computation needed, much faster than
///    second-order MAML while maintaining strong performance.
/// 3. <b>Interpretable:</b> Learned per-parameter learning rates reveal which
///    parameters are most important for quick adaptation.
/// 4. <b>Flexible Update Rules:</b> Can combine with various base optimizers
///    (SGD, Adam, RMSprop) for different adaptation characteristics.
/// </para>
/// <para>
/// <b>Reference:</b> Li, Z., Zhou, F., Chen, F., &amp; Li, H. (2017).
/// Meta-SGD: Learning to Learn Quickly for Few-Shot Learning.
/// </para>
/// </remarks>
public class MetaSGDAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly MetaSGDOptions<T, TInput, TOutput> _metaSGDOptions;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _metaSGDOptions;
    private readonly PerParameterOptimizer<T, TInput, TOutput> _optimizer;

    /// <summary>
    /// Initializes a new instance of the MetaSGDAlgorithm class.
    /// </summary>
    /// <param name="options">Meta-SGD configuration options containing the model and all hyperparameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when required components are not set in options.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration validation fails.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a Meta-SGD model that learns per-parameter optimizers:
    /// </para>
    /// <para>
    /// <b>What Meta-SGD needs:</b>
    /// - <b>MetaModel:</b> Neural network to be meta-trained (required)
    /// - <b>UpdateRuleType:</b> Type of update rule to learn (SGD, Adam, etc.)
    /// - <b>LearnLearningRate:</b> Whether to learn per-parameter learning rates (default: true)
    /// - <b>LearnMomentum:</b> Whether to learn per-parameter momentum (default: false)
    /// - <b>LearnDirection:</b> Whether to learn update direction sign (default: true)
    /// </para>
    /// <para>
    /// <b>What makes it different from MAML:</b>
    /// - MAML: Same learning rate for all parameters
    /// - Meta-SGD: Different learning rate per parameter
    /// - Meta-SGD learns optimizers, MAML learns initialization
    /// - Meta-SGD is first-order (faster), MAML is second-order (more accurate)
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Create Meta-SGD with minimal configuration
    /// var options = new MetaSGDOptions&lt;double, Tensor, Tensor&gt;(myNeuralNetwork);
    /// var metaSGD = new MetaSGDAlgorithm&lt;double, Tensor, Tensor&gt;(options);
    ///
    /// // Create Meta-SGD with full per-parameter optimization
    /// var options = new MetaSGDOptions&lt;double, Tensor, Tensor&gt;(myNeuralNetwork)
    /// {
    ///     UpdateRuleType = MetaSGDUpdateRuleType.Adam,
    ///     LearnLearningRate = true,
    ///     LearnMomentum = true,
    ///     LearnDirection = true,
    ///     LearnAdamBetas = true
    /// };
    /// var metaSGD = new MetaSGDAlgorithm&lt;double, Tensor, Tensor&gt;(options);
    /// </code>
    /// </example>
    public MetaSGDAlgorithm(MetaSGDOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? options.MetaModel.DefaultLossFunction,
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _metaSGDOptions = options;

        // Validate configuration
        if (!_metaSGDOptions.IsValid())
        {
            throw new ArgumentException("Meta-SGD configuration is invalid. Check all parameters.", nameof(options));
        }

        // Initialize per-parameter optimizer with learned coefficients
        var numParams = MetaModel.GetParameters().Length;
        _optimizer = new PerParameterOptimizer<T, TInput, TOutput>(numParams, _metaSGDOptions);

        // Initialize optimizer with warm-start values if enabled
        if (_metaSGDOptions.UseWarmStart)
        {
            InitializeOptimizer();
        }
    }

    /// <summary>
    /// Gets the algorithm type identifier for this meta-learner.
    /// </summary>
    /// <value>Returns <see cref="MetaLearningAlgorithmType.MetaSGD"/>.</value>
    /// <remarks>
    /// <para>
    /// This property identifies the algorithm as Meta-SGD, a first-order meta-learning
    /// algorithm that learns per-parameter learning rates, momentum terms, and update
    /// directions for fast task adaptation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells the framework which meta-learning algorithm
    /// is being used. Meta-SGD is characterized by its per-parameter optimization
    /// approach, which is simpler and faster than MAML while achieving competitive
    /// performance on few-shot learning tasks.
    /// </para>
    /// </remarks>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MetaSGD;

    /// <summary>
    /// Performs one meta-training step using Meta-SGD's per-parameter optimization approach.
    /// </summary>
    /// <param name="taskBatch">A batch of tasks to meta-train on, each containing support and query sets.</param>
    /// <returns>The average loss across all tasks in the batch (evaluated on query sets).</returns>
    /// <exception cref="ArgumentException">Thrown when the task batch is null or empty.</exception>
    /// <remarks>
    /// <para>
    /// Meta-SGD meta-training optimizes per-parameter learning coefficients:
    /// </para>
    /// <para>
    /// <b>For each task:</b>
    /// 1. Clone the meta-model with current meta-parameters
    /// 2. Perform K gradient descent steps using learned per-parameter optimizers
    /// 3. Evaluate adapted model on query set
    /// </para>
    /// <para>
    /// <b>Meta-Update:</b>
    /// 1. Compute gradients of query loss w.r.t. per-parameter coefficients
    /// 2. Update learning rates: α_i = α_i - η × ∂L_query/∂α_i
    /// 3. Update momentum (if enabled): β_i = β_i - η × ∂L_query/∂β_i
    /// 4. Update direction (if enabled): d_i = d_i - η × ∂L_query/∂d_i
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Meta-SGD learns how fast each weight should change.
    /// After seeing many tasks, it discovers that some weights need big updates
    /// (high learning rate) while others need small updates (low learning rate).
    /// This makes adaptation to new tasks much faster and more effective.
    /// </para>
    /// <para>
    /// <b>Key Difference from MAML:</b> While MAML computes how initialization
    /// affects final loss (requires second-order gradients), Meta-SGD directly
    /// learns the optimal update magnitude for each parameter (first-order only).
    /// </para>
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        T totalLoss = NumOps.Zero;

        foreach (var task in taskBatch.Tasks)
        {
            // Train on this episode and get the query loss
            T episodeLoss = TrainEpisode(task);
            totalLoss = NumOps.Add(totalLoss, episodeLoss);
        }

        // Return average loss across all tasks
        return NumOps.Divide(totalLoss, NumOps.FromDouble(taskBatch.BatchSize));
    }

    /// <summary>
    /// Adapts the meta-learned model to a new task using the learned per-parameter optimizers.
    /// </summary>
    /// <param name="task">The new task containing support set examples for adaptation.</param>
    /// <returns>A new model instance that has been adapted to the given task using learned optimizers.</returns>
    /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
    /// <remarks>
    /// <para>
    /// Meta-SGD adaptation uses the learned per-parameter learning rates, momentum,
    /// and directions to perform highly optimized gradient descent on the support set.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> When adapting to a new task, Meta-SGD uses the learned
    /// per-parameter optimizers to update the model. Each weight gets updated at its
    /// own optimal rate, making adaptation much faster than using a single learning
    /// rate for all weights.
    /// </para>
    /// <para>
    /// <b>Adaptation Process:</b>
    /// <code>
    /// for each adaptation step:
    ///     gradients = compute_gradients(model, support_set)
    ///     for each parameter i:
    ///         update_i = α_i × d_i × gradients[i] + β_i × velocity[i]
    ///         params[i] = params[i] - update_i
    /// </code>
    /// Where α_i, d_i, β_i are the learned per-parameter coefficients.
    /// </para>
    /// <para>
    /// <b>Advantages over MAML adaptation:</b>
    /// - Uses optimized per-parameter learning rates (not one rate for all)
    /// - Can include learned momentum for faster convergence
    /// - Direction coefficients can flip/scale gradients as needed
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        // Clone the meta model for task-specific adaptation
        var adaptedModel = CloneModel();

        // Create a clone of the per-parameter optimizer for this adaptation
        var taskOptimizer = _optimizer.Clone();

        // Perform adaptation using learned per-parameter optimizer
        AdaptWithLearnedOptimizer(task, adaptedModel, taskOptimizer);

        // Return the adapted model wrapped for the interface
        return new MetaSGDAdaptedModel<T, TInput, TOutput>(adaptedModel, taskOptimizer, _metaSGDOptions);
    }

    /// <summary>
    /// Trains the model and per-parameter optimizer on a single episode.
    /// </summary>
    /// <param name="task">The task for this episode.</param>
    /// <returns>The query loss after adaptation.</returns>
    /// <remarks>
    /// <para>
    /// This method performs the complete Meta-SGD training procedure for one task:
    /// 1. Save initial parameters
    /// 2. Adapt using per-parameter optimizer on support set
    /// 3. Evaluate on query set
    /// 4. Update per-parameter coefficients based on query loss
    /// </para>
    /// </remarks>
    private T TrainEpisode(IMetaLearningTask<T, TInput, TOutput> task)
    {
        // Get initial parameters
        var initialParams = MetaModel.GetParameters();
        var currentParams = new Vector<T>(initialParams.Length);
        for (int i = 0; i < initialParams.Length; i++)
        {
            currentParams[i] = initialParams[i];
        }

        // Clone model for this episode
        var episodeModel = CloneModel();
        episodeModel.SetParameters(currentParams);

        // Inner loop adaptation with per-parameter updates
        for (int step = 0; step < _metaSGDOptions.InnerSteps; step++)
        {
            // Compute gradients on support set
            var gradients = ComputeGradients(episodeModel, task.SupportInput, task.SupportOutput);

            // Clip gradients if threshold is set
            if (_metaSGDOptions.GradientClipThreshold.HasValue)
            {
                gradients = ClipGradients(gradients, _metaSGDOptions.GradientClipThreshold);
            }

            // Update each parameter with its learned optimizer
            for (int i = 0; i < currentParams.Length; i++)
            {
                currentParams[i] = _optimizer.UpdateParameter(i, currentParams[i], gradients[i]);
            }

            // Update model parameters
            episodeModel.SetParameters(currentParams);
        }

        // Evaluate on query set
        var queryPredictions = episodeModel.Predict(task.QueryInput);
        T queryLoss = ComputeLossFromOutput(queryPredictions, task.QueryOutput);

        // Compute meta-gradients for per-parameter optimizer update
        var metaGradients = ComputeMetaGradients(
            initialParams,
            currentParams,
            task.SupportInput,
            task.SupportOutput,
            task.QueryInput,
            task.QueryOutput,
            queryLoss);

        // Update per-parameter optimizer coefficients
        _optimizer.UpdateMetaParameters(metaGradients);

        return queryLoss;
    }

    /// <summary>
    /// Performs adaptation using the learned per-parameter optimizer.
    /// </summary>
    /// <param name="task">The task to adapt to.</param>
    /// <param name="model">The model to adapt.</param>
    /// <param name="optimizer">The per-parameter optimizer to use.</param>
    /// <remarks>
    /// <para>
    /// Uses the learned per-parameter learning rates, momentum, and directions
    /// to perform optimized gradient descent on the task's support set.
    /// </para>
    /// </remarks>
    private void AdaptWithLearnedOptimizer(
        IMetaLearningTask<T, TInput, TOutput> task,
        IFullModel<T, TInput, TOutput> model,
        PerParameterOptimizer<T, TInput, TOutput> optimizer)
    {
        var currentParams = model.GetParameters();

        // Inner loop adaptation with learned per-parameter optimizer
        for (int step = 0; step < _metaSGDOptions.AdaptationSteps; step++)
        {
            // Compute gradients on support set
            var gradients = ComputeGradients(model, task.SupportInput, task.SupportOutput);

            // Clip gradients if threshold is set
            if (_metaSGDOptions.GradientClipThreshold.HasValue)
            {
                gradients = ClipGradients(gradients, _metaSGDOptions.GradientClipThreshold);
            }

            // Update each parameter with its learned optimizer configuration
            for (int i = 0; i < currentParams.Length; i++)
            {
                currentParams[i] = optimizer.UpdateParameter(i, currentParams[i], gradients[i]);
            }

            // Update model parameters
            model.SetParameters(currentParams);
        }
    }

    /// <summary>
    /// Computes meta-gradients for updating the per-parameter optimizer coefficients.
    /// </summary>
    /// <param name="initialParams">Initial parameters before adaptation.</param>
    /// <param name="finalParams">Final parameters after adaptation.</param>
    /// <param name="supportInputs">Support set inputs.</param>
    /// <param name="supportOutputs">Support set outputs.</param>
    /// <param name="queryInputs">Query set inputs.</param>
    /// <param name="queryOutputs">Query set outputs.</param>
    /// <param name="queryLoss">Loss on query set.</param>
    /// <returns>Meta-gradients for per-parameter optimizer coefficients.</returns>
    /// <remarks>
    /// <para>
    /// This computes gradients with respect to the per-parameter optimizer coefficients
    /// (learning rates, momentum, direction). The computation uses finite differences
    /// for efficiency, as the exact gradient would require differentiating through
    /// the entire adaptation process.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells us how to adjust each per-parameter learning
    /// rate so that the model performs better on new examples after adaptation.
    /// </para>
    /// </remarks>
    private Vector<T> ComputeMetaGradients(
        Vector<T> initialParams,
        Vector<T> finalParams,
        TInput supportInputs,
        TOutput supportOutputs,
        TInput queryInputs,
        TOutput queryOutputs,
        T queryLoss)
    {
        // Get the number of meta-parameters (per-parameter coefficients)
        var numMetaParams = _optimizer.GetMetaParameterCount();
        var metaGradients = new Vector<T>(numMetaParams);

        // Use finite differences to approximate meta-gradients
        // This is more stable than backpropagating through the entire adaptation
        double epsilon = 1e-5;
        int paramIndex = 0;

        // Compute gradients for learning rates
        if (_metaSGDOptions.LearnLearningRate)
        {
            for (int i = 0; i < initialParams.Length; i++)
            {
                // Perturb learning rate and recompute loss
                var perturbedLoss = ComputePerturbedLoss(
                    i, epsilon, initialParams, supportInputs, supportOutputs, queryInputs, queryOutputs);

                // Finite difference gradient
                double grad = (NumOps.ToDouble(perturbedLoss) - NumOps.ToDouble(queryLoss)) / epsilon;
                metaGradients[paramIndex++] = NumOps.FromDouble(grad);
            }
        }

        // Compute gradients for momentum (if learning momentum)
        // Note: Momentum gradients require expensive second-order computation through the entire
        // adaptation trajectory. We use learning rate gradients as a proxy, which captures the
        // sensitivity of loss to parameter update magnitude.
        if (_metaSGDOptions.LearnMomentum)
        {
            for (int i = 0; i < initialParams.Length; i++)
            {
                // Use learning rate gradient as proxy (momentum affects effective step size)
                int lrIndex = i;
                if (lrIndex < metaGradients.Length && _metaSGDOptions.LearnLearningRate)
                {
                    // Scale the learning rate gradient to approximate momentum gradient
                    metaGradients[paramIndex++] = NumOps.Multiply(
                        metaGradients[lrIndex],
                        NumOps.FromDouble(0.1));
                }
                else
                {
                    metaGradients[paramIndex++] = NumOps.Zero;
                }
            }
        }

        // Compute gradients for direction (if learning direction)
        // Direction gradients follow similar proxy reasoning
        if (_metaSGDOptions.LearnDirection)
        {
            for (int i = 0; i < initialParams.Length; i++)
            {
                int lrIndex = i;
                if (lrIndex < metaGradients.Length && _metaSGDOptions.LearnLearningRate)
                {
                    // Direction affects gradient sign - use sign of LR gradient
                    double lrGrad = NumOps.ToDouble(metaGradients[lrIndex]);
                    metaGradients[paramIndex++] = NumOps.FromDouble(Math.Sign(lrGrad) * 0.01);
                }
                else
                {
                    metaGradients[paramIndex++] = NumOps.Zero;
                }
            }
        }

        // Compute gradients for Adam betas (if using Adam and learning betas)
        // Adam hyperparameters affect convergence dynamics - use LR gradients as proxy
        if (_metaSGDOptions.UpdateRuleType == MetaSGDUpdateRuleType.Adam && _metaSGDOptions.LearnAdamBetas)
        {
            for (int i = 0; i < initialParams.Length; i++)
            {
                int lrIndex = i;
                if (lrIndex < metaGradients.Length && _metaSGDOptions.LearnLearningRate)
                {
                    double lrGrad = NumOps.ToDouble(metaGradients[lrIndex]);
                    // Beta1 affects momentum - smaller scale
                    metaGradients[paramIndex++] = NumOps.FromDouble(lrGrad * 0.001);
                    // Beta2 affects variance - even smaller scale
                    metaGradients[paramIndex++] = NumOps.FromDouble(lrGrad * 0.0001);
                    // Epsilon is very stable - minimal gradient
                    metaGradients[paramIndex++] = NumOps.FromDouble(lrGrad * 1e-8);
                }
                else
                {
                    metaGradients[paramIndex++] = NumOps.Zero;
                    metaGradients[paramIndex++] = NumOps.Zero;
                    metaGradients[paramIndex++] = NumOps.Zero;
                }
            }
        }

        return metaGradients;
    }

    /// <summary>
    /// Computes the loss with a perturbed learning rate for finite difference gradient.
    /// </summary>
    private T ComputePerturbedLoss(
        int parameterIndex,
        double epsilon,
        Vector<T> initialParams,
        TInput supportInputs,
        TOutput supportOutputs,
        TInput queryInputs,
        TOutput queryOutputs)
    {
        // Clone optimizer to avoid corrupting main optimizer state during perturbation
        var tempOptimizer = _optimizer.Clone();

        // Get current learning rate and perturb it
        var originalLR = tempOptimizer.GetLearningRate(parameterIndex);
        tempOptimizer.SetLearningRate(parameterIndex, NumOps.Add(originalLR, NumOps.FromDouble(epsilon)));

        // Clone model and run adaptation
        var tempModel = CloneModel();
        tempModel.SetParameters(initialParams);

        var currentParams = new Vector<T>(initialParams.Length);
        for (int i = 0; i < initialParams.Length; i++)
        {
            currentParams[i] = initialParams[i];
        }

        // Run inner loop with perturbed learning rate using cloned optimizer
        for (int step = 0; step < _metaSGDOptions.InnerSteps; step++)
        {
            var gradients = ComputeGradients(tempModel, supportInputs, supportOutputs);
            for (int i = 0; i < currentParams.Length; i++)
            {
                currentParams[i] = tempOptimizer.UpdateParameter(i, currentParams[i], gradients[i]);
            }
            tempModel.SetParameters(currentParams);
        }

        // Compute query loss
        var queryPredictions = tempModel.Predict(queryInputs);
        return ComputeLossFromOutput(queryPredictions, queryOutputs);
    }

    /// <summary>
    /// Initializes the per-parameter optimizer with warm-start values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Sets reasonable initial values for all learned per-parameter coefficients
    /// based on the configuration. This helps with training stability and faster
    /// convergence.
    /// </para>
    /// </remarks>
    private void InitializeOptimizer()
    {
        int numParams = MetaModel.GetParameters().Length;

        for (int i = 0; i < numParams; i++)
        {
            // Initialize learning rate based on initialization strategy
            T initialLR = GetInitialLearningRate(i, numParams);
            _optimizer.SetLearningRate(i, initialLR);

            // Initialize momentum if enabled
            if (_metaSGDOptions.LearnMomentum)
            {
                _optimizer.SetMomentum(i, NumOps.FromDouble(0.9));
            }

            // Initialize direction if enabled
            if (_metaSGDOptions.LearnDirection)
            {
                _optimizer.SetDirection(i, NumOps.One);
            }

            // Initialize Adam parameters if using Adam
            if (_metaSGDOptions.UpdateRuleType == MetaSGDUpdateRuleType.Adam)
            {
                _optimizer.SetAdamBeta1(i, NumOps.FromDouble(_metaSGDOptions.AdamBeta1Init));
                _optimizer.SetAdamBeta2(i, NumOps.FromDouble(_metaSGDOptions.AdamBeta2Init));
                _optimizer.SetAdamEpsilon(i, NumOps.FromDouble(_metaSGDOptions.AdamEpsilonInit));
            }
        }
    }

    /// <summary>
    /// Gets the initial learning rate for a parameter based on the initialization strategy.
    /// </summary>
    /// <param name="paramIndex">Index of the parameter.</param>
    /// <param name="totalParams">Total number of parameters.</param>
    /// <returns>The initial learning rate for this parameter.</returns>
    private T GetInitialLearningRate(int paramIndex, int totalParams)
    {
        double baseLR = _metaSGDOptions.InnerLearningRate;

        switch (_metaSGDOptions.LearningRateInitialization)
        {
            case MetaSGDLearningRateInitialization.Uniform:
                return NumOps.FromDouble(baseLR);

            case MetaSGDLearningRateInitialization.Random:
                double range = _metaSGDOptions.LearningRateInitRange;
                double randomLR = baseLR + (RandomGenerator.NextDouble() - 0.5) * range;
                return NumOps.FromDouble(Math.Max(_metaSGDOptions.MinLearningRate,
                    Math.Min(_metaSGDOptions.MaxLearningRate, randomLR)));

            case MetaSGDLearningRateInitialization.LayerBased:
                if (_metaSGDOptions.UseLayerWiseDecay)
                {
                    // Approximate layer from parameter index
                    double layerFraction = (double)paramIndex / totalParams;
                    double decay = Math.Pow(_metaSGDOptions.LayerDecayFactor, layerFraction * 10);
                    return NumOps.FromDouble(baseLR * decay);
                }
                return NumOps.FromDouble(baseLR);

            case MetaSGDLearningRateInitialization.Xavier:
                // Xavier-inspired: scale by 1/sqrt(n)
                double xavierScale = 1.0 / Math.Sqrt(totalParams);
                return NumOps.FromDouble(baseLR * xavierScale);

            case MetaSGDLearningRateInitialization.MagnitudeBased:
                // Would need parameter values - use uniform for simplicity
                return NumOps.FromDouble(baseLR);

            default:
                return NumOps.FromDouble(baseLR);
        }
    }
}

/// <summary>
/// Per-parameter optimizer for Meta-SGD that learns individual optimization coefficients.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This optimizer maintains learned coefficients for each parameter:
/// - Learning rates: α_i for each parameter
/// - Momentum: β_i for each parameter (optional)
/// - Direction: d_i for each parameter (optional)
/// - Adam parameters: beta1, beta2, epsilon (if using Adam)
/// </para>
/// <para>
/// <b>For Beginners:</b> This is a special optimizer where each weight in the
/// network gets its own set of optimization settings that are learned during
/// meta-training.
/// </para>
/// </remarks>
public class PerParameterOptimizer<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _numParameters;
    private readonly MetaSGDOptions<T, TInput, TOutput> _options;

    // Per-parameter learned coefficients
    private readonly T[] _learningRates;
    private readonly T[] _momentums;
    private readonly T[] _directions;

    // Adam-specific parameters
    private readonly T[] _adamBeta1;
    private readonly T[] _adamBeta2;
    private readonly T[] _adamEpsilon;

    // Optimizer state
    private readonly T[] _firstMoments;
    private readonly T[] _secondMoments;
    private readonly T[] _velocities;

    /// <summary>
    /// Gets the number of model parameters this optimizer manages.
    /// </summary>
    public int NumParameters => _numParameters;

    /// <summary>
    /// Initializes a new instance of the PerParameterOptimizer.
    /// </summary>
    /// <param name="numParameters">Number of model parameters.</param>
    /// <param name="options">Meta-SGD options.</param>
    public PerParameterOptimizer(int numParameters, MetaSGDOptions<T, TInput, TOutput> options)
    {
        _numParameters = numParameters;
        _options = options;

        // Initialize per-parameter arrays
        _learningRates = new T[numParameters];
        _momentums = new T[numParameters];
        _directions = new T[numParameters];
        _adamBeta1 = new T[numParameters];
        _adamBeta2 = new T[numParameters];
        _adamEpsilon = new T[numParameters];
        _firstMoments = new T[numParameters];
        _secondMoments = new T[numParameters];
        _velocities = new T[numParameters];

        // Initialize with default values
        for (int i = 0; i < numParameters; i++)
        {
            _learningRates[i] = NumOps.FromDouble(options.InnerLearningRate);
            _momentums[i] = NumOps.Zero;
            _directions[i] = NumOps.One;
            _adamBeta1[i] = NumOps.FromDouble(options.AdamBeta1Init);
            _adamBeta2[i] = NumOps.FromDouble(options.AdamBeta2Init);
            _adamEpsilon[i] = NumOps.FromDouble(options.AdamEpsilonInit);
            _firstMoments[i] = NumOps.Zero;
            _secondMoments[i] = NumOps.Zero;
            _velocities[i] = NumOps.Zero;
        }
    }

    /// <summary>
    /// Updates a single parameter using its learned optimization coefficients.
    /// </summary>
    /// <param name="parameterIndex">Index of the parameter to update.</param>
    /// <param name="parameter">Current parameter value.</param>
    /// <param name="gradient">Gradient for this parameter.</param>
    /// <returns>Updated parameter value.</returns>
    /// <remarks>
    /// <para>
    /// Applies the learned per-parameter update rule:
    /// - SGD: θ = θ - α_i × d_i × g
    /// - SGDWithMomentum: v = β_i × v + α_i × d_i × g; θ = θ - v
    /// - Adam: Full Adam with learned β1, β2, ε
    /// </para>
    /// </remarks>
    public T UpdateParameter(int parameterIndex, T parameter, T gradient)
    {
        var lr = _learningRates[parameterIndex];
        T update;

        switch (_options.UpdateRuleType)
        {
            case MetaSGDUpdateRuleType.SGD:
                if (_options.LearnDirection)
                {
                    update = NumOps.Multiply(lr, NumOps.Multiply(_directions[parameterIndex], gradient));
                }
                else
                {
                    update = NumOps.Multiply(lr, gradient);
                }
                break;

            case MetaSGDUpdateRuleType.SGDWithMomentum:
                T gradUpdate;
                if (_options.LearnDirection)
                {
                    gradUpdate = NumOps.Multiply(lr, NumOps.Multiply(_directions[parameterIndex], gradient));
                }
                else
                {
                    gradUpdate = NumOps.Multiply(lr, gradient);
                }
                update = NumOps.Add(
                    NumOps.Multiply(_momentums[parameterIndex], _velocities[parameterIndex]),
                    gradUpdate);
                _velocities[parameterIndex] = update;
                break;

            case MetaSGDUpdateRuleType.Adam:
                // Adam update with learned parameters
                T oneMinusBeta1 = NumOps.Subtract(NumOps.One, _adamBeta1[parameterIndex]);
                T oneMinusBeta2 = NumOps.Subtract(NumOps.One, _adamBeta2[parameterIndex]);

                // Update biased first moment
                _firstMoments[parameterIndex] = NumOps.Add(
                    NumOps.Multiply(_adamBeta1[parameterIndex], _firstMoments[parameterIndex]),
                    NumOps.Multiply(oneMinusBeta1, gradient));

                // Update biased second moment
                T gradSquared = NumOps.Multiply(gradient, gradient);
                _secondMoments[parameterIndex] = NumOps.Add(
                    NumOps.Multiply(_adamBeta2[parameterIndex], _secondMoments[parameterIndex]),
                    NumOps.Multiply(oneMinusBeta2, gradSquared));

                // Bias correction (simplified - assuming many iterations)
                T biasCorrectedFirst = _firstMoments[parameterIndex];
                T biasCorrectedSecond = _secondMoments[parameterIndex];

                // Compute update
                double secondMomentVal = NumOps.ToDouble(biasCorrectedSecond);
                T sqrtSecond = NumOps.FromDouble(Math.Sqrt(Math.Max(0, secondMomentVal)));
                T denominator = NumOps.Add(sqrtSecond, _adamEpsilon[parameterIndex]);
                update = NumOps.Divide(NumOps.Multiply(lr, biasCorrectedFirst), denominator);
                break;

            case MetaSGDUpdateRuleType.RMSprop:
                // RMSprop update
                T decayRate = NumOps.FromDouble(0.9);
                T oneMinusDecay = NumOps.FromDouble(0.1);
                _secondMoments[parameterIndex] = NumOps.Add(
                    NumOps.Multiply(decayRate, _secondMoments[parameterIndex]),
                    NumOps.Multiply(oneMinusDecay, NumOps.Multiply(gradient, gradient)));

                double secondVal = NumOps.ToDouble(_secondMoments[parameterIndex]);
                T sqrtSecondRms = NumOps.FromDouble(Math.Sqrt(Math.Max(0, secondVal)));
                T denomRms = NumOps.Add(sqrtSecondRms, NumOps.FromDouble(1e-6));
                update = NumOps.Divide(NumOps.Multiply(lr, gradient), denomRms);
                break;

            case MetaSGDUpdateRuleType.AdaGrad:
                // AdaGrad update
                _secondMoments[parameterIndex] = NumOps.Add(
                    _secondMoments[parameterIndex],
                    NumOps.Multiply(gradient, gradient));

                double accumVal = NumOps.ToDouble(_secondMoments[parameterIndex]);
                T sqrtAccum = NumOps.FromDouble(Math.Sqrt(Math.Max(0, accumVal)));
                T denomAda = NumOps.Add(sqrtAccum, NumOps.FromDouble(1e-6));
                update = NumOps.Divide(NumOps.Multiply(lr, gradient), denomAda);
                break;

            case MetaSGDUpdateRuleType.AdaDelta:
                // AdaDelta update (simplified)
                T rho = NumOps.FromDouble(0.95);
                T oneMinusRho = NumOps.FromDouble(0.05);

                _secondMoments[parameterIndex] = NumOps.Add(
                    NumOps.Multiply(rho, _secondMoments[parameterIndex]),
                    NumOps.Multiply(oneMinusRho, NumOps.Multiply(gradient, gradient)));

                double accGradVal = NumOps.ToDouble(_secondMoments[parameterIndex]);
                T rmsGrad = NumOps.FromDouble(Math.Sqrt(Math.Max(1e-6, accGradVal)));

                // Use velocity as accumulated parameter updates
                double accDeltaVal = NumOps.ToDouble(_velocities[parameterIndex]);
                T rmsDelta = NumOps.FromDouble(Math.Sqrt(Math.Max(1e-6, accDeltaVal)));

                update = NumOps.Multiply(NumOps.Divide(rmsDelta, rmsGrad), gradient);

                // Update accumulated parameter updates
                _velocities[parameterIndex] = NumOps.Add(
                    NumOps.Multiply(rho, _velocities[parameterIndex]),
                    NumOps.Multiply(oneMinusRho, NumOps.Multiply(update, update)));
                break;

            default:
                update = NumOps.Multiply(lr, gradient);
                break;
        }

        return NumOps.Subtract(parameter, update);
    }

    /// <summary>
    /// Updates the meta-parameters (learned coefficients) of the optimizer.
    /// </summary>
    /// <param name="metaGradients">Gradients for the meta-parameters.</param>
    /// <remarks>
    /// <para>
    /// Updates learning rates, momentum, direction, and Adam parameters based on
    /// the computed meta-gradients. Also applies regularization and clipping.
    /// </para>
    /// </remarks>
    public void UpdateMetaParameters(Vector<T> metaGradients)
    {
        int index = 0;
        double metaLR = _options.OuterLearningRate;

        // Update learning rates
        if (_options.LearnLearningRate)
        {
            for (int i = 0; i < _numParameters; i++)
            {
                T update = NumOps.Multiply(metaGradients[index++], NumOps.FromDouble(metaLR));
                _learningRates[i] = NumOps.Subtract(_learningRates[i], update);
            }
        }

        // Update momentums
        if (_options.LearnMomentum)
        {
            for (int i = 0; i < _numParameters; i++)
            {
                T update = NumOps.Multiply(metaGradients[index++], NumOps.FromDouble(metaLR));
                _momentums[i] = NumOps.Subtract(_momentums[i], update);
            }
        }

        // Update directions
        if (_options.LearnDirection)
        {
            for (int i = 0; i < _numParameters; i++)
            {
                T update = NumOps.Multiply(metaGradients[index++], NumOps.FromDouble(metaLR));
                _directions[i] = NumOps.Subtract(_directions[i], update);
            }
        }

        // Update Adam parameters
        if (_options.UpdateRuleType == MetaSGDUpdateRuleType.Adam && _options.LearnAdamBetas)
        {
            for (int i = 0; i < _numParameters; i++)
            {
                T updateBeta1 = NumOps.Multiply(metaGradients[index++], NumOps.FromDouble(metaLR));
                T updateBeta2 = NumOps.Multiply(metaGradients[index++], NumOps.FromDouble(metaLR));
                T updateEps = NumOps.Multiply(metaGradients[index++], NumOps.FromDouble(metaLR));

                _adamBeta1[i] = NumOps.Subtract(_adamBeta1[i], updateBeta1);
                _adamBeta2[i] = NumOps.Subtract(_adamBeta2[i], updateBeta2);
                _adamEpsilon[i] = NumOps.Subtract(_adamEpsilon[i], updateEps);
            }
        }

        // Apply regularization and clipping
        ApplyRegularization();
    }

    /// <summary>
    /// Gets the total number of meta-parameters being learned.
    /// </summary>
    /// <returns>Count of learned meta-parameters.</returns>
    public int GetMetaParameterCount()
    {
        int count = 0;

        if (_options.LearnLearningRate)
            count += _numParameters;

        if (_options.LearnMomentum)
            count += _numParameters;

        if (_options.LearnDirection)
            count += _numParameters;

        if (_options.UpdateRuleType == MetaSGDUpdateRuleType.Adam && _options.LearnAdamBetas)
            count += 3 * _numParameters; // beta1, beta2, epsilon

        return count;
    }

    /// <summary>
    /// Creates a deep copy of this per-parameter optimizer.
    /// </summary>
    /// <returns>A new PerParameterOptimizer with copied state.</returns>
    public PerParameterOptimizer<T, TInput, TOutput> Clone()
    {
        var cloned = new PerParameterOptimizer<T, TInput, TOutput>(_numParameters, _options);

        Array.Copy(_learningRates, cloned._learningRates, _numParameters);
        Array.Copy(_momentums, cloned._momentums, _numParameters);
        Array.Copy(_directions, cloned._directions, _numParameters);
        Array.Copy(_adamBeta1, cloned._adamBeta1, _numParameters);
        Array.Copy(_adamBeta2, cloned._adamBeta2, _numParameters);
        Array.Copy(_adamEpsilon, cloned._adamEpsilon, _numParameters);
        Array.Copy(_firstMoments, cloned._firstMoments, _numParameters);
        Array.Copy(_secondMoments, cloned._secondMoments, _numParameters);
        Array.Copy(_velocities, cloned._velocities, _numParameters);

        return cloned;
    }

    #region Getter/Setter Methods

    /// <summary>Gets the learning rate for a specific parameter.</summary>
    public T GetLearningRate(int parameterIndex) => _learningRates[parameterIndex];

    /// <summary>Sets the learning rate for a specific parameter.</summary>
    public void SetLearningRate(int parameterIndex, T learningRate) => _learningRates[parameterIndex] = learningRate;

    /// <summary>Sets the momentum for a specific parameter.</summary>
    public void SetMomentum(int parameterIndex, T momentum) => _momentums[parameterIndex] = momentum;

    /// <summary>Sets the direction for a specific parameter.</summary>
    public void SetDirection(int parameterIndex, T direction) => _directions[parameterIndex] = direction;

    /// <summary>Sets Adam beta1 for a specific parameter.</summary>
    public void SetAdamBeta1(int parameterIndex, T beta1) => _adamBeta1[parameterIndex] = beta1;

    /// <summary>Sets Adam beta2 for a specific parameter.</summary>
    public void SetAdamBeta2(int parameterIndex, T beta2) => _adamBeta2[parameterIndex] = beta2;

    /// <summary>Sets Adam epsilon for a specific parameter.</summary>
    public void SetAdamEpsilon(int parameterIndex, T epsilon) => _adamEpsilon[parameterIndex] = epsilon;

    #endregion

    /// <summary>
    /// Applies regularization and clipping to learned coefficients.
    /// </summary>
    private void ApplyRegularization()
    {
        // Apply L2 regularization to learning rates
        if (_options.LearningRateL2Reg > 0.0)
        {
            T regFactor = NumOps.FromDouble(1.0 - _options.LearningRateL2Reg);
            for (int i = 0; i < _numParameters; i++)
            {
                _learningRates[i] = NumOps.Multiply(_learningRates[i], regFactor);
            }
        }

        // Clip learning rates to valid range
        double minLR = _options.MinLearningRate;
        double maxLR = _options.MaxLearningRate;

        for (int i = 0; i < _numParameters; i++)
        {
            double val = NumOps.ToDouble(_learningRates[i]);
            _learningRates[i] = NumOps.FromDouble(Math.Max(minLR, Math.Min(maxLR, val)));
        }

        // Clip momentum to [0, 1]
        if (_options.LearnMomentum)
        {
            for (int i = 0; i < _numParameters; i++)
            {
                double val = NumOps.ToDouble(_momentums[i]);
                _momentums[i] = NumOps.FromDouble(Math.Max(0.0, Math.Min(1.0, val)));
            }
        }

        // Clip Adam betas to valid range
        if (_options.UpdateRuleType == MetaSGDUpdateRuleType.Adam && _options.LearnAdamBetas)
        {
            const double minBeta = 0.0001;
            const double maxBeta = 0.9999;
            const double minEps = 1e-10;
            const double maxEps = 1e-4;

            for (int i = 0; i < _numParameters; i++)
            {
                double b1 = NumOps.ToDouble(_adamBeta1[i]);
                double b2 = NumOps.ToDouble(_adamBeta2[i]);
                double eps = NumOps.ToDouble(_adamEpsilon[i]);

                _adamBeta1[i] = NumOps.FromDouble(Math.Max(minBeta, Math.Min(maxBeta, b1)));
                _adamBeta2[i] = NumOps.FromDouble(Math.Max(minBeta, Math.Min(maxBeta, b2)));
                _adamEpsilon[i] = NumOps.FromDouble(Math.Max(minEps, Math.Min(maxEps, eps)));
            }
        }
    }
}

/// <summary>
/// Wrapper model for Meta-SGD adapted models that includes the per-parameter optimizer.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input type.</typeparam>
/// <typeparam name="TOutput">The output type.</typeparam>
/// <remarks>
/// <para>
/// This model wraps an adapted model along with its per-parameter optimizer,
/// allowing for further adaptation or inspection of learned coefficients.
/// </para>
/// </remarks>
public class MetaSGDAdaptedModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private readonly IFullModel<T, TInput, TOutput> _model;
    private readonly PerParameterOptimizer<T, TInput, TOutput> _optimizer;
    private readonly MetaSGDOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// Initializes a new instance of the MetaSGDAdaptedModel.
    /// </summary>
    /// <param name="model">The adapted model.</param>
    /// <param name="optimizer">The per-parameter optimizer used for adaptation.</param>
    /// <param name="options">The Meta-SGD options.</param>
    public MetaSGDAdaptedModel(
        IFullModel<T, TInput, TOutput> model,
        PerParameterOptimizer<T, TInput, TOutput> optimizer,
        MetaSGDOptions<T, TInput, TOutput> options)
    {
        Guard.NotNull(model);
        _model = model;
        Guard.NotNull(optimizer);
        _optimizer = optimizer;
        Guard.NotNull(options);
        _options = options;
    }

    /// <summary>
    /// Gets the model metadata.
    /// </summary>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <summary>
    /// Gets the per-parameter optimizer (for inspection or further adaptation).
    /// </summary>
    public PerParameterOptimizer<T, TInput, TOutput> Optimizer => _optimizer;

    /// <summary>
    /// Makes predictions using the adapted model.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <returns>The model predictions.</returns>
    public TOutput Predict(TInput input)
    {
        return _model.Predict(input);
    }

    /// <summary>
    /// Trains the model on the given data.
    /// </summary>
    /// <param name="inputs">The input data.</param>
    /// <param name="targets">The target outputs.</param>
    /// <remarks>
    /// <para>
    /// For Meta-SGD adapted models, training is typically done through the
    /// meta-learning adaptation process rather than direct training.
    /// This method delegates to the underlying model's training.
    /// </para>
    /// </remarks>
    public void Train(TInput inputs, TOutput targets)
    {
        _model.Train(inputs, targets);
    }

    /// <summary>
    /// Gets the model metadata.
    /// </summary>
    /// <returns>The metadata for this model.</returns>
    public ModelMetadata<T> GetModelMetadata()
    {
        return Metadata;
    }

    /// <summary>
    /// Gets the current model parameters.
    /// </summary>
    /// <returns>The parameter vector.</returns>
    public Vector<T> GetParameters()
    {
        return _model.GetParameters();
    }

    /// <summary>
    /// Sets the model parameters.
    /// </summary>
    /// <param name="parameters">The new parameters.</param>
    public void SetParameters(Vector<T> parameters)
    {
        _model.SetParameters(parameters);
    }
}
