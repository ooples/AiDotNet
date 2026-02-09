using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for the Meta-SGD (Meta Stochastic Gradient Descent) algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// Meta-SGD extends MAML by learning not just the model initialization but also per-parameter
/// learning rates, momentum terms, and update directions. This effectively learns a custom
/// optimizer configuration for each parameter, enabling highly specialized adaptation strategies.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of Meta-SGD as "learning how to learn" at the finest grain:
/// </para>
/// <para>
/// In standard training, you pick one learning rate for all parameters. But different parts
/// of a neural network might benefit from different learning rates. Meta-SGD figures this out
/// automatically by learning:
/// - <b>Per-parameter learning rates:</b> Some weights need small updates, others larger
/// - <b>Per-parameter momentum:</b> Some weights benefit from momentum, others don't
/// - <b>Update directions:</b> Sometimes the gradient direction should be flipped or scaled
/// </para>
/// <para>
/// <b>Algorithm Overview:</b>
/// <code>
/// # For each model parameter θ_i, learn:
/// #   - α_i: the optimal learning rate
/// #   - β_i: the optimal momentum coefficient
/// #   - d_i: the optimal update direction/scaling
///
/// # Meta-training:
/// for each task in task_batch:
///     adapted_params = initial_params.copy()
///     for step in range(K_inner):
///         gradients = compute_gradients(adapted_params, support_set)
///         for i in range(num_params):
///             # Per-parameter update rule
///             update_i = α_i * d_i * gradients[i] + β_i * velocity[i]
///             adapted_params[i] -= update_i
///
///     query_loss = evaluate(adapted_params, query_set)
///
/// # Meta-update: optimize α_i, β_i, d_i using query_loss gradient
/// </code>
/// </para>
/// <para>
/// <b>Key Insights:</b>
/// 1. Per-parameter optimization allows heterogeneous learning rates across layers
/// 2. First-order method: no Hessian computation needed, much faster than second-order MAML
/// 3. Learned optimizers reveal which parameters are important for quick adaptation
/// 4. Can combine with various base update rules (SGD, Adam, RMSprop)
/// </para>
/// <para>
/// <b>Reference:</b> Li, Z., Zhou, F., Chen, F., &amp; Li, H. (2017).
/// Meta-SGD: Learning to Learn Quickly for Few-Shot Learning.
/// </para>
/// </remarks>
public class MetaSGDOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>
    /// Gets or sets the meta-model to be trained. This is the only required property.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The model must implement IFullModel to support parameter getting/setting
    /// required for Meta-SGD's per-parameter optimization. Each parameter in the model
    /// will have its own learned learning rate, momentum, and direction coefficients.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the neural network whose parameters you want to
    /// meta-train. Meta-SGD will learn how to optimize each weight in this network.
    /// </para>
    /// </remarks>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }

    #endregion

    #region Optional Properties with Defaults

    /// <summary>
    /// Gets or sets the loss function for training.
    /// </summary>
    /// <value>Default: null (uses model's default loss function if available).</value>
    /// <remarks>
    /// <para>
    /// The loss function is used both in the inner loop (task adaptation) and
    /// outer loop (meta-optimization) to guide the learning of per-parameter optimizers.
    /// </para>
    /// </remarks>
    public ILossFunction<T>? LossFunction { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for meta-parameter updates (outer loop).
    /// </summary>
    /// <value>Default: null (uses built-in Adam optimizer with OuterLearningRate).</value>
    /// <remarks>
    /// <para>
    /// This optimizer updates the learned per-parameter learning rates, momentums,
    /// and directions. Adam is typically used as it handles the sparse gradients
    /// from per-parameter optimization well.
    /// </para>
    /// </remarks>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for inner-loop adaptation.
    /// </summary>
    /// <value>Default: null (uses the learned per-parameter optimizer).</value>
    /// <remarks>
    /// <para>
    /// In Meta-SGD, this is typically not used directly because the per-parameter
    /// optimizer with learned coefficients replaces standard optimization.
    /// This is provided for compatibility with the base interface.
    /// </para>
    /// </remarks>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the episodic data loader for sampling tasks.
    /// </summary>
    /// <value>Default: null (tasks must be provided manually to MetaTrain).</value>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// Gets or sets the learning rate for the inner loop (task adaptation).
    /// </summary>
    /// <value>Default: 0.01.</value>
    /// <remarks>
    /// <para>
    /// In Meta-SGD, this serves as the initial value for per-parameter learning rates
    /// when using uniform initialization. During meta-training, each parameter will
    /// learn its own optimal learning rate, which may diverge from this initial value.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the starting point for all learning rates.
    /// Meta-SGD will adjust each one individually as it learns.
    /// </para>
    /// </remarks>
    public double InnerLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the learning rate for the outer loop (meta-optimization).
    /// </summary>
    /// <value>Default: 0.001.</value>
    /// <remarks>
    /// <para>
    /// This controls how quickly the per-parameter learning rates, momentums, and
    /// directions are updated during meta-training. A lower value provides more
    /// stable learning but slower convergence.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This controls how fast Meta-SGD learns the optimal
    /// per-parameter configurations. Too high causes instability, too low is slow.
    /// </para>
    /// </remarks>
    public double OuterLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of gradient steps to take during inner loop adaptation.
    /// </summary>
    /// <value>Default: 5 (typical for few-shot learning).</value>
    /// <remarks>
    /// <para>
    /// Meta-SGD uses first-order optimization, so more adaptation steps are computationally
    /// cheaper than in MAML. However, too many steps can lead to overfitting on the
    /// support set.
    /// </para>
    /// </remarks>
    public int AdaptationSteps { get; set; } = 5;

    /// <summary>
    /// Gets or sets the number of inner steps during meta-training.
    /// </summary>
    /// <value>Default: 5 (matches AdaptationSteps by default).</value>
    /// <remarks>
    /// <para>
    /// This can be different from AdaptationSteps to allow for different behavior
    /// during training vs. adaptation. Some implementations use fewer inner steps
    /// during training for efficiency.
    /// </para>
    /// </remarks>
    public int InnerSteps { get; set; } = 5;

    /// <summary>
    /// Gets or sets the number of tasks to sample per meta-training iteration.
    /// </summary>
    /// <value>Default: 4 (typical meta-batch size).</value>
    /// <remarks>
    /// <para>
    /// Larger batch sizes provide more stable gradients for the meta-parameters
    /// but require more memory and computation per iteration.
    /// </para>
    /// </remarks>
    public int MetaBatchSize { get; set; } = 4;

    /// <summary>
    /// Gets or sets the total number of meta-training iterations to perform.
    /// </summary>
    /// <value>Default: 10000 (Meta-SGD typically needs many iterations).</value>
    /// <remarks>
    /// <para>
    /// Meta-SGD often requires more iterations than MAML because it has more
    /// meta-parameters to learn (per-parameter coefficients). Monitor the
    /// validation loss to determine when to stop.
    /// </para>
    /// </remarks>
    public int NumMetaIterations { get; set; } = 10000;

    /// <summary>
    /// Gets or sets whether to use first-order approximation.
    /// </summary>
    /// <value>Default: true (Meta-SGD is inherently first-order).</value>
    /// <remarks>
    /// <para>
    /// Meta-SGD is designed as a first-order algorithm. Unlike MAML, it doesn't
    /// require computing gradients through the adaptation process. This property
    /// is always effectively true for standard Meta-SGD.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> First-order means Meta-SGD doesn't need to compute
    /// complex second-order derivatives, making it much faster than MAML.
    /// </para>
    /// </remarks>
    public bool UseFirstOrder { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum gradient norm for gradient clipping.
    /// </summary>
    /// <value>Default: 10.0 (prevents exploding gradients during meta-training).</value>
    /// <remarks>
    /// <para>
    /// Gradient clipping is particularly important in Meta-SGD because the
    /// per-parameter learning rates can amplify gradients if they grow too large.
    /// </para>
    /// </remarks>
    public double? GradientClipThreshold { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    /// <value>Default: null (non-deterministic).</value>
    public int? RandomSeed { get => Seed; set => Seed = value; }

    /// <summary>
    /// Gets or sets the number of tasks to use for evaluation.
    /// </summary>
    /// <value>Default: 100.</value>
    public int EvaluationTasks { get; set; } = 100;

    /// <summary>
    /// Gets or sets how often to evaluate during meta-training.
    /// </summary>
    /// <value>Default: 500.</value>
    public int EvaluationFrequency { get; set; } = 500;

    /// <summary>
    /// Gets or sets whether to save checkpoints during training.
    /// </summary>
    /// <value>Default: false.</value>
    public bool EnableCheckpointing { get; set; } = false;

    /// <summary>
    /// Gets or sets how often to save checkpoints.
    /// </summary>
    /// <value>Default: 1000.</value>
    public int CheckpointFrequency { get; set; } = 1000;

    #endregion

    #region Meta-SGD Specific Properties - Core

    /// <summary>
    /// Gets or sets the update rule type for per-parameter optimization.
    /// </summary>
    /// <value>Default: MetaSGDUpdateRuleType.SGD.</value>
    /// <remarks>
    /// <para>
    /// <b>Update Rule Types:</b>
    /// - <b>SGD:</b> Standard gradient descent with learned learning rates
    /// - <b>SGDWithMomentum:</b> Adds learned momentum terms per parameter
    /// - <b>Adam:</b> Full Adam optimizer with learned beta parameters
    /// - <b>RMSprop:</b> RMSprop with learned decay rates
    /// - <b>AdaGrad:</b> AdaGrad with learned accumulation
    /// - <b>AdaDelta:</b> AdaDelta with learned decay
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Start with SGD and only move to more complex rules
    /// if you find SGD isn't working well for your problem.
    /// </para>
    /// </remarks>
    public MetaSGDUpdateRuleType UpdateRuleType { get; set; } = MetaSGDUpdateRuleType.SGD;

    /// <summary>
    /// Gets or sets whether to learn per-parameter learning rates.
    /// </summary>
    /// <value>Default: true (the core feature of Meta-SGD).</value>
    /// <remarks>
    /// <para>
    /// This is the defining feature of Meta-SGD. Each parameter gets its own
    /// learned learning rate that is optimized during meta-training to enable
    /// fast adaptation to new tasks.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Keep this true! This is what makes Meta-SGD special.
    /// </para>
    /// </remarks>
    public bool LearnLearningRate { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to learn per-parameter momentum coefficients.
    /// </summary>
    /// <value>Default: false (adds complexity, only enable if needed).</value>
    /// <remarks>
    /// <para>
    /// When enabled, each parameter learns its own momentum coefficient.
    /// This can help with parameters that benefit from momentum-based updates
    /// but adds to the number of meta-parameters to learn.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Leave this false initially. Only enable if you find
    /// that per-parameter learning rates alone aren't sufficient.
    /// </para>
    /// </remarks>
    public bool LearnMomentum { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to learn per-parameter update direction signs.
    /// </summary>
    /// <value>Default: true (helps with gradient sign ambiguity).</value>
    /// <remarks>
    /// <para>
    /// The direction parameter can flip or scale the gradient direction for each
    /// parameter. This helps when the natural gradient direction isn't optimal
    /// for fast adaptation.
    /// </para>
    /// <para>
    /// <b>Mathematical formulation:</b>
    /// θ_i' = θ_i - α_i × d_i × ∇_θ_i L
    /// where d_i is the learned direction scaling factor.
    /// </para>
    /// </remarks>
    public bool LearnDirection { get; set; } = true;

    #endregion

    #region Meta-SGD Specific Properties - Learning Rate Configuration

    /// <summary>
    /// Gets or sets the initialization strategy for per-parameter learning rates.
    /// </summary>
    /// <value>Default: MetaSGDLearningRateInitialization.Uniform.</value>
    /// <remarks>
    /// <para>
    /// <b>Initialization strategies:</b>
    /// - <b>Uniform:</b> All learning rates start at InnerLearningRate
    /// - <b>Random:</b> Random values within LearningRateInitRange
    /// - <b>MagnitudeBased:</b> Based on parameter magnitudes
    /// - <b>LayerBased:</b> Different rates per layer depth
    /// - <b>Xavier:</b> Xavier-style initialization
    /// </para>
    /// </remarks>
    public MetaSGDLearningRateInitialization LearningRateInitialization { get; set; } = MetaSGDLearningRateInitialization.Uniform;

    /// <summary>
    /// Gets or sets the initialization range for learning rates when using random initialization.
    /// </summary>
    /// <value>Default: 0.1.</value>
    /// <remarks>
    /// <para>
    /// When using Random initialization, learning rates are initialized uniformly
    /// in [InnerLearningRate - range/2, InnerLearningRate + range/2].
    /// </para>
    /// </remarks>
    public double LearningRateInitRange { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the minimum allowed per-parameter learning rate.
    /// </summary>
    /// <value>Default: 1e-6 (prevents learning rates from becoming too small).</value>
    /// <remarks>
    /// <para>
    /// Clipping learning rates to a minimum prevents parameters from becoming
    /// "frozen" during adaptation.
    /// </para>
    /// </remarks>
    public double MinLearningRate { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the maximum allowed per-parameter learning rate.
    /// </summary>
    /// <value>Default: 1.0 (prevents learning rates from becoming too large).</value>
    /// <remarks>
    /// <para>
    /// Clipping learning rates to a maximum prevents unstable updates that
    /// could blow up during adaptation.
    /// </para>
    /// </remarks>
    public double MaxLearningRate { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the L2 regularization coefficient for learned learning rates.
    /// </summary>
    /// <value>Default: 0.0 (no regularization).</value>
    /// <remarks>
    /// <para>
    /// Regularizing learning rates prevents them from growing too large and
    /// helps with generalization across tasks.
    /// </para>
    /// </remarks>
    public double LearningRateL2Reg { get; set; } = 0.0;

    #endregion

    #region Meta-SGD Specific Properties - Layer-wise Configuration

    /// <summary>
    /// Gets or sets whether to apply layer-wise learning rate decay.
    /// </summary>
    /// <value>Default: false.</value>
    /// <remarks>
    /// <para>
    /// When enabled, deeper layers get smaller initial learning rates.
    /// This can help with training stability in very deep networks.
    /// </para>
    /// <para>
    /// <b>Formula:</b> layer_lr = base_lr × (LayerDecayFactor ^ layer_depth)
    /// </para>
    /// </remarks>
    public bool UseLayerWiseDecay { get; set; } = false;

    /// <summary>
    /// Gets or sets the decay factor per layer depth.
    /// </summary>
    /// <value>Default: 0.9.</value>
    /// <remarks>
    /// <para>
    /// Only used when UseLayerWiseDecay is true. A value of 0.9 means each
    /// successive layer has 90% of the previous layer's learning rate.
    /// </para>
    /// </remarks>
    public double LayerDecayFactor { get; set; } = 0.9;

    #endregion

    #region Meta-SGD Specific Properties - Parameter Grouping

    /// <summary>
    /// Gets or sets whether to use parameter grouping.
    /// </summary>
    /// <value>Default: false.</value>
    /// <remarks>
    /// <para>
    /// When enabled, parameters are grouped and share learning rates within
    /// groups. This reduces the number of meta-parameters and can improve
    /// generalization when the model has many parameters.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Enable this for very large models to reduce
    /// memory usage and potentially improve generalization.
    /// </para>
    /// </remarks>
    public bool UseParameterGrouping { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of parameter groups.
    /// </summary>
    /// <value>Default: 10.</value>
    /// <remarks>
    /// <para>
    /// Only used when UseParameterGrouping is true. Parameters are partitioned
    /// into this many groups, with each group sharing a single learned learning rate.
    /// </para>
    /// </remarks>
    public int NumParameterGroups { get; set; } = 10;

    /// <summary>
    /// Gets or sets whether to use parameter sharing based on similarity.
    /// </summary>
    /// <value>Default: false.</value>
    /// <remarks>
    /// <para>
    /// When enabled, parameters with similar values or gradients share
    /// learning rate configurations. This can improve sample efficiency.
    /// </para>
    /// </remarks>
    public bool UseParameterSharing { get; set; } = false;

    /// <summary>
    /// Gets or sets the similarity threshold for parameter sharing.
    /// </summary>
    /// <value>Default: 0.95.</value>
    /// <remarks>
    /// <para>
    /// Only used when UseParameterSharing is true. Parameters with similarity
    /// above this threshold will share learning rate configurations.
    /// </para>
    /// </remarks>
    public double ParameterSharingThreshold { get; set; } = 0.95;

    #endregion

    #region Meta-SGD Specific Properties - Adam Configuration

    /// <summary>
    /// Gets or sets whether to learn Adam beta parameters when using Adam update rule.
    /// </summary>
    /// <value>Default: false.</value>
    /// <remarks>
    /// <para>
    /// When enabled with Adam update rule, learns per-parameter beta1 and beta2
    /// coefficients. This adds significant complexity but can improve performance.
    /// </para>
    /// </remarks>
    public bool LearnAdamBetas { get; set; } = false;

    /// <summary>
    /// Gets or sets the initial value for Adam beta1.
    /// </summary>
    /// <value>Default: 0.9 (standard Adam default).</value>
    public double AdamBeta1Init { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the initial value for Adam beta2.
    /// </summary>
    /// <value>Default: 0.999 (standard Adam default).</value>
    public double AdamBeta2Init { get; set; } = 0.999;

    /// <summary>
    /// Gets or sets the epsilon value for Adam numerical stability.
    /// </summary>
    /// <value>Default: 1e-8.</value>
    public double AdamEpsilonInit { get; set; } = 1e-8;

    #endregion

    #region Meta-SGD Specific Properties - Advanced

    /// <summary>
    /// Gets or sets whether to use trust region for parameter updates.
    /// </summary>
    /// <value>Default: false.</value>
    /// <remarks>
    /// <para>
    /// Trust region methods constrain the magnitude of parameter updates,
    /// which can improve stability during meta-training.
    /// </para>
    /// </remarks>
    public bool UseTrustRegion { get; set; } = false;

    /// <summary>
    /// Gets or sets the trust region radius.
    /// </summary>
    /// <value>Default: 1.0.</value>
    /// <remarks>
    /// <para>
    /// Maximum allowed magnitude for parameter updates. Only used when
    /// UseTrustRegion is true.
    /// </para>
    /// </remarks>
    public double TrustRegionRadius { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to use Hessian-free approximation for meta-gradients.
    /// </summary>
    /// <value>Default: false.</value>
    /// <remarks>
    /// <para>
    /// Hessian-free methods can provide better meta-gradient estimates at
    /// the cost of additional computation.
    /// </para>
    /// </remarks>
    public bool UseHessianFree { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of samples for curvature approximation.
    /// </summary>
    /// <value>Default: 10.</value>
    /// <remarks>
    /// <para>
    /// Number of random directions used for Hessian-vector product approximation.
    /// Only used when UseHessianFree is true.
    /// </para>
    /// </remarks>
    public int NumCurvatureSamples { get; set; } = 10;

    /// <summary>
    /// Gets or sets whether to use warm-start initialization for the optimizer.
    /// </summary>
    /// <value>Default: true.</value>
    /// <remarks>
    /// <para>
    /// When enabled, initializes per-parameter learning rates and other
    /// coefficients to reasonable default values based on the configuration.
    /// </para>
    /// </remarks>
    public bool UseWarmStart { get; set; } = true;

    #endregion

    #region Meta-SGD Specific Properties - Learning Rate Schedule

    /// <summary>
    /// Gets or sets whether to use a learning rate schedule for meta-learning rates.
    /// </summary>
    /// <value>Default: false.</value>
    /// <remarks>
    /// <para>
    /// When enabled, the meta-learning rate (outer loop) follows a schedule
    /// during training. This can help with convergence.
    /// </para>
    /// </remarks>
    public bool UseLearningRateSchedule { get; set; } = false;

    /// <summary>
    /// Gets or sets the learning rate schedule type.
    /// </summary>
    /// <value>Default: MetaSGDLearningRateScheduleType.CosineAnnealing.</value>
    public MetaSGDLearningRateScheduleType LearningRateSchedule { get; set; } = MetaSGDLearningRateScheduleType.CosineAnnealing;

    /// <summary>
    /// Gets or sets the number of warmup episodes for learning rate schedule.
    /// </summary>
    /// <value>Default: 1000.</value>
    /// <remarks>
    /// <para>
    /// During warmup, the learning rate gradually increases from a small value
    /// to the target value. This can help with training stability.
    /// </para>
    /// </remarks>
    public int ScheduleWarmupEpisodes { get; set; } = 1000;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the MetaSGDOptions class with the required meta-model.
    /// </summary>
    /// <param name="metaModel">The meta-model to be trained (required).</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel is null.</exception>
    /// <example>
    /// <code>
    /// // Create Meta-SGD options with minimal configuration
    /// var options = new MetaSGDOptions&lt;double, Tensor, Tensor&gt;(myNeuralNetwork);
    /// var metaSGD = new MetaSGDAlgorithm&lt;double, Tensor, Tensor&gt;(options);
    ///
    /// // Create Meta-SGD options with custom per-parameter optimizer
    /// var options = new MetaSGDOptions&lt;double, Tensor, Tensor&gt;(myNeuralNetwork)
    /// {
    ///     UpdateRuleType = MetaSGDUpdateRuleType.Adam,
    ///     LearnMomentum = true,
    ///     LearnDirection = true,
    ///     LearnAdamBetas = true,
    ///     UseParameterGrouping = true,
    ///     NumParameterGroups = 20
    /// };
    /// </code>
    /// </example>
    public MetaSGDOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel));
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <summary>
    /// Validates that all Meta-SGD configuration options are properly set.
    /// </summary>
    /// <returns>True if the configuration is valid for Meta-SGD training; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// Validates all required hyperparameters and Meta-SGD-specific settings:
    /// - Standard meta-learning parameters (learning rates, steps, etc.)
    /// - Learning rate bounds and regularization
    /// - Parameter grouping configuration
    /// - Adam beta parameters if using Adam
    /// - Trust region settings
    /// </para>
    /// </remarks>
    public bool IsValid()
    {
        // Basic validation
        if (MetaModel == null)
            return false;

        if (InnerLearningRate <= 0 || OuterLearningRate <= 0)
            return false;

        if (AdaptationSteps <= 0 || InnerSteps <= 0)
            return false;

        if (MetaBatchSize <= 0 || NumMetaIterations <= 0)
            return false;

        if (EvaluationTasks <= 0)
            return false;

        // Learning rate bounds validation
        if (MinLearningRate <= 0 || MinLearningRate >= MaxLearningRate)
            return false;

        if (MaxLearningRate <= MinLearningRate || MaxLearningRate > 10.0)
            return false;

        // Initialization range validation
        if (LearningRateInitRange <= 0 || LearningRateInitRange > 1.0)
            return false;

        // Layer decay validation
        if (UseLayerWiseDecay)
        {
            if (LayerDecayFactor <= 0 || LayerDecayFactor > 1.0)
                return false;
        }

        // Regularization validation
        if (LearningRateL2Reg < 0 || LearningRateL2Reg > 1.0)
            return false;

        // Parameter grouping validation
        if (UseParameterGrouping)
        {
            if (NumParameterGroups <= 0 || NumParameterGroups > 1000)
                return false;
        }

        // Adam parameters validation
        if (LearnAdamBetas)
        {
            if (AdamBeta1Init <= 0 || AdamBeta1Init >= 1.0)
                return false;

            if (AdamBeta2Init <= 0 || AdamBeta2Init >= 1.0)
                return false;

            if (AdamBeta2Init <= AdamBeta1Init)
                return false;
        }

        // Trust region validation
        if (UseTrustRegion)
        {
            if (TrustRegionRadius <= 0)
                return false;
        }

        // Hessian-free validation
        if (UseHessianFree)
        {
            if (NumCurvatureSamples <= 0 || NumCurvatureSamples > 100)
                return false;
        }

        // Parameter sharing validation
        if (UseParameterSharing)
        {
            if (ParameterSharingThreshold < 0 || ParameterSharingThreshold > 1.0)
                return false;
        }

        // Schedule warmup validation
        if (UseLearningRateSchedule)
        {
            if (ScheduleWarmupEpisodes < 0)
                return false;
        }

        return true;
    }

    /// <summary>
    /// Creates a deep copy of the Meta-SGD options.
    /// </summary>
    /// <returns>A new MetaSGDOptions instance with the same configuration values.</returns>
    public IMetaLearnerOptions<T> Clone()
    {
        return new MetaSGDOptions<T, TInput, TOutput>(MetaModel)
        {
            // Base properties
            LossFunction = LossFunction,
            MetaOptimizer = MetaOptimizer,
            InnerOptimizer = InnerOptimizer,
            DataLoader = DataLoader,
            InnerLearningRate = InnerLearningRate,
            OuterLearningRate = OuterLearningRate,
            AdaptationSteps = AdaptationSteps,
            InnerSteps = InnerSteps,
            MetaBatchSize = MetaBatchSize,
            NumMetaIterations = NumMetaIterations,
            UseFirstOrder = UseFirstOrder,
            GradientClipThreshold = GradientClipThreshold,
            RandomSeed = RandomSeed,
            EvaluationTasks = EvaluationTasks,
            EvaluationFrequency = EvaluationFrequency,
            EnableCheckpointing = EnableCheckpointing,
            CheckpointFrequency = CheckpointFrequency,

            // Meta-SGD core properties
            UpdateRuleType = UpdateRuleType,
            LearnLearningRate = LearnLearningRate,
            LearnMomentum = LearnMomentum,
            LearnDirection = LearnDirection,

            // Learning rate configuration
            LearningRateInitialization = LearningRateInitialization,
            LearningRateInitRange = LearningRateInitRange,
            MinLearningRate = MinLearningRate,
            MaxLearningRate = MaxLearningRate,
            LearningRateL2Reg = LearningRateL2Reg,

            // Layer-wise configuration
            UseLayerWiseDecay = UseLayerWiseDecay,
            LayerDecayFactor = LayerDecayFactor,

            // Parameter grouping
            UseParameterGrouping = UseParameterGrouping,
            NumParameterGroups = NumParameterGroups,
            UseParameterSharing = UseParameterSharing,
            ParameterSharingThreshold = ParameterSharingThreshold,

            // Adam configuration
            LearnAdamBetas = LearnAdamBetas,
            AdamBeta1Init = AdamBeta1Init,
            AdamBeta2Init = AdamBeta2Init,
            AdamEpsilonInit = AdamEpsilonInit,

            // Advanced settings
            UseTrustRegion = UseTrustRegion,
            TrustRegionRadius = TrustRegionRadius,
            UseHessianFree = UseHessianFree,
            NumCurvatureSamples = NumCurvatureSamples,
            UseWarmStart = UseWarmStart,

            // Learning rate schedule
            UseLearningRateSchedule = UseLearningRateSchedule,
            LearningRateSchedule = LearningRateSchedule,
            ScheduleWarmupEpisodes = ScheduleWarmupEpisodes
        };
    }

    #endregion

    #region Utility Methods

    /// <summary>
    /// Gets the total number of learnable meta-parameters.
    /// </summary>
    /// <param name="numModelParameters">Number of model parameters.</param>
    /// <returns>Total number of meta-parameters to learn.</returns>
    /// <remarks>
    /// <para>
    /// The total depends on which features are enabled:
    /// - Per-parameter learning rates: +numModelParameters
    /// - Per-parameter momentum: +numModelParameters
    /// - Per-parameter direction: +numModelParameters
    /// - Adam betas (if learning): +2*numModelParameters
    /// </para>
    /// </remarks>
    public int GetTotalMetaParameters(int numModelParameters)
    {
        int effectiveParams = UseParameterGrouping
            ? Math.Min(NumParameterGroups, numModelParameters)
            : numModelParameters;

        int metaParams = 0;

        if (LearnLearningRate)
            metaParams += effectiveParams;

        if (LearnMomentum)
            metaParams += effectiveParams;

        if (LearnDirection)
            metaParams += effectiveParams;

        if (LearnAdamBetas && UpdateRuleType == MetaSGDUpdateRuleType.Adam)
            metaParams += 3 * effectiveParams; // beta1, beta2, epsilon

        return metaParams;
    }

    /// <summary>
    /// Gets the effective number of parameter groups.
    /// </summary>
    /// <param name="numModelParameters">Number of model parameters.</param>
    /// <returns>Number of parameter groups after considering options.</returns>
    public int GetEffectiveParameterGroups(int numModelParameters)
    {
        if (!UseParameterGrouping)
            return numModelParameters;

        return Math.Min(NumParameterGroups, numModelParameters);
    }

    #endregion
}

#region Meta-SGD Enums

/// <summary>
/// Update rule types for Meta-SGD per-parameter optimization.
/// </summary>
/// <remarks>
/// <para>
/// These define the base optimization algorithm that Meta-SGD learns to configure
/// on a per-parameter basis.
/// </para>
/// </remarks>
public enum MetaSGDUpdateRuleType
{
    /// <summary>
    /// Standard Stochastic Gradient Descent with learned per-parameter learning rates.
    /// Update: θ_i = θ_i - α_i × ∇L
    /// </summary>
    SGD,

    /// <summary>
    /// SGD with learned per-parameter momentum coefficients.
    /// Update: v_i = β_i × v_i + α_i × ∇L; θ_i = θ_i - v_i
    /// </summary>
    SGDWithMomentum,

    /// <summary>
    /// Adam optimizer with optionally learned beta parameters per parameter.
    /// </summary>
    Adam,

    /// <summary>
    /// RMSprop optimizer with learned decay rates per parameter.
    /// </summary>
    RMSprop,

    /// <summary>
    /// AdaGrad optimizer with learned accumulation per parameter.
    /// </summary>
    AdaGrad,

    /// <summary>
    /// AdaDelta optimizer with learned decay per parameter.
    /// </summary>
    AdaDelta
}

/// <summary>
/// Learning rate initialization strategies for Meta-SGD.
/// </summary>
/// <remarks>
/// <para>
/// Different initialization strategies can affect how quickly Meta-SGD converges
/// and the quality of the learned per-parameter optimizers.
/// </para>
/// </remarks>
public enum MetaSGDLearningRateInitialization
{
    /// <summary>
    /// Initialize all learning rates to the same value (InnerLearningRate).
    /// </summary>
    Uniform,

    /// <summary>
    /// Initialize learning rates uniformly at random within LearningRateInitRange.
    /// </summary>
    Random,

    /// <summary>
    /// Initialize learning rates based on parameter magnitudes.
    /// Larger parameters get smaller learning rates.
    /// </summary>
    MagnitudeBased,

    /// <summary>
    /// Initialize learning rates based on layer depth.
    /// Deeper layers get smaller learning rates.
    /// </summary>
    LayerBased,

    /// <summary>
    /// Initialize learning rates using Xavier-style initialization.
    /// Based on fan-in and fan-out of each layer.
    /// </summary>
    Xavier
}

/// <summary>
/// Learning rate schedule types for Meta-SGD meta-training.
/// </summary>
/// <remarks>
/// <para>
/// These schedules control how the meta-learning rate (outer loop) changes
/// during meta-training.
/// </para>
/// </remarks>
public enum MetaSGDLearningRateScheduleType
{
    /// <summary>
    /// No scheduling, constant learning rate throughout training.
    /// </summary>
    Constant,

    /// <summary>
    /// Step decay at specified intervals (e.g., halve every N episodes).
    /// </summary>
    StepDecay,

    /// <summary>
    /// Exponential decay over time.
    /// </summary>
    Exponential,

    /// <summary>
    /// Cosine annealing schedule for smooth decay.
    /// </summary>
    CosineAnnealing,

    /// <summary>
    /// Cyclical learning rates that oscillate between bounds.
    /// </summary>
    Cyclical
}

#endregion
