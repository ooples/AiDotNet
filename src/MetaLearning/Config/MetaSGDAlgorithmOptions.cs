using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Config;

/// <summary>
/// Configuration options for Meta-SGD algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// Meta-SGD extends MAML by learning not just the model initialization but also
/// per-parameter learning rates and update directions. This effectively learns an
/// optimizer for each parameter, allowing for more sophisticated adaptation strategies.
/// </para>
/// <para><b>For Beginners:</b> Meta-SGD learns how to learn by discovering optimal
/// ways to update each parameter:</para>
///
/// Key concepts:
/// - <b>Per-parameter learning rates:</b> Each parameter gets its own learning rate
/// - <b>Update directions:</b> Custom momentum-like terms for each parameter
/// - <b>Learnable optimizers:</b> The algorithm discovers optimization strategies
/// - <b>Fast adaptation:</b> Few gradient steps needed for new tasks
///
/// <para>
/// <b>Advanced Features:</b>
/// - Second-order optimization with learned parameters
/// - Adaptive update rules (SGD, Adam, RMSprop)
/// - Hierarchical parameter grouping
/// - Regularization of learning rates
/// - Bi-level optimization with learned optimizers
/// </para>
/// </remarks>
public class MetaSGDAlgorithmOptions<T, TInput, TOutput>
    where T : struct, IEquatable<T>, IFormattable
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the update rule type for parameter updates.
    /// </summary>
    /// <value>
    /// The type of optimization rule to use.
    /// Default is SGD.
    /// </value>
    /// <remarks>
    /// <b>Update Rule Types:</b>
    /// - <b>SGD:</b> Standard stochastic gradient descent
    /// - <b>Adam:</b> Adaptive moments (first and second)
    /// - <b>RMSprop:</b> Root mean square propagation
    /// - <b>AdaGrad:</b> Adaptive gradients
    /// </remarks>
    public UpdateRuleType UpdateRuleType { get; set; } = UpdateRuleType.SGD;

    /// <summary>
    /// Gets or sets whether to learn per-parameter learning rates.
    /// </summary>
    /// <value>
    /// If true, learns a learning rate for each parameter.
    /// If false, uses a single learning rate for all parameters.
    /// Default is true.
    /// </value>
    public bool LearnPerParameterLearningRates { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to learn per-parameter update directions.
    /// </summary>
    /// <value>
    /// If true, learns momentum-like terms for each parameter.
    /// Default is true.
    /// </value>
    public bool LearnDirection { get; set; } = true;

    /// <summary>
    /// Gets or sets the initialization strategy for per-parameter learning rates.
    /// </summary>
    /// <value>
    /// How to initialize the learning rates for each parameter.
    /// Default is Uniform.
    /// </value>
    public LearningRateInitialization LearningRateInitialization { get; set; } = LearningRateInitialization.Uniform;

    /// <summary>
    /// Gets or sets the initialization range for learning rates.
    /// </summary>
    /// <value>
    /// Range for random initialization of learning rates.
    /// Used with Random initialization strategy.
    /// Default is 0.1.
    /// </value>
    public double LearningRateInitRange { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the minimum allowed learning rate.
    /// </summary>
    /// <value>
    /// Lower bound for learned learning rates.
    /// Prevents learning rates from becoming too small.
    /// Default is 1e-6.
    /// </value>
    public double MinLearningRate { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the maximum allowed learning rate.
    /// </summary>
    /// <value>
    /// Upper bound for learned learning rates.
    /// Prevents learning rates from becoming too large.
    /// Default is 1.0.
    /// </value>
    public double MaxLearningRate { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to apply layer-wise learning rate decay.
    /// </summary>
    /// <value>
    /// If true, deeper layers get smaller learning rates.
    /// Helps with training stability in deep networks.
    /// Default is false.
    /// </value>
    public bool UseLayerWiseDecay { get; set; } = false;

    /// <summary>
    /// Gets or sets the decay factor per layer depth.
    /// </summary>
    /// <value>
    /// Multiplicative factor for each additional layer depth.
    /// Only used when UseLayerWiseDecay is true.
    /// Default is 0.9.
    /// </value>
    public double LayerDecayFactor { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the learning rate regularization coefficient.
    /// </summary>
    /// <value>
    /// L2 regularization strength for learned learning rates.
    /// Prevents extreme learning rate values.
    /// Default is 0.0.
    /// </value>
    public double LearningRateRegularization { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets whether to use parameter grouping.
    /// </summary>
    /// <value>
    /// If true, groups parameters and learns shared optimizers.
    /// Reduces memory usage for large models.
    /// Default is false.
    /// </value>
    public bool UseParameterGrouping { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of parameter groups.
    /// </summary>
    /// <value>
    /// Number of groups to partition parameters into.
    /// Only used when UseParameterGrouping is true.
    /// Default is 10.
    /// </value>
    public int NumParameterGroups { get; set; } = 10;

    /// <summary>
    /// Gets or sets whether to learn Adam beta parameters.
    /// </summary>
    /// <value>
    /// If true, learns Adam's beta1 and beta2 parameters.
    /// Only used with Adam update rule.
    /// Default is false.
    /// </value>
    public bool LearnAdamBetas { get; set; } = false;

    /// <summary>
    /// Gets or sets the Adam beta1 initialization value.
    /// </summary>
    /// <value>
    /// Initial value for Adam's first moment coefficient.
    /// Only used when LearnAdamBetas is true.
    /// Default is 0.9.
    /// </value>
    public double AdamBeta1Init { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the Adam beta2 initialization value.
    /// </summary>
    /// <value>
    /// Initial value for Adam's second moment coefficient.
    /// Only used when LearnAdamBetas is true.
    /// Default is 0.999.
    /// </value>
    public double AdamBeta2Init { get; set; } = 0.999;

    /// <summary>
    /// Gets or sets whether to use trust region for updates.
    /// </summary>
    /// <value>
    /// If true, constrains parameter updates to a trust region.
    /// Helps prevent destabilizing large updates.
    /// Default is false.
    /// </value>
    public bool UseTrustRegion { get; set; } = false;

    /// <summary>
    /// Gets or sets the trust region radius.
    /// </summary>
    /// <value>
    /// Maximum allowed parameter update magnitude.
    /// Only used when UseTrustRegion is true.
    /// Default is 1.0.
    /// </value>
    public double TrustRegionRadius { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to use Hessian-free approximation.
    /// </summary>
    /// <value>
    /// If true, approximates Hessian-vector products.
    /// Reduces computational cost for large models.
    /// Default is false.
    /// </value>
    public bool UseHessianFree { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of curvature samples.
    /// </summary>
    /// <value>
    /// Number of random directions for curvature approximation.
    /// Only used when UseHessianFree is true.
    /// Default is 10.
    /// </value>
    public int NumCurvatureSamples { get; set; } = 10;

    /// <summary>
    /// Gets or sets whether to use parameter sharing.
    /// </summary>
    /// <value>
    /// If true, shares learning rates across similar parameters.
    /// Can improve sample efficiency.
    /// Default is false.
    /// </value>
    public bool UseParameterSharing { get; set; } = false;

    /// <summary>
    /// Gets or sets the parameter sharing threshold.
    /// </summary>
    /// <value>
    /// Similarity threshold for sharing learning rates.
    /// Only used when UseParameterSharing is true.
    /// Default is 0.95.
    /// </value>
    public double ParameterSharingThreshold { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets whether to schedule learning rates.
    /// </summary>
    /// <value>
    /// If true, uses a schedule for meta-learning rates.
    /// Can help with convergence stability.
    /// Default is false.
    /// </value>
    public bool UseLearningRateSchedule { get; set; } = false;

    /// <summary>
    /// Gets or sets the learning rate schedule type.
    /// </summary>
    /// <value>
    /// Type of schedule for meta-learning rates.
    /// Only used when UseLearningRateSchedule is true.
    /// Default is CosineAnnealing.
    /// </value>
    public LearningRateScheduleType LearningRateSchedule { get; set; } = LearningRateScheduleType.CosineAnnealing;

    /// <summary>
    /// Gets or sets the schedule warmup episodes.
    /// </summary>
    /// <value>
    /// Number of episodes for learning rate warmup.
    /// Only used when UseLearningRateSchedule is true.
    /// Default is 1000.
    /// </value>
    public int ScheduleWarmupEpisodes { get; set; } = 1000;

    /// <summary>
    /// Creates a default Meta-SGD configuration with standard values.
    /// </summary>
    /// <remarks>
    /// Default configuration based on Meta-SGD paper:
    /// - Update rule: SGD
    /// - Per-parameter learning rates: True
    /// - Update directions: True
    /// - Learning rate bounds: [1e-6, 1.0]
    /// </remarks>
    public MetaSGDAlgorithmOptions()
    {
        // Set default values
        InnerLearningRate = NumOps.FromDouble(0.01);
        AdaptationSteps = 5;
    }

    /// <summary>
    /// Creates a Meta-SGD configuration with custom values.
    /// </summary>
    /// <param name="updateRuleType">Type of optimization rule.</param>
    /// <param name="learnPerParameterLearningRates">Whether to learn per-parameter learning rates.</param>
    /// <param name="learnDirection">Whether to learn update directions.</param>
    /// <param name="learningRateInitRange">Initialization range for learning rates.</param>
    /// <param name="useLayerWiseDecay">Whether to use layer-wise decay.</param>
    /// <param name="layerDecayFactor">Decay factor per layer.</param>
    /// <param name="learningRateRegularization">Regularization for learning rates.</param>
    /// <param name="useTrustRegion">Whether to use trust region.</param>
    /// <param name="trustRegionRadius">Trust region radius.</param>
    /// <param name="innerLearningRate">Inner loop learning rate.</param>
    /// <param name="adaptationSteps">Number of inner adaptation steps.</param>
    /// <param name="numEpisodes">Number of training episodes.</param>
    public MetaSGDAlgorithmOptions(
        UpdateRuleType updateRuleType = UpdateRuleType.SGD,
        bool learnPerParameterLearningRates = true,
        bool learnDirection = true,
        double learningRateInitRange = 0.1,
        bool useLayerWiseDecay = false,
        double layerDecayFactor = 0.9,
        double learningRateRegularization = 0.0,
        bool useTrustRegion = false,
        double trustRegionRadius = 1.0,
        double innerLearningRate = 0.01,
        int adaptationSteps = 5,
        int numEpisodes = 10000)
    {
        UpdateRuleType = updateRuleType;
        LearnPerParameterLearningRates = learnPerParameterLearningRates;
        LearnDirection = learnDirection;
        LearningRateInitRange = learningRateInitRange;
        UseLayerWiseDecay = useLayerWiseDecay;
        LayerDecayFactor = layerDecayFactor;
        LearningRateRegularization = learningRateRegularization;
        UseTrustRegion = useTrustRegion;
        TrustRegionRadius = trustRegionRadius;
        InnerLearningRate = NumOps.FromDouble(innerLearningRate);
        AdaptationSteps = adaptationSteps;
        NumEpisodes = numEpisodes;
    }

    /// <summary>
    /// Validates the configuration parameters.
    /// </summary>
    /// <returns>True if all parameters are valid, false otherwise.</returns>
    public virtual bool IsValid()
    {
        // Check base class validation
            return false;

        // Check learning rate bounds
        if (MinLearningRate <= 0.0 || MinLearningRate >= MaxLearningRate)
            return false;

        if (MaxLearningRate <= MinLearningRate || MaxLearningRate > 10.0)
            return false;

        // Check initialization range
        if (LearningRateInitRange <= 0.0 || LearningRateInitRange > 1.0)
            return false;

        // Check layer decay factor
        if (UseLayerWiseDecay)
        {
            if (LayerDecayFactor <= 0.0 || LayerDecayFactor > 1.0)
                return false;
        }

        // Check regularization
        if (LearningRateRegularization < 0.0 || LearningRateRegularization > 1.0)
            return false;

        // Check parameter grouping
        if (UseParameterGrouping)
        {
            if (NumParameterGroups <= 0 || NumParameterGroups > 1000)
                return false;
        }

        // Check Adam parameters
        if (LearnAdamBetas)
        {
            if (AdamBeta1Init <= 0.0 || AdamBeta1Init >= 1.0)
                return false;

            if (AdamBeta2Init <= 0.0 || AdamBeta2Init >= 1.0)
                return false;

            if (AdamBeta2Init <= AdamBeta1Init)
                return false;
        }

        // Check trust region
        if (UseTrustRegion)
        {
            if (TrustRegionRadius <= 0.0)
                return false;
        }

        // Check curvature samples
        if (UseHessianFree)
        {
            if (NumCurvatureSamples <= 0 || NumCurvatureSamples > 100)
                return false;
        }

        // Check parameter sharing
        if (UseParameterSharing)
        {
            if (ParameterSharingThreshold < 0.0 || ParameterSharingThreshold > 1.0)
                return false;
        }

        // Check schedule warmup
        if (UseLearningRateSchedule)
        {
            if (ScheduleWarmupEpisodes < 0)
                return false;
        }

        return true;
    }

    /// <summary>
    /// Gets the total number of learnable meta-parameters.
    /// </summary>
    /// <param name="numParameters">Number of model parameters.</param>
    /// <returns>Total number of meta-parameters to learn.</returns>
    public int GetTotalMetaParameters(int numParameters)
    {
        int metaParams = numParameters; // Base parameters

        if (LearnPerParameterLearningRates)
        {
            metaParams += numParameters; // Learning rates per parameter
        }

        if (LearnDirection)
        {
            metaParams += numParameters; // Direction vectors per parameter
        }

        if (LearnAdamBetas)
        {
            metaParams += 2; // beta1 and beta2
        }

        return metaParams;
    }

    /// <summary>
    /// Gets the effective number of parameter groups.
    /// </summary>
    /// <param name="numParameters">Number of model parameters.</param>
    /// <returns>Number of parameter groups after considering options.</returns>
    public int GetEffectiveParameterGroups(int numParameters)
    {
        if (!UseParameterGrouping)
        {
            return numParameters;
        }

        return Math.Min(NumParameterGroups, numParameters);
    }
}

/// <summary>
/// Types of update rules for Meta-SGD optimization.
/// </summary>
public enum UpdateRuleType
{
    /// <summary>
    /// Standard stochastic gradient descent.
    /// </summary>
    SGD,

    /// <summary>
    /// Adaptive moments (Adam optimizer).
    /// </summary>
    Adam,

    /// <summary>
    /// Root mean square propagation.
    /// </summary>
    RMSprop,

    /// <summary>
    /// Adaptive gradients.
    /// </summary>
    AdaGrad
}

/// <summary>
/// Learning rate initialization strategies.
/// </summary>
public enum LearningRateInitialization
{
    /// <summary>
    /// Initialize all learning rates to the same value.
    /// </summary>
    Uniform,

    /// <summary>
    /// Initialize learning rates uniformly at random.
    /// </summary>
    Random,

    /// <summary>
    /// Initialize learning rates based on parameter magnitude.
    /// </summary>
    MagnitudeBased,

    /// <summary>
    /// Initialize learning rates based on layer depth.
    /// </summary>
    LayerBased,

    /// <summary>
    /// Initialize learning rates using Xavier initialization.
    /// </summary>
    Xavier
}

/// <summary>
/// Learning rate schedule types.
/// </summary>
public enum LearningRateScheduleType
{
    /// <summary>
    /// No scheduling, constant learning rate.
    /// </summary>
    Constant,

    /// <summary>
    /// Step decay at specified intervals.
    /// </summary>
    StepDecay,

    /// <summary>
    /// Exponential decay.
    /// </summary>
    Exponential,

    /// <summary>
    /// Cosine annealing schedule.
    /// </summary>
    CosineAnnealing,

    /// <summary>
    /// Cyclical learning rates.
    /// </summary>
    Cyclical
}