using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ContinualLearning.Config;

/// <summary>
/// Production-ready configuration for continual learning algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This class provides all settings needed for continual learning.
/// All properties are nullable - when null, industry-standard defaults are applied automatically.
/// This "zero-config" approach means you can get started quickly while still having full
/// control when needed.</para>
///
/// <para><b>Usage Example:</b>
/// <code>
/// // Simple usage with defaults
/// var config = new ContinualLearnerConfig&lt;double&gt;();
///
/// // Custom EWC configuration
/// var ewcConfig = new ContinualLearnerConfig&lt;double&gt;
/// {
///     EwcLambda = 5000.0,        // Stronger regularization
///     FisherSamples = 500,       // More samples for Fisher computation
///     MemorySize = 2000          // Larger memory buffer
/// };
///
/// // Custom LwF configuration
/// var lwfConfig = new ContinualLearnerConfig&lt;double&gt;
/// {
///     DistillationTemperature = 4.0,  // Softer distributions
///     DistillationWeight = 2.0        // Prioritize old knowledge
/// };
/// </code>
/// </para>
///
/// <para><b>Industry Standard Defaults:</b>
/// <list type="bullet">
/// <item><description>Learning Rate: 0.001 (Adam optimizer standard)</description></item>
/// <item><description>Batch Size: 32 (balance of speed and gradient noise)</description></item>
/// <item><description>EWC Lambda: 1000 (based on Kirkpatrick et al. 2017)</description></item>
/// <item><description>Distillation Temperature: 2.0 (based on Li and Hoiem 2017)</description></item>
/// </list>
/// </para>
/// </remarks>
public class ContinualLearnerConfig<T> : IContinualLearnerConfig<T>
{
    private static INumericOperations<T>? _numOps;
    private static INumericOperations<T> NumOps => _numOps ??= MathHelper.GetNumericOperations<T>();

    #region Backing Fields for Proper Nullable Semantics
    // Note: For unconstrained generic T, T? does NOT create Nullable<T> for value types.
    // Using object? backing fields allows proper null tracking regardless of T being value or reference type.
    private object? _learningRate;
    private object? _ewcLambda;
    private object? _onlineEwcGamma;
    private object? _distillationTemperature;
    private object? _distillationWeight;
    private object? _gemMemoryStrength;
    private object? _agemMargin;
    private object? _siC;
    private object? _siXi;
    private object? _masLambda;
    private object? _packNetPruneRatio;
    private object? _pnnLateralScaling;
    private object? _biCValidationFraction;
    private object? _hatSparsity;
    private object? _hatSmax;
    private object? _gradientClipNorm;
    private object? _weightDecay;
    #endregion

    #region Core Training Parameters

    /// <inheritdoc/>
    public T? LearningRate
    {
        get => _learningRate is T val ? val : default;
        set => _learningRate = value;
    }

    /// <inheritdoc/>
    public int? EpochsPerTask { get; set; }

    /// <inheritdoc/>
    public int? BatchSize { get; set; }

    #endregion

    #region Memory Parameters

    /// <inheritdoc/>
    public int? MemorySize { get; set; }

    /// <inheritdoc/>
    public int? SamplesPerTask { get; set; }

    /// <inheritdoc/>
    public MemorySamplingStrategy? MemoryStrategy { get; set; }

    /// <inheritdoc/>
    public bool? UsePrioritizedReplay { get; set; }

    #endregion

    #region EWC-Specific Parameters

    /// <inheritdoc/>
    public T? EwcLambda
    {
        get => _ewcLambda is T val ? val : default;
        set => _ewcLambda = value;
    }

    /// <inheritdoc/>
    public int? FisherSamples { get; set; }

    /// <inheritdoc/>
    public bool? UseEmpiricalFisher { get; set; }

    /// <inheritdoc/>
    public bool? NormalizeFisher { get; set; }

    #endregion

    #region Online-EWC Parameters

    /// <inheritdoc/>
    public T? OnlineEwcGamma
    {
        get => _onlineEwcGamma is T val ? val : default;
        set => _onlineEwcGamma = value;
    }

    #endregion

    #region LwF-Specific Parameters

    /// <inheritdoc/>
    public T? DistillationTemperature
    {
        get => _distillationTemperature is T val ? val : default;
        set => _distillationTemperature = value;
    }

    /// <inheritdoc/>
    public T? DistillationWeight
    {
        get => _distillationWeight is T val ? val : default;
        set => _distillationWeight = value;
    }

    /// <inheritdoc/>
    public bool? UseSoftTargets { get; set; }

    #endregion

    #region GEM-Specific Parameters

    /// <inheritdoc/>
    public T? GemMemoryStrength
    {
        get => _gemMemoryStrength is T val ? val : default;
        set => _gemMemoryStrength = value;
    }

    /// <inheritdoc/>
    public T? AGemMargin
    {
        get => _agemMargin is T val ? val : default;
        set => _agemMargin = value;
    }

    /// <inheritdoc/>
    public int? AGemReferenceGradients { get; set; }

    #endregion

    #region SI-Specific Parameters

    /// <inheritdoc/>
    public T? SiC
    {
        get => _siC is T val ? val : default;
        set => _siC = value;
    }

    /// <inheritdoc/>
    public T? SiXi
    {
        get => _siXi is T val ? val : default;
        set => _siXi = value;
    }

    #endregion

    #region MAS-Specific Parameters

    /// <inheritdoc/>
    public T? MasLambda
    {
        get => _masLambda is T val ? val : default;
        set => _masLambda = value;
    }

    #endregion

    #region PackNet-Specific Parameters

    /// <inheritdoc/>
    public T? PackNetPruneRatio
    {
        get => _packNetPruneRatio is T val ? val : default;
        set => _packNetPruneRatio = value;
    }

    /// <inheritdoc/>
    public int? PackNetRetrainEpochs { get; set; }

    #endregion

    #region Progressive Neural Networks Parameters

    /// <inheritdoc/>
    public bool? PnnUseLateralConnections { get; set; }

    /// <inheritdoc/>
    public T? PnnLateralScaling
    {
        get => _pnnLateralScaling is T val ? val : default;
        set => _pnnLateralScaling = value;
    }

    #endregion

    #region iCaRL-Specific Parameters

    /// <inheritdoc/>
    public int? ICarlExemplarsPerClass { get; set; }

    /// <inheritdoc/>
    public bool? ICarlUseHerding { get; set; }

    #endregion

    #region BiC-Specific Parameters

    /// <inheritdoc/>
    public T? BiCValidationFraction
    {
        get => _biCValidationFraction is T val ? val : default;
        set => _biCValidationFraction = value;
    }

    #endregion

    #region HAT-Specific Parameters

    /// <inheritdoc/>
    public T? HatSparsity
    {
        get => _hatSparsity is T val ? val : default;
        set => _hatSparsity = value;
    }

    /// <inheritdoc/>
    public T? HatSmax
    {
        get => _hatSmax is T val ? val : default;
        set => _hatSmax = value;
    }

    #endregion

    #region Evaluation Parameters

    /// <inheritdoc/>
    public bool? ComputeBackwardTransfer { get; set; }

    /// <inheritdoc/>
    public bool? ComputeForwardTransfer { get; set; }

    /// <inheritdoc/>
    public int? EvaluationFrequency { get; set; }

    #endregion

    #region Advanced Parameters

    /// <inheritdoc/>
    public int? RandomSeed { get; set; }

    /// <inheritdoc/>
    public int? MaxTasks { get; set; }

    /// <inheritdoc/>
    public bool? UseGradientClipping { get; set; }

    /// <inheritdoc/>
    public T? GradientClipNorm
    {
        get => _gradientClipNorm is T val ? val : default;
        set => _gradientClipNorm = value;
    }

    /// <inheritdoc/>
    public bool? UseWeightDecay { get; set; }

    /// <inheritdoc/>
    public T? WeightDecay
    {
        get => _weightDecay is T val ? val : default;
        set => _weightDecay = value;
    }

    #endregion

    #region Effective Value Getters (apply defaults when null)

    /// <summary>
    /// Gets the effective learning rate (applies default if null).
    /// </summary>
    public T GetEffectiveLearningRate() =>
        _learningRate is T val ? val : NumOps.FromDouble(0.001);

    /// <summary>
    /// Gets the effective epochs per task (applies default if null).
    /// </summary>
    public int GetEffectiveEpochsPerTask() =>
        EpochsPerTask ?? 10;

    /// <summary>
    /// Gets the effective batch size (applies default if null).
    /// </summary>
    public int GetEffectiveBatchSize() =>
        BatchSize ?? 32;

    /// <summary>
    /// Gets the effective memory size (applies default if null).
    /// </summary>
    public int GetEffectiveMemorySize() =>
        MemorySize ?? 1000;

    /// <summary>
    /// Gets the effective samples per task (applies default if null).
    /// </summary>
    public int GetEffectiveSamplesPerTask(int numTasks) =>
        SamplesPerTask ?? Math.Max(1, GetEffectiveMemorySize() / Math.Max(1, numTasks));

    /// <summary>
    /// Gets the effective memory strategy (applies default if null).
    /// </summary>
    public MemorySamplingStrategy GetEffectiveMemoryStrategy() =>
        MemoryStrategy ?? MemorySamplingStrategy.Reservoir;

    /// <summary>
    /// Gets whether to use prioritized replay (applies default if null).
    /// </summary>
    public bool GetEffectiveUsePrioritizedReplay() =>
        UsePrioritizedReplay ?? false;

    /// <summary>
    /// Gets the effective EWC lambda (applies default if null).
    /// </summary>
    public T GetEffectiveEwcLambda() =>
        _ewcLambda is T val ? val : NumOps.FromDouble(1000.0);

    /// <summary>
    /// Gets the effective Fisher samples (applies default if null).
    /// </summary>
    public int GetEffectiveFisherSamples() =>
        FisherSamples ?? 200;

    /// <summary>
    /// Gets whether to use empirical Fisher (applies default if null).
    /// </summary>
    public bool GetEffectiveUseEmpiricalFisher() =>
        UseEmpiricalFisher ?? true;

    /// <summary>
    /// Gets whether to normalize Fisher (applies default if null).
    /// </summary>
    public bool GetEffectiveNormalizeFisher() =>
        NormalizeFisher ?? true;

    /// <summary>
    /// Gets the effective Online-EWC gamma (applies default if null).
    /// </summary>
    public T GetEffectiveOnlineEwcGamma() =>
        _onlineEwcGamma is T val ? val : NumOps.FromDouble(0.95);

    /// <summary>
    /// Gets the effective distillation temperature (applies default if null).
    /// </summary>
    public T GetEffectiveDistillationTemperature() =>
        _distillationTemperature is T val ? val : NumOps.FromDouble(2.0);

    /// <summary>
    /// Gets the effective distillation weight (applies default if null).
    /// </summary>
    public T GetEffectiveDistillationWeight() =>
        _distillationWeight is T val ? val : NumOps.FromDouble(1.0);

    /// <summary>
    /// Gets whether to use soft targets (applies default if null).
    /// </summary>
    public bool GetEffectiveUseSoftTargets() =>
        UseSoftTargets ?? true;

    /// <summary>
    /// Gets the effective GEM memory strength (applies default if null).
    /// </summary>
    public T GetEffectiveGemMemoryStrength() =>
        _gemMemoryStrength is T val ? val : NumOps.FromDouble(0.5);

    /// <summary>
    /// Gets the effective A-GEM margin (applies default if null).
    /// </summary>
    public T GetEffectiveAGemMargin() =>
        _agemMargin is T val ? val : NumOps.FromDouble(0.0);

    /// <summary>
    /// Gets the effective A-GEM reference gradients (applies default if null).
    /// </summary>
    public int GetEffectiveAGemReferenceGradients() =>
        AGemReferenceGradients ?? 256;

    /// <summary>
    /// Gets the effective SI c parameter (applies default if null).
    /// </summary>
    public T GetEffectiveSiC() =>
        _siC is T val ? val : NumOps.FromDouble(0.1);

    /// <summary>
    /// Gets the effective SI xi parameter (applies default if null).
    /// </summary>
    public T GetEffectiveSiXi() =>
        _siXi is T val ? val : NumOps.FromDouble(0.1);

    /// <summary>
    /// Gets the effective MAS lambda (applies default if null).
    /// </summary>
    public T GetEffectiveMasLambda() =>
        _masLambda is T val ? val : NumOps.FromDouble(1.0);

    /// <summary>
    /// Gets the effective PackNet prune ratio (applies default if null).
    /// </summary>
    public T GetEffectivePackNetPruneRatio() =>
        _packNetPruneRatio is T val ? val : NumOps.FromDouble(0.75);

    /// <summary>
    /// Gets the effective PackNet retrain epochs (applies default if null).
    /// </summary>
    public int GetEffectivePackNetRetrainEpochs() =>
        PackNetRetrainEpochs ?? 5;

    /// <summary>
    /// Gets whether to use PNN lateral connections (applies default if null).
    /// </summary>
    public bool GetEffectivePnnUseLateralConnections() =>
        PnnUseLateralConnections ?? true;

    /// <summary>
    /// Gets the effective PNN lateral scaling (applies default if null).
    /// </summary>
    public T GetEffectivePnnLateralScaling() =>
        _pnnLateralScaling is T val ? val : NumOps.FromDouble(1.0);

    /// <summary>
    /// Gets the effective iCaRL exemplars per class (applies default if null).
    /// </summary>
    public int GetEffectiveICarlExemplarsPerClass() =>
        ICarlExemplarsPerClass ?? 20;

    /// <summary>
    /// Gets whether to use iCaRL herding (applies default if null).
    /// </summary>
    public bool GetEffectiveICarlUseHerding() =>
        ICarlUseHerding ?? true;

    /// <summary>
    /// Gets the effective BiC validation fraction (applies default if null).
    /// </summary>
    public T GetEffectiveBiCValidationFraction() =>
        _biCValidationFraction is T val ? val : NumOps.FromDouble(0.1);

    /// <summary>
    /// Gets the effective HAT sparsity (applies default if null).
    /// </summary>
    public T GetEffectiveHatSparsity() =>
        _hatSparsity is T val ? val : NumOps.FromDouble(0.01);

    /// <summary>
    /// Gets the effective HAT smax (applies default if null).
    /// </summary>
    public T GetEffectiveHatSmax() =>
        _hatSmax is T val ? val : NumOps.FromDouble(400.0);

    /// <summary>
    /// Gets whether to compute backward transfer (applies default if null).
    /// </summary>
    public bool GetEffectiveComputeBackwardTransfer() =>
        ComputeBackwardTransfer ?? true;

    /// <summary>
    /// Gets whether to compute forward transfer (applies default if null).
    /// </summary>
    public bool GetEffectiveComputeForwardTransfer() =>
        ComputeForwardTransfer ?? true;

    /// <summary>
    /// Gets the effective evaluation frequency (applies default if null).
    /// </summary>
    public int GetEffectiveEvaluationFrequency() =>
        EvaluationFrequency ?? 1;

    /// <summary>
    /// Gets the effective max tasks (applies default if null).
    /// </summary>
    public int GetEffectiveMaxTasks() =>
        MaxTasks ?? 100;

    /// <summary>
    /// Gets whether to use gradient clipping (applies default if null).
    /// </summary>
    public bool GetEffectiveUseGradientClipping() =>
        UseGradientClipping ?? false;

    /// <summary>
    /// Gets the effective gradient clip norm (applies default if null).
    /// </summary>
    public T GetEffectiveGradientClipNorm() =>
        _gradientClipNorm is T val ? val : NumOps.FromDouble(1.0);

    /// <summary>
    /// Gets whether to use weight decay (applies default if null).
    /// </summary>
    public bool GetEffectiveUseWeightDecay() =>
        UseWeightDecay ?? false;

    /// <summary>
    /// Gets the effective weight decay (applies default if null).
    /// </summary>
    public T GetEffectiveWeightDecay() =>
        _weightDecay is T val ? val : NumOps.FromDouble(0.0001);

    #endregion

    /// <inheritdoc/>
    public bool IsValid()
    {
        // Validate learning rate if specified (check backing field for proper null semantics)
        if (_learningRate != null && Convert.ToDouble(_learningRate) <= 0)
            return false;

        // Validate epochs if specified
        if (EpochsPerTask.HasValue && EpochsPerTask.Value <= 0)
            return false;

        // Validate batch size if specified
        if (BatchSize.HasValue && BatchSize.Value <= 0)
            return false;

        // Validate memory size if specified
        if (MemorySize.HasValue && MemorySize.Value < 0)
            return false;

        // Validate EWC lambda if specified (check backing field for proper null semantics)
        if (_ewcLambda != null && Convert.ToDouble(_ewcLambda) < 0)
            return false;

        // Validate Fisher samples if specified
        if (FisherSamples.HasValue && FisherSamples.Value <= 0)
            return false;

        // Validate distillation temperature if specified (check backing field for proper null semantics)
        if (_distillationTemperature != null && Convert.ToDouble(_distillationTemperature) <= 0)
            return false;

        // Validate GEM memory strength if specified (check backing field for proper null semantics)
        if (_gemMemoryStrength != null)
        {
            var gemStrength = Convert.ToDouble(_gemMemoryStrength);
            if (gemStrength < 0 || gemStrength > 1)
                return false;
        }

        // Validate PackNet prune ratio if specified (check backing field for proper null semantics)
        if (_packNetPruneRatio != null)
        {
            var pruneRatio = Convert.ToDouble(_packNetPruneRatio);
            if (pruneRatio <= 0 || pruneRatio >= 1)
                return false;
        }

        // Validate BiC validation fraction if specified (check backing field for proper null semantics)
        if (_biCValidationFraction != null)
        {
            var valFraction = Convert.ToDouble(_biCValidationFraction);
            if (valFraction <= 0 || valFraction >= 1)
                return false;
        }

        return true;
    }

    /// <summary>
    /// Creates a configuration optimized for EWC strategy.
    /// </summary>
    public static ContinualLearnerConfig<T> ForEwc(T? lambda = default, int? fisherSamples = null)
    {
        var config = new ContinualLearnerConfig<T>();
        if (lambda != null)
            config.EwcLambda = lambda;
        if (fisherSamples.HasValue)
            config.FisherSamples = fisherSamples;
        return config;
    }

    /// <summary>
    /// Creates a configuration optimized for LwF strategy.
    /// </summary>
    public static ContinualLearnerConfig<T> ForLwf(T? temperature = default, T? weight = default)
    {
        var config = new ContinualLearnerConfig<T>();
        if (temperature != null)
            config.DistillationTemperature = temperature;
        if (weight != null)
            config.DistillationWeight = weight;
        return config;
    }

    /// <summary>
    /// Creates a configuration optimized for GEM strategy.
    /// </summary>
    public static ContinualLearnerConfig<T> ForGem(T? memoryStrength = default, int? memorySize = null)
    {
        var config = new ContinualLearnerConfig<T>();
        if (memoryStrength != null)
            config.GemMemoryStrength = memoryStrength;
        if (memorySize.HasValue)
            config.MemorySize = memorySize;
        return config;
    }

    /// <summary>
    /// Creates a configuration optimized for experience replay.
    /// </summary>
    public static ContinualLearnerConfig<T> ForExperienceReplay(
        int? memorySize = null,
        MemorySamplingStrategy? strategy = null,
        bool? prioritized = null)
    {
        var config = new ContinualLearnerConfig<T>();
        if (memorySize.HasValue)
            config.MemorySize = memorySize;
        if (strategy.HasValue)
            config.MemoryStrategy = strategy;
        if (prioritized.HasValue)
            config.UsePrioritizedReplay = prioritized;
        return config;
    }
}
