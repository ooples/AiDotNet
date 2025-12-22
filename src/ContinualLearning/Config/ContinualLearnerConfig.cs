using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.Helpers;

namespace AiDotNet.ContinualLearning.Config;

/// <summary>
/// Production-ready configuration for continual learning algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This class provides all settings needed for continual learning.
/// All properties have industry-standard defaults set in the constructor.</para>
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
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    #region Core Training Parameters

    /// <inheritdoc/>
    public T LearningRate { get; set; }

    /// <inheritdoc/>
    public int EpochsPerTask { get; set; }

    /// <inheritdoc/>
    public int BatchSize { get; set; }

    #endregion

    #region Memory Parameters

    /// <inheritdoc/>
    public int MemorySize { get; set; }

    /// <inheritdoc/>
    public int SamplesPerTask { get; set; }

    /// <inheritdoc/>
    public MemorySamplingStrategy MemoryStrategy { get; set; }

    /// <inheritdoc/>
    public bool UsePrioritizedReplay { get; set; }

    #endregion

    #region EWC-Specific Parameters

    /// <inheritdoc/>
    public T EwcLambda { get; set; }

    /// <inheritdoc/>
    public int FisherSamples { get; set; }

    /// <inheritdoc/>
    public bool UseEmpiricalFisher { get; set; }

    /// <inheritdoc/>
    public bool NormalizeFisher { get; set; }

    #endregion

    #region Online-EWC Parameters

    /// <inheritdoc/>
    public T OnlineEwcGamma { get; set; }

    #endregion

    #region LwF-Specific Parameters

    /// <inheritdoc/>
    public T DistillationTemperature { get; set; }

    /// <inheritdoc/>
    public T DistillationWeight { get; set; }

    /// <inheritdoc/>
    public bool UseSoftTargets { get; set; }

    #endregion

    #region GEM-Specific Parameters

    /// <inheritdoc/>
    public T GemMemoryStrength { get; set; }

    /// <inheritdoc/>
    public T AGemMargin { get; set; }

    /// <inheritdoc/>
    public int AGemReferenceGradients { get; set; }

    #endregion

    #region SI-Specific Parameters

    /// <inheritdoc/>
    public T SiC { get; set; }

    /// <inheritdoc/>
    public T SiXi { get; set; }

    #endregion

    #region MAS-Specific Parameters

    /// <inheritdoc/>
    public T MasLambda { get; set; }

    #endregion

    #region PackNet-Specific Parameters

    /// <inheritdoc/>
    public T PackNetPruneRatio { get; set; }

    /// <inheritdoc/>
    public int PackNetRetrainEpochs { get; set; }

    #endregion

    #region Progressive Neural Networks Parameters

    /// <inheritdoc/>
    public bool PnnUseLateralConnections { get; set; }

    /// <inheritdoc/>
    public T PnnLateralScaling { get; set; }

    #endregion

    #region iCaRL-Specific Parameters

    /// <inheritdoc/>
    public int ICarlExemplarsPerClass { get; set; }

    /// <inheritdoc/>
    public bool ICarlUseHerding { get; set; }

    #endregion

    #region BiC-Specific Parameters

    /// <inheritdoc/>
    public T BiCValidationFraction { get; set; }

    #endregion

    #region HAT-Specific Parameters

    /// <inheritdoc/>
    public T HatSparsity { get; set; }

    /// <inheritdoc/>
    public T HatSmax { get; set; }

    #endregion

    #region Evaluation Parameters

    /// <inheritdoc/>
    public bool ComputeBackwardTransfer { get; set; }

    /// <inheritdoc/>
    public bool ComputeForwardTransfer { get; set; }

    /// <inheritdoc/>
    public int EvaluationFrequency { get; set; }

    #endregion

    #region Advanced Parameters

    /// <inheritdoc/>
    public int? RandomSeed { get; set; }

    /// <inheritdoc/>
    public int MaxTasks { get; set; }

    /// <inheritdoc/>
    public bool UseGradientClipping { get; set; }

    /// <inheritdoc/>
    public T GradientClipNorm { get; set; }

    /// <inheritdoc/>
    public bool UseWeightDecay { get; set; }

    /// <inheritdoc/>
    public T WeightDecay { get; set; }

    #endregion

    /// <summary>
    /// Initializes a new instance with industry-standard default values.
    /// </summary>
    public ContinualLearnerConfig()
    {
        // Core Training Parameters
        LearningRate = NumOps.FromDouble(0.001);
        EpochsPerTask = 10;
        BatchSize = 32;

        // Memory Parameters
        MemorySize = 1000;
        SamplesPerTask = 100; // Default, can be auto-calculated
        MemoryStrategy = MemorySamplingStrategy.Reservoir;
        UsePrioritizedReplay = false;

        // EWC-Specific Parameters
        EwcLambda = NumOps.FromDouble(1000.0);
        FisherSamples = 200;
        UseEmpiricalFisher = true;
        NormalizeFisher = true;

        // Online-EWC Parameters
        OnlineEwcGamma = NumOps.FromDouble(0.95);

        // LwF-Specific Parameters
        DistillationTemperature = NumOps.FromDouble(2.0);
        DistillationWeight = NumOps.One;
        UseSoftTargets = true;

        // GEM-Specific Parameters
        GemMemoryStrength = NumOps.FromDouble(0.5);
        AGemMargin = NumOps.Zero;
        AGemReferenceGradients = 256;

        // SI-Specific Parameters
        SiC = NumOps.FromDouble(0.1);
        SiXi = NumOps.FromDouble(0.1);

        // MAS-Specific Parameters
        MasLambda = NumOps.One;

        // PackNet-Specific Parameters
        PackNetPruneRatio = NumOps.FromDouble(0.75);
        PackNetRetrainEpochs = 5;

        // Progressive Neural Networks Parameters
        PnnUseLateralConnections = true;
        PnnLateralScaling = NumOps.One;

        // iCaRL-Specific Parameters
        ICarlExemplarsPerClass = 20;
        ICarlUseHerding = true;

        // BiC-Specific Parameters
        BiCValidationFraction = NumOps.FromDouble(0.1);

        // HAT-Specific Parameters
        HatSparsity = NumOps.FromDouble(0.01);
        HatSmax = NumOps.FromDouble(400.0);

        // Evaluation Parameters
        ComputeBackwardTransfer = true;
        ComputeForwardTransfer = true;
        EvaluationFrequency = 1;

        // Advanced Parameters
        RandomSeed = null;
        MaxTasks = 100;
        UseGradientClipping = false;
        GradientClipNorm = NumOps.One;
        UseWeightDecay = false;
        WeightDecay = NumOps.FromDouble(0.0001);
    }

    /// <inheritdoc/>
    public bool IsValid()
    {
        // Validate learning rate
        if (NumOps.Compare(LearningRate, NumOps.Zero) <= 0)
            return false;

        // Validate epochs
        if (EpochsPerTask <= 0)
            return false;

        // Validate batch size
        if (BatchSize <= 0)
            return false;

        // Validate memory size
        if (MemorySize < 0)
            return false;

        // Validate EWC lambda
        if (NumOps.Compare(EwcLambda, NumOps.Zero) < 0)
            return false;

        // Validate Fisher samples
        if (FisherSamples <= 0)
            return false;

        // Validate distillation temperature
        if (NumOps.Compare(DistillationTemperature, NumOps.Zero) <= 0)
            return false;

        // Validate GEM memory strength (0-1)
        if (NumOps.Compare(GemMemoryStrength, NumOps.Zero) < 0 ||
            NumOps.Compare(GemMemoryStrength, NumOps.One) > 0)
            return false;

        // Validate PackNet prune ratio (0-1 exclusive)
        if (NumOps.Compare(PackNetPruneRatio, NumOps.Zero) <= 0 ||
            NumOps.Compare(PackNetPruneRatio, NumOps.One) >= 0)
            return false;

        // Validate BiC validation fraction (0-1 exclusive)
        if (NumOps.Compare(BiCValidationFraction, NumOps.Zero) <= 0 ||
            NumOps.Compare(BiCValidationFraction, NumOps.One) >= 0)
            return false;

        return true;
    }

    /// <summary>
    /// Creates a configuration optimized for EWC strategy.
    /// </summary>
    /// <param name="lambda">EWC regularization strength. Use NumOps.FromDouble() to convert from double.</param>
    /// <param name="fisherSamples">Optional number of samples for Fisher Information computation.</param>
    public static ContinualLearnerConfig<T> ForEwc(T lambda, int? fisherSamples = null)
    {
        var config = new ContinualLearnerConfig<T>
        {
            EwcLambda = lambda
        };
        if (fisherSamples.HasValue)
            config.FisherSamples = fisherSamples.Value;
        return config;
    }

    /// <summary>
    /// Creates a configuration optimized for LwF strategy with default distillation weight.
    /// </summary>
    /// <param name="temperature">Distillation temperature. Use NumOps.FromDouble() to convert from double.</param>
    public static ContinualLearnerConfig<T> ForLwf(T temperature)
    {
        return new ContinualLearnerConfig<T>
        {
            DistillationTemperature = temperature
        };
    }

    /// <summary>
    /// Creates a configuration optimized for LwF strategy.
    /// </summary>
    /// <param name="temperature">Distillation temperature. Use NumOps.FromDouble() to convert from double.</param>
    /// <param name="weight">Distillation weight.</param>
    public static ContinualLearnerConfig<T> ForLwf(T temperature, T weight)
    {
        return new ContinualLearnerConfig<T>
        {
            DistillationTemperature = temperature,
            DistillationWeight = weight
        };
    }

    /// <summary>
    /// Creates a configuration optimized for GEM strategy.
    /// </summary>
    /// <param name="memoryStrength">GEM memory strength. Use NumOps.FromDouble() to convert from double.</param>
    /// <param name="memorySize">Optional memory buffer size.</param>
    public static ContinualLearnerConfig<T> ForGem(T memoryStrength, int? memorySize = null)
    {
        var config = new ContinualLearnerConfig<T>
        {
            GemMemoryStrength = memoryStrength
        };
        if (memorySize.HasValue)
            config.MemorySize = memorySize.Value;
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
            config.MemorySize = memorySize.Value;
        if (strategy.HasValue)
            config.MemoryStrategy = strategy.Value;
        if (prioritized.HasValue)
            config.UsePrioritizedReplay = prioritized.Value;
        return config;
    }
}
