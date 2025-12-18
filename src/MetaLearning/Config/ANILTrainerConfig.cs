using AiDotNet.Interfaces;

namespace AiDotNet.MetaLearning.Config;

/// <summary>
/// Configuration for the ANIL (Almost No Inner Loop) meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// ANIL (Almost No Inner Loop) is a computationally efficient variant of MAML
/// that achieves competitive performance by freezing most network parameters
/// and only adapting the final layers during the inner loop.
/// </para>
/// <para><b>For Beginners:</b> This configuration controls ANIL's behavior:
///
/// - <b>FrozenLayerRatio:</b> What percentage of parameters to freeze (typical: 0.8)
/// - <b>InnerLearningRate:</b> Learning rate for the adaptable head (typical: 0.01)
/// - <b>MetaLearningRate:</b> Learning rate for frozen features (typical: 0.001)
/// - <b>InnerSteps:</b> How many gradient steps per task (typical: 5)
/// - <b>MetaBatchSize:</b> Tasks processed per meta-update (typical: 4-16)
///
/// ANIL vs MAML:
/// - ANIL freezes 80% of parameters (big speedup)
/// - MAML updates all parameters (slower but more flexible)
/// - Both achieve similar performance in practice
/// - ANIL uses ~10% of MAML's computation
/// </para>
/// </remarks>
public class ANILTrainerConfig<T> : IMetaLearnerConfig<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc/>
    public T InnerLearningRate { get; set; } = NumOps.FromDouble(0.01);

    /// <inheritdoc/>
    public T MetaLearningRate { get; set; } = NumOps.FromDouble(0.001);

    /// <inheritdoc/>
    public int InnerSteps { get; set; } = 5;

    /// <inheritdoc/>
    public int MetaBatchSize { get; set; } = 4;

    /// <inheritdoc/>
    public int NumMetaIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the fraction of parameters to freeze during adaptation.
    /// </summary>
    /// <value>
    /// Ratio of frozen parameters (0.0 to 1.0). Default is 0.8 (freeze 80%).
    /// Only the remaining (1 - frozenRatio) parameters are adapted.
    /// </value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This controls how much of the network is frozen:
    ///
    /// Frozen ratios and their effects:
    /// - 0.9 (90% frozen): Fastest, but less flexible
    /// - 0.8 (80% frozen): Best speed/performance trade-off (recommended)
    /// - 0.7 (70% frozen): More flexible, slightly slower
    /// - 0.5 (50% frozen): Much slower, approaching MAML
    /// - 0.0 (0% frozen): Full MAML (no freezing)
    ///
    /// Recommended values:
    /// - Large networks: 0.85-0.9 (most parameters frozen)
    /// - Medium networks: 0.75-0.85 (balance)
    /// - Small networks: 0.5-0.75 (more adaptivity)
    /// </para>
    /// </remarks>
    public double FrozenLayerRatio { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets whether to use progressive unfreezing during training.
    /// </summary>
    /// <value>
    /// If true, gradually unfreeze more parameters as training progresses.
    /// Default is false for simplicity.
    /// </value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Progressive unfreezing lets the model learn
    /// more over time without changing the base architecture:
    ///
    /// How it works:
    /// 1. Start with many frozen parameters (fast learning)
    /// 2. As training progresses, gradually unfreeze more
    /// 3. End with fewer frozen parameters (full flexibility)
    ///
    /// This combines the speed benefits of freezing with the flexibility
    /// of full adaptation, adapting the model's capacity as needed.
    /// </para>
    /// </remarks>
    public bool UseProgressiveUnfreezing { get; set; } = false;

    /// <summary>
    /// Gets or sets the unfreezing schedule for progressive training.
    /// </summary>
    /// <value>
    /// Array of frozen ratios to use at different training stages.
    /// Default is [0.9, 0.8, 0.7, 0.6] (progressively unfreeze).
    /// </value>
    /// <remarks>
    /// Each element represents the frozen ratio at a training stage.
    /// The model transitions to the next ratio after reaching certain milestones.
    /// </remarks>
    public double[] UnfreezingSchedule { get; set; } = { 0.9, 0.8, 0.7, 0.6 };

    /// <summary>
    /// Gets or sets the iterations between unfreezing stages.
    /// </summary>
    /// <value>
    /// Number of meta-training iterations before reducing frozen ratio.
    /// Default is 250 iterations per stage.
    /// </value>
    public int UnfreezingInterval { get; set; } = 250;

    /// <summary>
    /// Gets or sets whether to use layer-aware freezing instead of simple ratio.
    /// </summary>
    /// <value>
    /// If true, freeze based on network architecture (layers instead of flat ratio).
    /// If false, use flat frozen ratio across all parameters.
    /// Default is false (simpler flat freezing).
    /// </value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Layer-aware freezing respects network structure:
    ///
    /// Flat freezing (simple):
    /// - Treats all parameters equally
    /// - Freezes first X% of parameters
    - Easy to implement and understand
    ///
    /// Layer-aware freezing (advanced):
    /// - Freezes entire layers together
    /// - Keeps layer structure intact
    - More principled approach
    ///
    /// Example with layer-aware:
    /// - Freeze: Conv1, Conv2, Conv3 (early layers)
    /// - Adapt: Conv4, Conv5, FC1 (later layers)
    /// - Preserves hierarchical feature learning
    /// </para>
    /// </remarks>
    public bool UseLayerAwareFreezing { get; set; } = false;

    /// <summary>
    /// Gets or sets which layers to freeze when using layer-aware freezing.
    /// </summary>
    /// <value>
    /// Array of layer indices to freeze (0-indexed).
    /// If empty, automatically freezes based on layer analysis.
    /// Default is empty (automatic detection).
    /// </value>
    public int[] FrozenLayerIndices { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Creates a default ANIL configuration with standard values.
    /// </summary>
    /// <remarks>
    /// Default values based on the ANIL paper (Raghu et al., 2020):
    /// - Inner learning rate: 0.01 (head adaptation rate)
    /// - Meta learning rate: 0.001 (feature update rate)
    /// - Inner steps: 5 (balance between adaptation and speed)
    /// - Meta batch size: 4 (good balance for stability)
    /// - Frozen layer ratio: 0.8 (80% frozen, 20% adaptable)
    /// - Num meta iterations: 1000 (standard training duration)
    /// </remarks>
    public ANILTrainerConfig()
    {
    }

    /// <summary>
    /// Creates an ANIL configuration with custom values.
    /// </summary>
    /// <param name="innerLearningRate">Learning rate for head adaptation.</param>
    /// <param name="metaLearningRate">Learning rate for meta-parameters.</param>
    /// <param name="innerSteps">Number of gradient steps per task.</param>
    /// <param name="metaBatchSize">Number of tasks per meta-update.</param>
    /// <param name="numMetaIterations">Total number of meta-training iterations.</param>
    /// <param name="frozenLayerRatio">Fraction of parameters to freeze (0-1).</param>
    /// <param name="useProgressiveUnfreezing">Whether to gradually unfreeze parameters.</param>
    /// <param name="unfreezingSchedule">Schedule for progressive unfreezing.</param>
    /// <param name="unfreezingInterval">Iterations between unfreezing stages.</param>
    /// <param name="useLayerAwareFreezing">Whether to freeze by layers instead of ratio.</param>
    /// <param name="frozenLayerIndices">Specific layer indices to freeze.</param>
    public ANILTrainerConfig(
        double innerLearningRate,
        double metaLearningRate,
        int innerSteps,
        int metaBatchSize = 4,
        int numMetaIterations = 1000,
        double frozenLayerRatio = 0.8,
        bool useProgressiveUnfreezing = false,
        double[]? unfreezingSchedule = null,
        int unfreezingInterval = 250,
        bool useLayerAwareFreezing = false,
        int[]? frozenLayerIndices = null)
    {
        InnerLearningRate = NumOps.FromDouble(innerLearningRate);
        MetaLearningRate = NumOps.FromDouble(metaLearningRate);
        InnerSteps = innerSteps;
        MetaBatchSize = metaBatchSize;
        NumMetaIterations = numMetaIterations;
        FrozenLayerRatio = frozenLayerRatio;
        UseProgressiveUnfreezing = useProgressiveUnfreezing;
        UnfreezingSchedule = unfreezingSchedule ?? new double[] { 0.9, 0.8, 0.7, 0.6 };
        UnfreezingInterval = unfreezingInterval;
        UseLayerAwareFreezing = useLayerAwareFreezing;
        FrozenLayerIndices = frozenLayerIndices ?? Array.Empty<int>();
    }

    /// <inheritdoc/>
    public bool IsValid()
    {
        var innerLr = Convert.ToDouble(InnerLearningRate);
        var metaLr = Convert.ToDouble(MetaLearningRate);

        return innerLr > 0 && innerLr <= 1.0 &&
               metaLr > 0 && metaLr <= 1.0 &&
               InnerSteps > 0 && InnerSteps <= 100 &&
               MetaBatchSize > 0 && MetaBatchSize <= 128 &&
               NumMetaIterations > 0 && NumMetaIterations <= 1000000 &&
               FrozenLayerRatio >= 0.0 && FrozenLayerRatio <= 1.0 &&
               FrozenLayerRatio < 1.0; // Must freeze at least one parameter
    }
}