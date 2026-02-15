using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Uncertainty estimation method for FewTURE.
/// </summary>
public enum FewTUREUncertaintyMethod
{
    /// <summary>Prediction entropy: H(p) = -sum(p_i * log(p_i))</summary>
    Entropy,
    /// <summary>Prediction variance across token-level features.</summary>
    Variance,
    /// <summary>Monte Carlo dropout-based uncertainty.</summary>
    MCDropout
}

/// <summary>
/// Configuration options for FewTURE (Few-shot Transformer with Uncertainty and Reliable Estimation) (Hiller et al., ECCV 2022).
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// FewTURE uses vision transformers with token-level local features and uncertainty estimation
/// for few-shot classification. It operates on patch tokens from a ViT backbone.
/// </para>
/// <para><b>For Beginners:</b> FewTURE combines transformers with uncertainty:
/// 1. Uses a Vision Transformer (ViT) to get patch-level features (tokens)
/// 2. Matches queries to support classes at the token/patch level
/// 3. Estimates uncertainty for each prediction
/// 4. Uncertain predictions are treated more carefully during classification
///
/// This is like comparing puzzle pieces: instead of comparing whole images,
/// compare individual patches and aggregate the evidence.
/// </para>
/// <para>
/// Reference: Hiller, M., Ma, R., Harber, M., &amp; Ommer, B. (2022).
/// Rethinking Generalization in Few-Shot Classification. ECCV 2022.
/// </para>
/// </remarks>
public class FewTUREOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    #region Required Properties
    /// <summary>Gets or sets the feature extractor model.</summary>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }
    #endregion

    #region Standard Meta-Learning Properties
    /// <inheritdoc cref="IMetaLearnerOptions{T}.InnerLearningRate"/>
    public double InnerLearningRate { get; set; } = 0.01;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.OuterLearningRate"/>
    public double OuterLearningRate { get; set; } = 0.001;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.AdaptationSteps"/>
    public int AdaptationSteps { get; set; } = 1;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.MetaBatchSize"/>
    public int MetaBatchSize { get; set; } = 4;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.NumMetaIterations"/>
    public int NumMetaIterations { get; set; } = 1000;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.GradientClipThreshold"/>
    public double? GradientClipThreshold { get; set; } = 10.0;
    /// <summary>Gets or sets the random seed.</summary>
    public int? RandomSeed { get => Seed; set => Seed = value; }
    /// <inheritdoc cref="IMetaLearnerOptions{T}.EvaluationTasks"/>
    public int EvaluationTasks { get; set; } = 100;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.EvaluationFrequency"/>
    public int EvaluationFrequency { get; set; } = 100;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.EnableCheckpointing"/>
    public bool EnableCheckpointing { get; set; } = false;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.CheckpointFrequency"/>
    public int CheckpointFrequency { get; set; } = 500;
    /// <inheritdoc cref="IMetaLearnerOptions{T}.UseFirstOrder"/>
    public bool UseFirstOrder { get; set; } = true;
    /// <summary>Gets or sets the loss function.</summary>
    public ILossFunction<T>? LossFunction { get; set; }
    /// <summary>Gets or sets the outer loop optimizer.</summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }
    /// <summary>Gets or sets the inner loop optimizer.</summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }
    /// <summary>Gets or sets the episodic data loader.</summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }
    #endregion

    #region FewTURE-Specific Properties
    /// <summary>Gets or sets the number of patch tokens to consider.</summary>
    /// <value>Default is 196 (14x14 patches from a 224x224 image).</value>
    public int NumTokens { get; set; } = 196;
    /// <summary>Gets or sets the uncertainty estimation method.</summary>
    /// <value>Default is <see cref="FewTUREUncertaintyMethod.Entropy"/> (prediction entropy).</value>
    public FewTUREUncertaintyMethod UncertaintyMethod { get; set; } = FewTUREUncertaintyMethod.Entropy;
    /// <summary>Gets or sets the uncertainty threshold for reliable estimation.</summary>
    /// <value>Default is 0.5.</value>
    public double UncertaintyThreshold { get; set; } = 0.5;
    #endregion

    #region Constructors
    /// <summary>Initializes a new instance of FewTUREOptions.</summary>
    public FewTUREOptions(IFullModel<T, TInput, TOutput> metaModel)
    { Guard.NotNull(metaModel); MetaModel = metaModel; }
    #endregion

    #region IMetaLearnerOptions Implementation
    /// <inheritdoc/>
    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0 && NumMetaIterations > 0 && NumTokens > 0 && UncertaintyThreshold > 0;
    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone() => new FewTUREOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        NumTokens = NumTokens, UncertaintyMethod = UncertaintyMethod, UncertaintyThreshold = UncertaintyThreshold
    };
    #endregion
}
