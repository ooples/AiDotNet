using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for CAML (Context-Aware Meta-Learning) (Fifty et al., NeurIPS 2023).
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// CAML uses a frozen, large-scale pretrained vision transformer as a universal feature extractor
/// and adapts to few-shot tasks using non-parametric, context-aware classification.
/// </para>
/// <para><b>For Beginners:</b> CAML leverages powerful pretrained models:
/// 1. Uses a large pretrained model (e.g., CLIP, DINO) to extract features
/// 2. No fine-tuning of the backbone at all - features are frozen
/// 3. A lightweight context module adapts features based on the support set
/// 4. Classification is done by comparing adapted query features to prototypes
///
/// The key insight is that modern pretrained models produce such good features
/// that complex meta-learning is unnecessary - just adapt the classifier.
/// </para>
/// <para>
/// Reference: Fifty, C., Duan, D., Junkins, R.G., Amid, E., Leskovec, J.,
/// Re, C., &amp; Thrun, S. (2023). Context-Aware Meta-Learning. NeurIPS 2023.
/// </para>
/// </remarks>
public class CAMLOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    public int AdaptationSteps { get; set; } = 0;
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

    #region CAML-Specific Properties
    /// <summary>
    /// Gets or sets the context embedding dimension.
    /// </summary>
    /// <value>Default is 128.</value>
    public int ContextDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets whether to freeze the backbone during meta-training.
    /// </summary>
    /// <value>Default is true (recommended for pretrained backbones).</value>
    public bool FreezeBackbone { get; set; } = true;
    #endregion

    #region Constructors
    /// <summary>Initializes a new instance of CAMLOptions.</summary>
    public CAMLOptions(IFullModel<T, TInput, TOutput> metaModel)
    { MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel)); }
    #endregion

    #region IMetaLearnerOptions Implementation
    /// <inheritdoc/>
    public bool IsValid() => MetaModel != null && OuterLearningRate > 0 && MetaBatchSize > 0 && NumMetaIterations > 0;
    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone() => new CAMLOptions<T, TInput, TOutput>(MetaModel)
    {
        LossFunction = LossFunction, MetaOptimizer = MetaOptimizer, InnerOptimizer = InnerOptimizer,
        DataLoader = DataLoader, InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
        AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize, NumMetaIterations = NumMetaIterations,
        GradientClipThreshold = GradientClipThreshold, RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
        EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
        CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
        ContextDimension = ContextDimension, FreezeBackbone = FreezeBackbone
    };
    #endregion
}
