using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for FEAT (Few-shot Embedding Adaptation with Transformer) (Ye et al., CVPR 2020).
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// FEAT adapts embeddings to be task-specific using a set-to-set transformer that takes
/// class prototypes as input and outputs task-adapted prototypes. The transformer captures
/// inter-class relationships to produce more discriminative prototypes.
/// </para>
/// <para><b>For Beginners:</b> FEAT improves upon ProtoNets by making prototypes "task-aware":
///
/// **Problem with ProtoNets:**
/// Each class prototype is computed independently (mean of support features).
/// But knowing about OTHER classes in the task could help: if class A and B are similar,
/// their prototypes should be pushed apart.
///
/// **FEAT's solution:**
/// Use a transformer to let prototypes "talk to each other":
/// 1. Compute initial prototypes (like ProtoNets)
/// 2. Feed all prototypes through a set-to-set transformer
/// 3. The transformer adjusts each prototype based on the others
/// 4. Result: task-adapted prototypes that are more discriminative
///
/// **Analogy:**
/// Imagine placing labels on a map:
/// - ProtoNets: Places each label independently
/// - FEAT: Adjusts labels so they're evenly spread and don't overlap
/// </para>
/// <para>
/// Reference: Ye, H.J., Hu, H., Zhan, D.C., &amp; Sha, F. (2020).
/// Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions. CVPR 2020.
/// </para>
/// </remarks>
public class FEATOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>
    /// Gets or sets the feature extractor model.
    /// </summary>
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

    #region FEAT-Specific Properties

    /// <summary>
    /// Gets or sets the number of attention heads in the set-to-set transformer.
    /// </summary>
    /// <value>Default is 1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many different ways the transformer compares
    /// prototypes simultaneously. More heads = more diverse comparisons.
    /// </para>
    /// </remarks>
    public int NumTransformerHeads { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of transformer layers.
    /// </summary>
    /// <value>Default is 1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many rounds of "prototype conversation" to do.
    /// 1 layer is usually sufficient for few-shot tasks with 5 classes.
    /// </para>
    /// </remarks>
    public int NumTransformerLayers { get; set; } = 1;

    /// <summary>
    /// Gets or sets the balance weight between contrastive loss and classification loss.
    /// </summary>
    /// <value>Default is 0.5.</value>
    /// <remarks>
    /// <para>
    /// The total loss is: alpha * contrastive_loss + (1 - alpha) * classification_loss.
    /// The contrastive loss encourages adapted prototypes to be close to their original
    /// prototypes while being far from other classes.
    /// </para>
    /// <para><b>For Beginners:</b> Balances two training goals:
    /// - Classification: Be accurate on query examples
    /// - Contrastive: Don't change prototypes too much from their original positions
    /// 0.5 gives equal weight to both.
    /// </para>
    /// </remarks>
    public double ContrastiveWeight { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the temperature for similarity computation.
    /// </summary>
    /// <value>Default is 64.0.</value>
    public double Temperature { get; set; } = 64.0;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of FEATOptions.
    /// </summary>
    /// <param name="metaModel">The feature extractor model (required).</param>
    public FEATOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel));
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <inheritdoc/>
    public bool IsValid()
    {
        return MetaModel != null &&
               OuterLearningRate > 0 &&
               MetaBatchSize > 0 &&
               NumMetaIterations > 0 &&
               NumTransformerHeads > 0 &&
               NumTransformerLayers > 0;
    }

    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone()
    {
        return new FEATOptions<T, TInput, TOutput>(MetaModel)
        {
            LossFunction = LossFunction, MetaOptimizer = MetaOptimizer,
            InnerOptimizer = InnerOptimizer, DataLoader = DataLoader,
            InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
            AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize,
            NumMetaIterations = NumMetaIterations, GradientClipThreshold = GradientClipThreshold,
            RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
            EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
            CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
            NumTransformerHeads = NumTransformerHeads, NumTransformerLayers = NumTransformerLayers,
            ContrastiveWeight = ContrastiveWeight, Temperature = Temperature
        };
    }

    #endregion
}
