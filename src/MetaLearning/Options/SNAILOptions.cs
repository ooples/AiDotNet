using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for SNAIL (Simple Neural Attentive Meta-Learner) algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// SNAIL combines temporal convolutions with causal attention to perform sequence-to-sequence
/// meta-learning. It processes support examples as a sequence and uses attention to selectively
/// recall relevant examples when classifying query examples.
/// </para>
/// <para><b>For Beginners:</b> SNAIL treats few-shot learning as a sequence problem:
///
/// **The Idea:**
/// Feed all support examples (with their labels) one by one into the model,
/// then feed query examples (without labels). The model learns to:
/// 1. Remember important examples (using temporal convolutions)
/// 2. Focus on relevant ones (using attention)
/// 3. Predict labels for query examples
///
/// **Analogy:**
/// Imagine reading a detective story:
/// - Support examples are like clues presented throughout the story
/// - Temporal convolutions help you remember clues from different points in time
/// - Attention helps you focus on the most relevant clues when solving the mystery
/// - The query is the mystery to solve, using all the clues you've gathered
///
/// **Architecture:**
/// - Temporal Convolutions (TC): Capture short-range dependencies in the example sequence
/// - Causal Attention: Capture long-range dependencies by attending to any previous example
/// - Together: TC for local patterns + Attention for global patterns = powerful meta-learner
/// </para>
/// <para>
/// Reference: Mishra, N., Rohaninejad, M., Chen, X., &amp; Abbeel, P. (2018).
/// A Simple Neural Attentive Learner. ICLR 2018.
/// </para>
/// </remarks>
public class SNAILOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>
    /// Gets or sets the base feature extractor model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This network converts raw inputs into features
    /// before they're processed by SNAIL's temporal convolutions and attention.
    /// </para>
    /// </remarks>
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

    #region SNAIL-Specific Properties

    /// <summary>
    /// Gets or sets the number of attention heads for the causal attention blocks.
    /// </summary>
    /// <value>Default is 1.</value>
    /// <remarks>
    /// <para>
    /// Multi-head attention allows the model to attend to different aspects of the
    /// example sequence simultaneously. More heads = more diverse attention patterns.
    /// </para>
    /// <para><b>For Beginners:</b> How many different "focuses" the attention mechanism has.
    /// With 1 head, it looks at examples one way. With 4 heads, it looks at examples
    /// from 4 different perspectives simultaneously.
    /// </para>
    /// </remarks>
    public int NumAttentionHeads { get; set; } = 1;

    /// <summary>
    /// Gets or sets the key dimension for attention.
    /// </summary>
    /// <value>Default is 64.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The size of the "summary" each example gets before
    /// the model decides how much attention to pay to it. 64 is good for most tasks.
    /// </para>
    /// </remarks>
    public int AttentionKeyDim { get; set; } = 64;

    /// <summary>
    /// Gets or sets the value dimension for attention.
    /// </summary>
    /// <value>Default is 64.</value>
    public int AttentionValueDim { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of temporal convolution filters per block.
    /// </summary>
    /// <value>Default is 32.</value>
    /// <remarks>
    /// <para>
    /// Temporal convolutions capture local patterns in the example sequence.
    /// More filters can detect more diverse local patterns.
    /// </para>
    /// <para><b>For Beginners:</b> How many local pattern detectors to use.
    /// Each filter detects a different kind of pattern in the sequence of examples.
    /// 32 is a good default.
    /// </para>
    /// </remarks>
    public int NumTCFilters { get; set; } = 32;

    /// <summary>
    /// Gets or sets the number of TC+Attention blocks to stack.
    /// </summary>
    /// <value>Default is 2.</value>
    /// <remarks>
    /// <para>
    /// Each block consists of several temporal convolution layers followed by a
    /// causal attention layer. Stacking blocks increases the model's capacity
    /// to capture complex sequence patterns.
    /// </para>
    /// <para><b>For Beginners:</b> How many times to repeat the "detect local patterns
    /// then attend globally" process. 2 blocks is usually sufficient for few-shot tasks.
    /// </para>
    /// </remarks>
    public int NumBlocks { get; set; } = 2;

    /// <summary>
    /// Gets or sets the maximum sequence length the model can handle.
    /// </summary>
    /// <value>Default is 100 (supports up to ~20-way 5-shot tasks).</value>
    /// <remarks>
    /// <para>
    /// The sequence length equals N*K (support) + N*Q (query) where N=ways, K=shots, Q=query per class.
    /// This sets the maximum TC kernel span and positional encoding length.
    /// </para>
    /// <para><b>For Beginners:</b> The maximum number of examples (support + query) the
    /// model can process at once. 100 handles most few-shot settings. Increase if you
    /// have many-way or many-shot tasks.
    /// </para>
    /// </remarks>
    public int MaxSequenceLength { get; set; } = 100;

    /// <summary>
    /// Gets or sets the dropout rate for temporal convolutions and attention.
    /// </summary>
    /// <value>Default is 0.1.</value>
    public double DropoutRate { get; set; } = 0.1;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of SNAILOptions.
    /// </summary>
    /// <param name="metaModel">The feature extractor model (required).</param>
    public SNAILOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        Guard.NotNull(metaModel);
        MetaModel = metaModel;
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
               NumAttentionHeads > 0 &&
               AttentionKeyDim > 0 &&
               AttentionValueDim > 0 &&
               NumTCFilters > 0 &&
               NumBlocks > 0 &&
               MaxSequenceLength > 0 &&
               DropoutRate >= 0 && DropoutRate < 1.0;
    }

    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone()
    {
        return new SNAILOptions<T, TInput, TOutput>(MetaModel)
        {
            LossFunction = LossFunction, MetaOptimizer = MetaOptimizer,
            InnerOptimizer = InnerOptimizer, DataLoader = DataLoader,
            InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
            AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize,
            NumMetaIterations = NumMetaIterations, GradientClipThreshold = GradientClipThreshold,
            RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
            EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
            CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
            NumAttentionHeads = NumAttentionHeads, AttentionKeyDim = AttentionKeyDim,
            AttentionValueDim = AttentionValueDim, NumTCFilters = NumTCFilters,
            NumBlocks = NumBlocks, MaxSequenceLength = MaxSequenceLength,
            DropoutRate = DropoutRate
        };
    }

    #endregion
}
