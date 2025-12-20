using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for Relation Networks algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// Relation Networks learn to compare query examples with class examples by learning
/// a relation function that measures similarity. Unlike metric learning approaches
/// that use fixed distance functions, Relation Networks learn the relation function
/// end-to-end.
/// </para>
/// <para><b>For Beginners:</b> Relation Networks learns how to compare examples:
///
/// 1. Encode all examples (support and query) with a feature encoder
/// 2. For each query, concatenate with each support example's features
/// 3. Pass concatenated features through a relation module (neural network)
/// 4. The relation module outputs a similarity score
/// 5. Apply softmax to get class probabilities
///
/// Instead of using predefined distances (like Euclidean), it learns a neural
/// network to measure "how related" two examples are.
/// </para>
/// </remarks>
public class RelationNetworkOptions<T, TInput, TOutput> : IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>
    /// Gets or sets the meta-model (feature encoder) to be trained. This is the only required property.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The meta-model should be a feature encoder that maps inputs to an embedding space.
    /// The relation module will then learn to compare embeddings from this encoder.
    /// </para>
    /// <para><b>For Beginners:</b> This is typically a CNN for images or an MLP for tabular data.
    /// The encoder learns to produce embeddings that the relation module can effectively compare.
    /// </para>
    /// </remarks>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }

    #endregion

    #region Optional Properties with Defaults

    /// <summary>
    /// Gets or sets the loss function for training.
    /// Default: null (uses cross-entropy loss internally).
    /// </summary>
    public ILossFunction<T>? LossFunction { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for network updates.
    /// Default: null (uses built-in Adam optimizer with OuterLearningRate).
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for inner loop updates.
    /// Default: null (Relation Networks uses feed-forward comparison, not gradient-based inner loop).
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the episodic data loader for sampling tasks.
    /// Default: null (tasks must be provided manually to MetaTrain).
    /// </summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// Gets or sets the learning rate for the inner loop (not used in Relation Networks).
    /// </summary>
    /// <value>Default is 0.01.</value>
    public double InnerLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the learning rate for the outer loop (encoder and relation module training).
    /// </summary>
    /// <value>Default is 0.001.</value>
    public double OuterLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of adaptation steps.
    /// </summary>
    /// <value>Default is 1 (Relation Networks uses feed-forward comparison).</value>
    public int AdaptationSteps { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of tasks to sample per meta-training iteration.
    /// </summary>
    /// <value>Default is 4.</value>
    public int MetaBatchSize { get; set; } = 4;

    /// <summary>
    /// Gets or sets the total number of meta-training iterations.
    /// </summary>
    /// <value>Default is 1000.</value>
    public int NumMetaIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the maximum gradient norm for gradient clipping.
    /// </summary>
    /// <value>Default is 10.0.</value>
    public double? GradientClipThreshold { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Gets or sets the number of tasks to use for evaluation.
    /// </summary>
    public int EvaluationTasks { get; set; } = 100;

    /// <summary>
    /// Gets or sets how often to evaluate the meta-learner.
    /// </summary>
    public int EvaluationFrequency { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether to save model checkpoints.
    /// </summary>
    public bool EnableCheckpointing { get; set; } = false;

    /// <summary>
    /// Gets or sets how often to save checkpoints.
    /// </summary>
    public int CheckpointFrequency { get; set; } = 500;

    /// <summary>
    /// Gets or sets whether to use first-order approximation.
    /// </summary>
    /// <value>Default is true since Relation Networks doesn't use gradient-based inner loop.</value>
    public bool UseFirstOrder { get; set; } = true;

    #endregion

    #region Relation Network-Specific Properties

    /// <summary>
    /// Gets or sets the type of relation module architecture.
    /// </summary>
    /// <value>Default is Concatenate.</value>
    public RelationModuleType RelationType { get; set; } = RelationModuleType.Concatenate;

    /// <summary>
    /// Gets or sets the aggregation method for combining multiple support example scores.
    /// </summary>
    /// <value>Default is Mean.</value>
    public RelationAggregationMethod AggregationMethod { get; set; } = RelationAggregationMethod.Mean;

    /// <summary>
    /// Gets or sets the number of output classes.
    /// </summary>
    /// <value>Default is 5.</value>
    public int NumClasses { get; set; } = 5;

    /// <summary>
    /// Gets or sets whether to use multi-head relation.
    /// </summary>
    /// <value>Default is false.</value>
    public bool UseMultiHeadRelation { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of heads for multi-head relation.
    /// </summary>
    /// <value>Default is 4.</value>
    public int NumHeads { get; set; } = 4;

    /// <summary>
    /// Gets or sets whether to apply feature transformation before relation.
    /// </summary>
    /// <value>Default is false.</value>
    public bool ApplyFeatureTransform { get; set; } = false;

    /// <summary>
    /// Gets or sets the dimension for feature concatenation.
    /// </summary>
    /// <value>Default is 0 (first feature dimension).</value>
    public int ConcatenationDimension { get; set; } = 0;

    /// <summary>
    /// Gets or sets the hidden dimension for the relation module.
    /// </summary>
    /// <value>Default is 64.</value>
    public int RelationHiddenDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the L2 regularization strength for the feature encoder.
    /// </summary>
    /// <value>Default is 0.0 (no regularization).</value>
    public double FeatureEncoderL2Reg { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the L2 regularization strength for the relation module.
    /// </summary>
    /// <value>Default is 0.0 (no regularization).</value>
    public double RelationModuleL2Reg { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the dropout rate for the relation module.
    /// </summary>
    /// <value>Default is 0.0 (no dropout).</value>
    public double RelationDropout { get; set; } = 0.0;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the RelationNetworkOptions class with the required meta-model.
    /// </summary>
    /// <param name="metaModel">The feature encoder to be trained (required).</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel is null.</exception>
    public RelationNetworkOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel));
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <summary>
    /// Validates that all Relation Network configuration options are properly set.
    /// </summary>
    /// <returns>True if the configuration is valid; otherwise, false.</returns>
    public bool IsValid()
    {
        return MetaModel != null &&
               OuterLearningRate > 0 &&
               NumClasses > 0 &&
               RelationHiddenDimension > 0 &&
               MetaBatchSize > 0 &&
               NumMetaIterations > 0;
    }

    /// <summary>
    /// Creates a deep copy of the Relation Network options.
    /// </summary>
    /// <returns>A new RelationNetworkOptions instance with the same configuration.</returns>
    public IMetaLearnerOptions<T> Clone()
    {
        return new RelationNetworkOptions<T, TInput, TOutput>(MetaModel)
        {
            LossFunction = LossFunction,
            MetaOptimizer = MetaOptimizer,
            InnerOptimizer = InnerOptimizer,
            DataLoader = DataLoader,
            InnerLearningRate = InnerLearningRate,
            OuterLearningRate = OuterLearningRate,
            AdaptationSteps = AdaptationSteps,
            MetaBatchSize = MetaBatchSize,
            NumMetaIterations = NumMetaIterations,
            GradientClipThreshold = GradientClipThreshold,
            RandomSeed = RandomSeed,
            EvaluationTasks = EvaluationTasks,
            EvaluationFrequency = EvaluationFrequency,
            EnableCheckpointing = EnableCheckpointing,
            CheckpointFrequency = CheckpointFrequency,
            UseFirstOrder = UseFirstOrder,
            RelationType = RelationType,
            AggregationMethod = AggregationMethod,
            NumClasses = NumClasses,
            UseMultiHeadRelation = UseMultiHeadRelation,
            NumHeads = NumHeads,
            ApplyFeatureTransform = ApplyFeatureTransform,
            ConcatenationDimension = ConcatenationDimension,
            RelationHiddenDimension = RelationHiddenDimension,
            FeatureEncoderL2Reg = FeatureEncoderL2Reg,
            RelationModuleL2Reg = RelationModuleL2Reg,
            RelationDropout = RelationDropout
        };
    }

    #endregion
}
