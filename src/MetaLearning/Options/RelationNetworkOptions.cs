using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

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
/// 2. Sum each class's support embeddings into one class feature
/// 3. Concatenate each class feature with each query's features and pass the pair through a relation module
/// 4. The relation module outputs a relation score between 0 and 1
/// 5. The class with the highest score is the prediction; training pushes the true class's score to 1 and the
///    others to 0 with mean squared error
///
/// Instead of using predefined distances (like Euclidean), it learns a neural
/// network to measure "how related" two examples are.
/// </para>
/// </remarks>
public class RelationNetworkOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
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
    /// Gets or sets the loss that scores the relation scores against the match indicators.
    /// Default: null, which uses mean squared error - the paper's objective, eq. 2.
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
    public int? RandomSeed { get => Seed; set => Seed = value; }

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
    /// Gets or sets the relation module's architecture.
    /// </summary>
    /// <value>Default is Concatenate, the paper's module.</value>
    /// <remarks>
    /// <para>See <see cref="RelationModuleType"/>. Every architecture ends in the paper's sigmoid unit and is trained on
    /// the same relation loss.</para>
    /// </remarks>
    public RelationModuleType RelationType { get; set; } = RelationModuleType.Concatenate;

    /// <summary>
    /// Gets or sets how a class's support examples meet a query.
    /// </summary>
    /// <value>Default is EmbeddingSum, the paper's K-shot rule.</value>
    /// <remarks>
    /// <para>
    /// <see cref="RelationAggregationMethod.EmbeddingSum"/> sums each class's embeddings and computes one relation
    /// per class (Sung et al. 2018, section 3.2). The other methods compute one relation per support example and
    /// pool them per class. The default used to be Mean, whose per-example relations are not the paper's.
    /// </para>
    /// </remarks>
    public RelationAggregationMethod AggregationMethod { get; set; } = RelationAggregationMethod.EmbeddingSum;

    /// <summary>
    /// Gets or sets the number of output classes.
    /// </summary>
    /// <value>Default is 5.</value>
    public int NumClasses { get; set; } = 5;

    /// <summary>
    /// Gets or sets whether to average several independently initialised relation modules.
    /// </summary>
    /// <value>Default is false.</value>
    /// <remarks>
    /// <para>An extension: <see cref="NumHeads"/> relation modules of the configured architecture, each with its own
    /// weights, whose relation scores are averaged. All are trained on the same loss.</para>
    /// </remarks>
    public bool UseMultiHeadRelation { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of relation modules averaged when <see cref="UseMultiHeadRelation"/> is on.
    /// </summary>
    /// <value>Default is 4.</value>
    public int NumHeads { get; set; } = 4;

    /// <summary>
    /// Gets or sets whether to apply a learned linear map to the embeddings before pairing them.
    /// </summary>
    /// <value>Default is false.</value>
    /// <remarks>
    /// <para>An extension: embeddings <c>h</c> become <c>A h</c> before class features are formed and paired, with
    /// <c>A</c> starting at the identity - the paper's pairing - and trained on the relation loss.</para>
    /// </remarks>
    public bool ApplyFeatureTransform { get; set; } = false;

    /// <summary>
    /// Gets or sets the width of the relation module's pair representation.
    /// </summary>
    /// <value>Default is 8, the width of the relation module's hidden layer in the paper's Figure 2.</value>
    public int RelationHiddenDimension { get; set; } = 8;

    /// <summary>
    /// Gets or sets the weight decay of the embedding module.
    /// </summary>
    /// <value>Default is 0.0 (no regularization).</value>
    /// <remarks>
    /// <para>Adds <c>lambda * theta</c> to the embedding module's gradient - the gradient of
    /// <c>(lambda / 2) ||theta||^2</c>. It used to be added to the reported loss only and changed no update.</para>
    /// </remarks>
    public double FeatureEncoderL2Reg { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the weight decay of the relation modules.
    /// </summary>
    /// <value>Default is 0.0 (no regularization).</value>
    /// <remarks>
    /// <para>Adds <c>lambda * phi</c> to the relation modules' gradient. The paper uses weight decay on its zero-shot
    /// embedding layers.</para>
    /// </remarks>
    public double RelationModuleL2Reg { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the dropout rate of the relation module's pair representation while meta-training.
    /// </summary>
    /// <value>Default is 0.0 (no dropout).</value>
    /// <remarks>
    /// <para>An extension: inverted dropout on the representation that feeds the sigmoid unit, one mask per episode
    /// and head. Adaptation and prediction never drop anything. Must be in [0, 1).</para>
    /// </remarks>
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
        Guard.NotNull(metaModel);
        MetaModel = metaModel;
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
               (!UseMultiHeadRelation || NumHeads >= 1) &&
               RelationDropout >= 0 && RelationDropout < 1 &&
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
            RelationHiddenDimension = RelationHiddenDimension,
            FeatureEncoderL2Reg = FeatureEncoderL2Reg,
            RelationModuleL2Reg = RelationModuleL2Reg,
            RelationDropout = RelationDropout
        };
    }

    #endregion
}
