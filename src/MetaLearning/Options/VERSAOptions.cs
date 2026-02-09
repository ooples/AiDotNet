using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for VERSA (Versatile and Efficient Few-shot Learning) algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// VERSA uses an amortization network to produce task-specific parameters in a single
/// forward pass, eliminating the need for inner-loop gradient descent entirely.
/// Given support examples, the amortization network directly outputs classifier parameters.
/// </para>
/// <para><b>For Beginners:</b> VERSA takes a completely different approach to few-shot learning:
///
/// **Traditional meta-learning (MAML, etc.):**
/// "Here are 5 examples. Let me run gradient descent to learn a classifier..." (slow)
///
/// **VERSA:**
/// "Here are 5 examples. *single forward pass* Here's your classifier." (instant)
///
/// How? VERSA trains a separate neural network (the "amortization network") whose job
/// is to look at a set of examples and immediately output the weights for a classifier.
/// It's like having a factory that produces customized classifiers on demand.
///
/// Key advantages:
/// - No inner-loop optimization at all (fastest possible adaptation)
/// - Naturally handles variable numbers of support examples
/// - The amortization network learns to extract task-relevant statistics
/// </para>
/// <para>
/// Reference: Gordon, J., Bronskill, J., Bauer, M., Nowozin, S., &amp; Turner, R. E. (2019).
/// Meta-Learning Probabilistic Inference for Prediction. ICLR 2019.
/// </para>
/// </remarks>
public class VERSAOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>
    /// Gets or sets the feature extractor model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This neural network converts raw inputs into features.
    /// VERSA uses these features both for the amortization network input and for classification.
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
    /// <remarks>
    /// <para>VERSA uses a single forward pass for adaptation, so this is effectively 1.
    /// Higher values have no effect since the amortization network produces parameters directly.
    /// </para>
    /// </remarks>
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
    /// <summary>Gets or sets the inner loop optimizer (unused for VERSA).</summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }
    /// <summary>Gets or sets the episodic data loader.</summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    #endregion

    #region VERSA-Specific Properties

    /// <summary>
    /// Gets or sets the hidden dimension of the amortization network.
    /// </summary>
    /// <value>Default is 128.</value>
    /// <remarks>
    /// <para>
    /// The amortization network takes aggregated support features and produces classifier
    /// weights. This controls the hidden layer width of that network.
    /// </para>
    /// <para><b>For Beginners:</b> Controls how complex the "classifier factory" is.
    /// Larger values let it produce more nuanced classifiers but require more training.
    /// 128 is a good default for most problems.
    /// </para>
    /// </remarks>
    public int AmortizationHiddenDim { get; set; } = 128;

    /// <summary>
    /// Gets or sets the number of hidden layers in the amortization network.
    /// </summary>
    /// <value>Default is 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many layers deep the classifier factory is.
    /// 2-3 layers is usually sufficient. More layers add capacity but slow training.
    /// </para>
    /// </remarks>
    public int AmortizationNumLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the aggregation method for support set features.
    /// </summary>
    /// <value>Default is "mean".</value>
    /// <remarks>
    /// <para>
    /// Before feeding support features to the amortization network, they must be aggregated
    /// into a fixed-size representation. Options:
    /// - "mean": Average all support features per class
    /// - "sum": Sum all support features per class
    /// - "attention": Weighted aggregation using attention
    /// </para>
    /// <para><b>For Beginners:</b> How to combine multiple examples into one summary:
    /// - "mean": Take the average (simple, effective)
    /// - "sum": Add them up (similar to mean, emphasizes more examples)
    /// - "attention": Learn which examples to focus on (most powerful)
    /// </para>
    /// </remarks>
    public string AggregationMethod { get; set; } = "mean";

    /// <summary>
    /// Gets or sets whether to use a probabilistic (Bayesian) formulation.
    /// </summary>
    /// <value>Default is false.</value>
    /// <remarks>
    /// <para>
    /// When true, the amortization network outputs distribution parameters (mean and variance)
    /// instead of point estimates. This provides uncertainty quantification but adds complexity.
    /// </para>
    /// <para><b>For Beginners:</b> If true, the model also tells you how confident it is
    /// in its predictions. This is useful when you need to know "I'm not sure about this one."
    /// Set to false for simpler, faster training.
    /// </para>
    /// </remarks>
    public bool UseProbabilistic { get; set; } = false;

    /// <summary>
    /// Gets or sets the dropout rate for the amortization network.
    /// </summary>
    /// <value>Default is 0.1.</value>
    public double AmortizationDropout { get; set; } = 0.1;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of VERSAOptions.
    /// </summary>
    /// <param name="metaModel">The feature extractor model (required).</param>
    public VERSAOptions(IFullModel<T, TInput, TOutput> metaModel)
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
               AmortizationHiddenDim > 0 &&
               AmortizationNumLayers > 0 &&
               AmortizationDropout >= 0 && AmortizationDropout < 1;
    }

    /// <inheritdoc/>
    public IMetaLearnerOptions<T> Clone()
    {
        return new VERSAOptions<T, TInput, TOutput>(MetaModel)
        {
            LossFunction = LossFunction, MetaOptimizer = MetaOptimizer,
            InnerOptimizer = InnerOptimizer, DataLoader = DataLoader,
            InnerLearningRate = InnerLearningRate, OuterLearningRate = OuterLearningRate,
            AdaptationSteps = AdaptationSteps, MetaBatchSize = MetaBatchSize,
            NumMetaIterations = NumMetaIterations, GradientClipThreshold = GradientClipThreshold,
            RandomSeed = RandomSeed, EvaluationTasks = EvaluationTasks,
            EvaluationFrequency = EvaluationFrequency, EnableCheckpointing = EnableCheckpointing,
            CheckpointFrequency = CheckpointFrequency, UseFirstOrder = UseFirstOrder,
            AmortizationHiddenDim = AmortizationHiddenDim,
            AmortizationNumLayers = AmortizationNumLayers,
            AggregationMethod = AggregationMethod,
            UseProbabilistic = UseProbabilistic,
            AmortizationDropout = AmortizationDropout
        };
    }

    #endregion
}
