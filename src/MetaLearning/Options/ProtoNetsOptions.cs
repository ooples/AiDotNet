using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Distance functions supported by Prototypical Networks for measuring similarity between embeddings.
/// </summary>
/// <remarks>
/// <para>
/// The choice of distance function affects how the model measures similarity between
/// query embeddings and class prototypes. Different distance functions have different
/// properties and may work better for different types of data.
/// </para>
/// <para><b>For Beginners:</b> The distance function determines how we measure
/// "closeness" between examples. Euclidean is like measuring with a ruler,
/// Cosine measures the angle between vectors, and Mahalanobis accounts for
/// correlations in the data.
/// </para>
/// </remarks>
public enum ProtoNetsDistanceFunction
{
    /// <summary>
    /// Standard Euclidean (L2) distance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Computes sqrt(sum((a_i - b_i)^2)) - the straight-line distance between points.
    /// This is the most common choice and works well for most applications.
    /// </para>
    /// <para><b>Use When:</b> You have no specific reason to use another metric.</para>
    /// </remarks>
    Euclidean,

    /// <summary>
    /// Cosine distance (1 - cosine similarity).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Measures the angle between vectors, ignoring magnitude.
    /// Cosine distance = 1 - (a · b) / (||a|| * ||b||).
    /// </para>
    /// <para><b>Use When:</b> Vector magnitude is not meaningful and only
    /// direction matters (e.g., normalized embeddings).</para>
    /// </remarks>
    Cosine,

    /// <summary>
    /// Mahalanobis distance with learned or estimated covariance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Accounts for correlations between dimensions using a covariance matrix.
    /// Reduces to Euclidean when using identity covariance.
    /// </para>
    /// <para><b>Use When:</b> Dimensions have different scales or are correlated.</para>
    /// </remarks>
    Mahalanobis
}

/// <summary>
/// Configuration options for Prototypical Networks (ProtoNets) algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// Prototypical Networks learn a metric space where classification is performed by computing
/// distances to prototype representations of each class. Each prototype is the mean vector of
/// the support set examples for that class in the learned embedding space.
/// </para>
/// <para><b>For Beginners:</b> ProtoNets is one of the simplest and most effective
/// few-shot learning methods:
///
/// 1. Use a neural network to convert images/data into feature vectors
/// 2. For each class in a task, compute the "prototype" (average feature vector)
/// 3. To classify a new example, find the nearest prototype
/// 4. Train the network to make same-class examples cluster together
///
/// Unlike MAML, ProtoNets doesn't need gradient updates at test time - just compute
/// prototypes and measure distances!
/// </para>
/// </remarks>
public class ProtoNetsOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>
    /// Gets or sets the meta-model (feature encoder) to be trained. This is the only required property.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The meta-model should be a feature encoder that maps inputs to an embedding space
    /// where same-class examples are close together and different-class examples are far apart.
    /// </para>
    /// <para><b>For Beginners:</b> This is typically a CNN for images or an MLP for tabular data.
    /// The network learns to output feature vectors that cluster by class.
    /// </para>
    /// </remarks>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }

    #endregion

    #region Optional Properties with Defaults

    /// <summary>
    /// Gets or sets the loss function for training.
    /// Default: null (uses cross-entropy loss internally).
    /// </summary>
    /// <remarks>
    /// <para>
    /// ProtoNets typically uses cross-entropy loss with softmax over negative distances.
    /// This encourages the correct class prototype to be closer than other prototypes.
    /// </para>
    /// </remarks>
    public ILossFunction<T>? LossFunction { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for feature encoder updates.
    /// Default: null (uses built-in Adam optimizer with OuterLearningRate).
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the episodic data loader for sampling tasks.
    /// Default: null (tasks must be provided manually to MetaTrain).
    /// </summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// Gets or sets the learning rate for the inner loop (task adaptation).
    /// </summary>
    /// <value>The inner learning rate. Default is 0.01.</value>
    /// <remarks>
    /// <para>
    /// Note: ProtoNets doesn't perform gradient-based inner loop adaptation like MAML.
    /// This value is kept for interface compatibility but is not used in the core algorithm.
    /// Prototype computation is non-parametric.
    /// </para>
    /// </remarks>
    public double InnerLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the learning rate for the outer loop (encoder training).
    /// </summary>
    /// <value>The outer learning rate. Default is 0.001.</value>
    /// <remarks>
    /// <para>
    /// This controls how quickly the feature encoder is updated during training.
    /// A typical range is 0.0001 to 0.01.
    /// </para>
    /// <para><b>For Beginners:</b> This is how fast the embedding network learns.
    /// Start with 0.001 and adjust based on training stability.
    /// </para>
    /// </remarks>
    public double OuterLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of adaptation steps.
    /// </summary>
    /// <value>Default is 1 (ProtoNets uses non-parametric adaptation).</value>
    /// <remarks>
    /// <para>
    /// ProtoNets doesn't use gradient-based adaptation steps like MAML.
    /// This value is kept for interface compatibility. The "adaptation" in ProtoNets
    /// is simply computing class prototypes, which is done in one step.
    /// </para>
    /// </remarks>
    public int AdaptationSteps { get; set; } = 1;

    /// <summary>
    /// Gets or sets the number of tasks to sample per meta-training iteration.
    /// </summary>
    /// <value>The meta-batch size. Default is 4.</value>
    /// <remarks>
    /// <para>
    /// Each training step samples this many tasks (episodes), computes loss on each,
    /// and averages the gradients before updating the encoder.
    /// </para>
    /// </remarks>
    public int MetaBatchSize { get; set; } = 4;

    /// <summary>
    /// Gets or sets the total number of meta-training iterations.
    /// </summary>
    /// <value>The number of meta-iterations. Default is 1000.</value>
    public int NumMetaIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the maximum gradient norm for gradient clipping.
    /// </summary>
    /// <value>The gradient clip threshold, or null to disable. Default is 10.0.</value>
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
    /// <value>Default is true since ProtoNets doesn't use gradient-based inner loop.</value>
    /// <remarks>
    /// <para>
    /// For ProtoNets, this option is less relevant because adaptation is non-parametric
    /// (prototype computation, not gradient descent). This property is provided for
    /// interface compatibility with IMetaLearnerOptions.
    /// </para>
    /// </remarks>
    public bool UseFirstOrder { get; set; } = true;

    #endregion

    #region ProtoNets-Specific Properties

    /// <summary>
    /// Gets or sets the distance function for measuring similarity between embeddings.
    /// </summary>
    /// <value>The distance function. Default is Euclidean.</value>
    /// <remarks>
    /// <para>
    /// The distance function determines how similarity is measured between query
    /// embeddings and class prototypes. Euclidean distance is the default and
    /// works well for most applications.
    /// </para>
    /// <para><b>For Beginners:</b> Start with Euclidean. Try Cosine if your features
    /// are normalized. Use Mahalanobis if you have domain knowledge about feature correlations.
    /// </para>
    /// </remarks>
    public ProtoNetsDistanceFunction DistanceFunction { get; set; } = ProtoNetsDistanceFunction.Euclidean;

    /// <summary>
    /// Gets or sets the temperature for softmax scaling.
    /// </summary>
    /// <value>The temperature value. Default is 1.0.</value>
    /// <remarks>
    /// <para>
    /// Temperature controls the sharpness of the probability distribution:
    /// - Lower temperature (&lt; 1.0): Sharper, more confident predictions
    /// - Higher temperature (&gt; 1.0): Softer, more uniform predictions
    /// </para>
    /// <para><b>For Beginners:</b> Leave at 1.0 unless you want to calibrate confidence.
    /// Lower values make the model more "certain" about its predictions.
    /// </para>
    /// </remarks>
    public double Temperature { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to L2-normalize feature embeddings.
    /// </summary>
    /// <value>True to normalize features; false otherwise. Default is false.</value>
    /// <remarks>
    /// <para>
    /// When enabled, feature vectors are normalized to unit length before computing
    /// prototypes and distances. This can improve stability and is recommended when
    /// using cosine distance.
    /// </para>
    /// <para><b>For Beginners:</b> Enable this if you're using Cosine distance, or if
    /// you notice that feature magnitudes vary widely.
    /// </para>
    /// </remarks>
    public bool NormalizeFeatures { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to use an attention mechanism for prototype computation.
    /// </summary>
    /// <value>True to use attention; false for simple averaging. Default is false.</value>
    /// <remarks>
    /// <para>
    /// When enabled, uses learned attention weights to compute weighted prototypes
    /// instead of simple averaging. This can help focus on more informative examples.
    /// </para>
    /// <para><b>For Beginners:</b> Start with false (simple averaging).
    /// Enable if you have noisy or heterogeneous support sets.
    /// </para>
    /// </remarks>
    public bool UseAttentionMechanism { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to use adaptive class-specific scaling factors.
    /// </summary>
    /// <value>True to use adaptive scaling; false otherwise. Default is false.</value>
    /// <remarks>
    /// <para>
    /// When enabled, learns per-class scaling factors for distances.
    /// This allows the model to handle classes with different intra-class variances.
    /// </para>
    /// </remarks>
    public bool UseAdaptiveClassScaling { get; set; } = false;

    /// <summary>
    /// Gets or sets the scaling factor for Mahalanobis distance.
    /// </summary>
    /// <value>The Mahalanobis scaling factor. Default is 1.0.</value>
    /// <remarks>
    /// <para>
    /// This is a simplified scaling factor used when DistanceFunction is Mahalanobis.
    /// In a full implementation, this would be replaced by a learned covariance matrix.
    /// </para>
    /// </remarks>
    public double MahalanobisScaling { get; set; } = 1.0;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the ProtoNetsOptions class with the required meta-model.
    /// </summary>
    /// <param name="metaModel">The feature encoder to be trained (required).</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel is null.</exception>
    /// <example>
    /// <code>
    /// // Create ProtoNets with minimal configuration
    /// var options = new ProtoNetsOptions&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(myEncoder);
    /// var protoNets = new ProtoNetsAlgorithm&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(options);
    ///
    /// // Create ProtoNets with custom configuration
    /// var options = new ProtoNetsOptions&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(myEncoder)
    /// {
    ///     DistanceFunction = ProtoNetsDistanceFunction.Cosine,
    ///     NormalizeFeatures = true,
    ///     Temperature = 0.5
    /// };
    /// </code>
    /// </example>
    public ProtoNetsOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        MetaModel = metaModel ?? throw new ArgumentNullException(nameof(metaModel));
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <summary>
    /// Validates that all ProtoNets configuration options are properly set.
    /// </summary>
    /// <returns>True if the configuration is valid; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// Checks:
    /// - MetaModel is set
    /// - Learning rate is positive
    /// - Temperature is positive
    /// - Batch sizes and iteration counts are positive
    /// </para>
    /// </remarks>
    public bool IsValid()
    {
        return MetaModel != null &&
               OuterLearningRate > 0 &&
               Temperature > 0 &&
               MetaBatchSize > 0 &&
               NumMetaIterations > 0 &&
               EvaluationTasks > 0;
    }

    /// <summary>
    /// Creates a deep copy of the ProtoNets options.
    /// </summary>
    /// <returns>A new ProtoNetsOptions instance with the same configuration.</returns>
    public IMetaLearnerOptions<T> Clone()
    {
        return new ProtoNetsOptions<T, TInput, TOutput>(MetaModel)
        {
            LossFunction = LossFunction,
            MetaOptimizer = MetaOptimizer,
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
            DistanceFunction = DistanceFunction,
            Temperature = Temperature,
            NormalizeFeatures = NormalizeFeatures,
            UseAttentionMechanism = UseAttentionMechanism,
            UseAdaptiveClassScaling = UseAdaptiveClassScaling,
            MahalanobisScaling = MahalanobisScaling
        };
    }

    #endregion
}
