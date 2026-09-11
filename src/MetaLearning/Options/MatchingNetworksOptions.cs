using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Attention function types for Matching Networks.
/// </summary>
/// <remarks>
/// <para>
/// The attention function determines how similarity is measured between query
/// and support embeddings when computing attention weights.
/// </para>
/// <para><b>For Beginners:</b> This controls how the network decides which
/// support examples are most similar to the query example.
/// </para>
/// </remarks>
public enum MatchingNetworksAttentionFunction
{
    /// <summary>
    /// Cosine similarity between embeddings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Measures the angle between vectors, ignoring magnitude.
    /// </para>
    /// </remarks>
    Cosine,

    /// <summary>
    /// Dot product similarity.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Simple inner product, considers both angle and magnitude.
    /// </para>
    /// </remarks>
    DotProduct,

    /// <summary>
    /// Negative Euclidean distance.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Uses negative distance so higher values indicate more similarity.
    /// </para>
    /// </remarks>
    Euclidean,

    /// <summary>
    /// Learned bilinear kernel on unit-normalised embeddings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// An extension: <c>c(f, g) = f_hat' W g_hat</c> with <c>f_hat</c>, <c>g_hat</c> the unit-length embeddings and
    /// <c>W</c> learned on the episode loss. <c>W</c> starts at the identity, where the kernel is exactly the paper's
    /// cosine similarity. It used to fall back to cosine silently, learning nothing.
    /// </para>
    /// </remarks>
    Learned
}

/// <summary>
/// Configuration options for Matching Networks algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// Matching Networks use attention mechanisms over the support set to classify
/// query examples. It computes a weighted sum of support labels where weights are
/// determined by an attention function that measures similarity between examples.
/// </para>
/// <para><b>For Beginners:</b> Matching Networks learn to pay attention to similar examples:
///
/// 1. Encode all examples (support and query) with a shared encoder
/// 2. For each query, compute attention weights with all support examples
/// 3. Use similarity (cosine, dot product, etc.) for weights
/// 4. Predict weighted sum of support labels (soft nearest neighbor)
///
/// Unlike ProtoNets which uses class prototypes, Matching Networks consider
/// every support example individually.
/// </para>
/// </remarks>
public class MatchingNetworksOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>
    /// Gets or sets the meta-model (encoder) to be trained. This is the only required property.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The meta-model should be a feature encoder that maps inputs to an embedding space
    /// where similar examples are close together.
    /// </para>
    /// <para><b>For Beginners:</b> This is typically a CNN for images or an MLP for tabular data.
    /// The encoder learns to produce embeddings where attention weights are meaningful.
    /// </para>
    /// </remarks>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }

    #endregion

    #region Optional Properties with Defaults

    /// <summary>
    /// Gets or sets the loss that scores the attention class probabilities against the class indices.
    /// Default: null, which uses cross-entropy - on these probabilities the paper's <c>-log P(y | x, S)</c> (eq. 2).
    /// </summary>
    public ILossFunction<T>? LossFunction { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for encoder updates.
    /// Default: null (uses built-in Adam optimizer with OuterLearningRate).
    /// </summary>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the episodic data loader for sampling tasks.
    /// Default: null (tasks must be provided manually to MetaTrain).
    /// </summary>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// Gets or sets the learning rate for the inner loop (not used in Matching Networks).
    /// </summary>
    /// <value>Default is 0.01.</value>
    public double InnerLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the learning rate for the outer loop (encoder training).
    /// </summary>
    /// <value>Default is 0.001.</value>
    public double OuterLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of adaptation steps.
    /// </summary>
    /// <value>Default is 1 (Matching Networks uses attention-based adaptation).</value>
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
    /// <value>Default is true since Matching Networks doesn't use gradient-based inner loop.</value>
    public bool UseFirstOrder { get; set; } = true;

    #endregion

    #region Matching Networks-Specific Properties

    /// <summary>
    /// Gets or sets the attention function for computing similarity.
    /// </summary>
    /// <value>Default is Cosine.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b>
    /// - Cosine: Measures angle between vectors (most common)
    /// - DotProduct: Simple inner product
    /// - Euclidean: Uses distance (inverted so closer = higher)
    /// - Learned: Uses a neural network to compute similarity
    /// </para>
    /// </remarks>
    public MatchingNetworksAttentionFunction AttentionFunction { get; set; } = MatchingNetworksAttentionFunction.Cosine;

    /// <summary>
    /// Gets or sets the number of output classes.
    /// </summary>
    /// <value>Default is 5.</value>
    public int NumClasses { get; set; } = 5;

    /// <summary>
    /// Gets or sets whether to embed the support set in context with a bidirectional LSTM.
    /// </summary>
    /// <value>Default is false.</value>
    /// <remarks>
    /// <para>
    /// The support half of the paper's full context embeddings (appendix A.2):
    /// <c>g(x_i, S) = h_fwd_i + h_bwd_i + g'(x_i)</c>, a bidirectional LSTM run over the support set as a sequence,
    /// with a skip connection from the embedding network's output <c>g'(x_i)</c>. Its weights are learned on the
    /// episode loss. <see cref="UseFullContextEmbedding"/> turns this on as well.
    /// </para>
    /// </remarks>
    public bool UseBidirectionalEncoding { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to use the paper's full context embeddings for both the support set and the queries.
    /// </summary>
    /// <value>Default is false.</value>
    /// <remarks>
    /// <para>
    /// Full context embeddings (section 2.1.2, appendix A): the support set is embedded by the bidirectional LSTM of
    /// <see cref="UseBidirectionalEncoding"/>, and each query by <c>f(x, S) = attLSTM(f'(x), g(S), K)</c> - an LSTM
    /// that runs <see cref="ProcessingSteps"/> steps, each reading the embedded support set by content-based
    /// attention, <c>h_k = LSTM(f'(x), [h_(k-1), r_(k-1)], c_(k-1)) + f'(x)</c>. The paper evaluates this on
    /// miniImageNet. All LSTM weights are learned on the episode loss.
    /// </para>
    /// </remarks>
    public bool UseFullContextEmbedding { get; set; } = false;

    /// <summary>
    /// Gets or sets <c>K</c>, the number of attention read steps of the query context embedding.
    /// </summary>
    /// <value>Default is 5.</value>
    /// <remarks>
    /// <para>
    /// Used only with <see cref="UseFullContextEmbedding"/>. The paper does not state <c>K</c>: appendix A.1 defers
    /// to the "Process" block of Vinyals et al. 2015 (Order Matters), whose experiments try 0, 1, 5 and 10
    /// processing steps and find accuracy rising with the count. Five is our choice, the middle of that range.
    /// </para>
    /// </remarks>
    public int ProcessingSteps { get; set; } = 5;

    /// <summary>
    /// Gets or sets the L2 regularization strength.
    /// </summary>
    /// <value>Default is 0.0 (no regularization).</value>
    public double L2Regularization { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the temperature for softmax attention.
    /// </summary>
    /// <value>Default is 1.0.</value>
    /// <remarks>
    /// <para>
    /// Lower temperature makes attention weights more peaked (focused on one example).
    /// Higher temperature makes attention weights more uniform.
    /// </para>
    /// </remarks>
    public double Temperature { get; set; } = 1.0;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the MatchingNetworksOptions class with the required meta-model.
    /// </summary>
    /// <param name="metaModel">The encoder to be trained (required).</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel is null.</exception>
    /// <example>
    /// <code>
    /// // Create Matching Networks with minimal configuration
    /// var options = new MatchingNetworksOptions&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(myEncoder);
    /// var matchingNets = new MatchingNetworksAlgorithm&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(options);
    ///
    /// // Create with custom attention function
    /// var options = new MatchingNetworksOptions&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(myEncoder)
    /// {
    ///     AttentionFunction = MatchingNetworksAttentionFunction.DotProduct,
    ///     UseBidirectionalEncoding = true
    /// };
    /// </code>
    /// </example>
    public MatchingNetworksOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        Guard.NotNull(metaModel);
        MetaModel = metaModel;
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <summary>
    /// Validates that all Matching Networks configuration options are properly set.
    /// </summary>
    /// <returns>True if the configuration is valid; otherwise, false.</returns>
    public bool IsValid()
    {
        return MetaModel != null &&
               OuterLearningRate > 0 &&
               Temperature > 0 &&
               NumClasses > 0 &&
               ProcessingSteps >= 1 &&
               MetaBatchSize > 0 &&
               NumMetaIterations > 0;
    }

    /// <summary>
    /// Creates a deep copy of the Matching Networks options.
    /// </summary>
    /// <returns>A new MatchingNetworksOptions instance with the same configuration.</returns>
    public IMetaLearnerOptions<T> Clone()
    {
        return new MatchingNetworksOptions<T, TInput, TOutput>(MetaModel)
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
            UseFirstOrder = UseFirstOrder,
            AttentionFunction = AttentionFunction,
            NumClasses = NumClasses,
            UseBidirectionalEncoding = UseBidirectionalEncoding,
            UseFullContextEmbedding = UseFullContextEmbedding,
            ProcessingSteps = ProcessingSteps,
            L2Regularization = L2Regularization,
            Temperature = Temperature
        };
    }

    #endregion
}
