using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Options;

/// <summary>
/// Configuration options for the Graph Neural Network Meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// GNN-based meta-learning models tasks and examples as nodes in a graph,
/// with edges representing relationships between them. The graph neural network
/// learns to propagate information across the task structure to improve learning.
/// </para>
/// <para>
/// <b>For Beginners:</b> GNN Meta-learning treats the learning process as a graph problem:
/// 1. Each task or example becomes a node in a graph
/// 2. Relationships between tasks are edges (e.g., task similarity)
/// 3. A graph neural network learns to propagate useful information
/// 4. This allows the model to leverage task relationships for better adaptation
///
/// Imagine a social network where users are tasks and friendships are similarities.
/// By looking at what similar tasks learned, new tasks can adapt faster.
/// </para>
/// <para>
/// <b>Key Components:</b>
/// - <b>Node Embeddings:</b> Represent tasks/examples as vectors
/// - <b>Message Passing:</b> Share information between connected nodes
/// - <b>Graph Aggregation:</b> Combine node information for predictions
/// </para>
/// </remarks>
public class GNNMetaOptions<T, TInput, TOutput> : ModelOptions, IMetaLearnerOptions<T>
{
    #region Required Properties

    /// <summary>
    /// Gets or sets the meta-model to be trained. This is the only required property.
    /// </summary>
    public IFullModel<T, TInput, TOutput> MetaModel { get; set; }

    #endregion

    #region Optional Properties with Defaults

    /// <summary>
    /// Gets or sets the loss function for training.
    /// </summary>
    /// <value>Default: null (uses model's default loss function if available).</value>
    public ILossFunction<T>? LossFunction { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for meta-parameter updates (outer loop).
    /// </summary>
    /// <value>Default: null (uses built-in Adam optimizer).</value>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for inner-loop adaptation.
    /// </summary>
    /// <value>Default: null (uses built-in SGD optimizer).</value>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the episodic data loader for sampling tasks.
    /// </summary>
    /// <value>Default: null (tasks must be provided manually to MetaTrain).</value>
    public IEpisodicDataLoader<T, TInput, TOutput>? DataLoader { get; set; }

    /// <summary>
    /// Gets or sets the learning rate for the inner loop (task adaptation).
    /// </summary>
    /// <value>Default: 0.01.</value>
    public double InnerLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the learning rate for the outer loop (meta-update).
    /// </summary>
    /// <value>Default: 0.001.</value>
    public double OuterLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of gradient steps to take during inner loop adaptation.
    /// </summary>
    /// <value>Default: 5.</value>
    public int AdaptationSteps { get; set; } = 5;

    /// <summary>
    /// Gets or sets the number of tasks to sample per meta-training iteration.
    /// </summary>
    /// <value>Default: 4.</value>
    public int MetaBatchSize { get; set; } = 4;

    /// <summary>
    /// Gets or sets the total number of meta-training iterations to perform.
    /// </summary>
    /// <value>Default: 10000.</value>
    public int NumMetaIterations { get; set; } = 10000;

    /// <summary>
    /// Gets or sets whether to use first-order approximation.
    /// </summary>
    /// <value>Default: true.</value>
    public bool UseFirstOrder { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum gradient norm for gradient clipping.
    /// </summary>
    /// <value>Default: 10.0.</value>
    public double? GradientClipThreshold { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    /// <value>Default: null (non-deterministic).</value>
    public int? RandomSeed { get => Seed; set => Seed = value; }

    /// <summary>
    /// Gets or sets the number of tasks to use for evaluation.
    /// </summary>
    /// <value>Default: 100.</value>
    public int EvaluationTasks { get; set; } = 100;

    /// <summary>
    /// Gets or sets how often to evaluate during meta-training.
    /// </summary>
    /// <value>Default: 500.</value>
    public int EvaluationFrequency { get; set; } = 500;

    /// <summary>
    /// Gets or sets whether to save checkpoints during training.
    /// </summary>
    /// <value>Default: false.</value>
    public bool EnableCheckpointing { get; set; } = false;

    /// <summary>
    /// Gets or sets how often to save checkpoints.
    /// </summary>
    /// <value>Default: 1000.</value>
    public int CheckpointFrequency { get; set; } = 1000;

    #endregion

    #region GNN-Specific Properties

    /// <summary>
    /// Gets or sets the dimension of node embeddings in the task graph.
    /// </summary>
    /// <value>Default: 128.</value>
    public int NodeEmbeddingDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the dimension of edge features in the task graph.
    /// </summary>
    /// <value>Default: 64.</value>
    public int EdgeFeatureDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of message passing layers in the GNN.
    /// </summary>
    /// <value>Default: 3.</value>
    public int NumMessagePassingLayers { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of attention heads for graph attention.
    /// </summary>
    /// <value>Default: 4.</value>
    public int NumAttentionHeads { get; set; } = 4;

    /// <summary>
    /// Gets or sets the hidden dimension for the GNN layers.
    /// </summary>
    /// <value>Default: 256.</value>
    public int GNNHiddenDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the dropout rate for GNN layers.
    /// </summary>
    /// <value>Default: 0.1.</value>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the aggregation method for graph-level representations.
    /// </summary>
    /// <value>Default: Mean.</value>
    public GNNAggregationType AggregationType { get; set; } = GNNAggregationType.Mean;

    /// <summary>
    /// Gets or sets whether to use residual connections in GNN layers.
    /// </summary>
    /// <value>Default: true.</value>
    public bool UseResidualConnections { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use layer normalization in GNN layers.
    /// </summary>
    /// <value>Default: true.</value>
    public bool UseLayerNorm { get; set; } = true;

    /// <summary>
    /// Gets or sets how task similarity is computed for building the task graph.
    /// </summary>
    /// <value>Default: ParameterDistance.</value>
    public TaskSimilarityMetric SimilarityMetric { get; set; } = TaskSimilarityMetric.ParameterDistance;

    /// <summary>
    /// Gets or sets the threshold for creating edges between tasks.
    /// </summary>
    /// <value>Default: 0.5 (tasks with similarity above this are connected).</value>
    public double EdgeThreshold { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets whether to learn edge weights during training.
    /// </summary>
    /// <value>Default: true.</value>
    public bool LearnEdgeWeights { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use a fully connected task graph.
    /// </summary>
    /// <value>Default: false (uses sparse graph based on similarity).</value>
    public bool UseFullyConnectedGraph { get; set; } = false;

    /// <summary>
    /// Gets or sets the maximum number of neighbors per node in sparse graph.
    /// </summary>
    /// <value>Default: 10.</value>
    public int MaxNeighbors { get; set; } = 10;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the GNNMetaOptions class with the required meta-model.
    /// </summary>
    /// <param name="metaModel">The meta-model to be trained (required).</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel is null.</exception>
    public GNNMetaOptions(IFullModel<T, TInput, TOutput> metaModel)
    {
        Guard.NotNull(metaModel);
        MetaModel = metaModel;
    }

    #endregion

    #region IMetaLearnerOptions Implementation

    /// <summary>
    /// Validates that all GNN Meta configuration options are properly set.
    /// </summary>
    /// <returns>True if the configuration is valid for GNN Meta training; otherwise, false.</returns>
    public bool IsValid()
    {
        return MetaModel != null &&
               InnerLearningRate > 0 &&
               OuterLearningRate > 0 &&
               AdaptationSteps >= 0 &&
               MetaBatchSize > 0 &&
               NumMetaIterations > 0 &&
               NodeEmbeddingDimension > 0 &&
               NumMessagePassingLayers > 0 &&
               GNNHiddenDimension > 0 &&
               (EdgeThreshold > 0 && EdgeThreshold <= 1) &&
               MaxNeighbors > 0;
    }

    /// <summary>
    /// Creates a deep copy of the GNN Meta options.
    /// </summary>
    /// <returns>A new GNNMetaOptions instance with the same configuration values.</returns>
    public IMetaLearnerOptions<T> Clone()
    {
        return new GNNMetaOptions<T, TInput, TOutput>(MetaModel)
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
            UseFirstOrder = UseFirstOrder,
            GradientClipThreshold = GradientClipThreshold,
            RandomSeed = RandomSeed,
            EvaluationTasks = EvaluationTasks,
            EvaluationFrequency = EvaluationFrequency,
            EnableCheckpointing = EnableCheckpointing,
            CheckpointFrequency = CheckpointFrequency,
            NodeEmbeddingDimension = NodeEmbeddingDimension,
            EdgeFeatureDimension = EdgeFeatureDimension,
            NumMessagePassingLayers = NumMessagePassingLayers,
            NumAttentionHeads = NumAttentionHeads,
            GNNHiddenDimension = GNNHiddenDimension,
            DropoutRate = DropoutRate,
            AggregationType = AggregationType,
            UseResidualConnections = UseResidualConnections,
            UseLayerNorm = UseLayerNorm,
            SimilarityMetric = SimilarityMetric,
            EdgeThreshold = EdgeThreshold,
            LearnEdgeWeights = LearnEdgeWeights,
            UseFullyConnectedGraph = UseFullyConnectedGraph,
            MaxNeighbors = MaxNeighbors
        };
    }

    #endregion
}

/// <summary>
/// Specifies how nodes are aggregated to form graph-level representations.
/// </summary>
public enum GNNAggregationType
{
    /// <summary>
    /// Mean pooling of all node embeddings.
    /// </summary>
    Mean,

    /// <summary>
    /// Sum of all node embeddings.
    /// </summary>
    Sum,

    /// <summary>
    /// Maximum over each dimension of node embeddings.
    /// </summary>
    Max,

    /// <summary>
    /// Attention-weighted aggregation of node embeddings.
    /// </summary>
    Attention,

    /// <summary>
    /// Set2Set aggregation using LSTM.
    /// </summary>
    Set2Set
}

/// <summary>
/// Specifies how task similarity is computed for building the task graph.
/// </summary>
public enum TaskSimilarityMetric
{
    /// <summary>
    /// Distance between adapted model parameters.
    /// </summary>
    ParameterDistance,

    /// <summary>
    /// Similarity based on gradient directions.
    /// </summary>
    GradientSimilarity,

    /// <summary>
    /// Similarity based on task data distributions.
    /// </summary>
    DataDistribution,

    /// <summary>
    /// Learned similarity from a neural network.
    /// </summary>
    Learned
}
