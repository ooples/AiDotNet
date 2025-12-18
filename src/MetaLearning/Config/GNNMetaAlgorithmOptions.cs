using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Config;

/// <summary>
/// Configuration options for Graph Neural Network-based Meta-learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// GNN-based meta-learning models the relationships between tasks and examples
/// as a graph structure. The graph neural network learns to propagate information
/// across the graph to improve generalization and adaptation.
/// </para>
/// <para><b>For Beginners:</b> This configuration controls how the GNN meta-learner works:
///
/// Key parameters:
/// - <b>NodeEmbeddingDimension:</b> Size of node representations in the graph
/// - <b>NumGNNLayers:</b> Number of graph neural network layers
/// - <b>UseGraphAttention:</b> Whether to use attention in the GNN
/// - <b>LearnEdges:</b> Whether to learn edge weights automatically
/// - <b>PoolingStrategy:</b> How to aggregate node information
/// </para>
/// <para>
/// <b>Advanced Features:</b>
/// - Graph attention mechanisms
/// - Residual connections in GNN layers
/// - Edge weight learning
/// - Multiple edge types (contains, similar, related)
/// - Hierarchical graph pooling
/// - Message passing with support sets
/// </para>
/// </remarks>
public class GNNMetaAlgorithmOptions<T, TInput, TOutput> : MetaLearningOptions<T, TInput, TOutput>
    where T : struct, IEquatable<T>, IFormattable
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the input feature dimension.
    /// </summary>
    /// <value>
    /// Dimensionality of input features for each node.
    /// Default is 128.
    /// </value>
    public int InputDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the output prediction dimension.
    /// </summary>
    /// <value>
    /// Dimensionality of output predictions.
    /// For classification, this equals number of classes.
    /// Default is 10.
    /// </value>
    public int OutputDimension { get; set; } = 10;

    /// <summary>
    /// Gets or sets the node embedding dimension.
    /// </summary>
    /// <value>
    /// Size of node representations in the graph.
    /// Affects capacity of GNN layers.
    /// Default is 64.
    /// </value>
    public int NodeEmbeddingDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the GNN hidden dimension.
    /// </summary>
    /// <value>
    /// Hidden dimension for graph neural network layers.
    /// Default is 128.
    /// </value>
    public int GNNHiddenDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the feature encoder hidden dimension.
    /// </summary>
    /// <value>
    /// Hidden dimension for input feature encoding.
    /// Default is 256.
    /// </value>
    public int FeatureHiddenDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the decoder hidden dimension.
    /// </summary>
    /// <value>
    /// Hidden dimension for task decoder network.
    /// Default is 128.
    /// </value>
    public int DecoderHiddenDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the edge predictor hidden dimension.
    /// </summary>
    /// <value>
    /// Hidden dimension for edge prediction network.
    /// Only used when LearnEdges is true.
    /// Default is 64.
    /// </value>
    public int EdgeHiddenDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of GNN layers.
    /// </summary>
    /// <value>
    /// Number of graph neural network layers.
    /// More layers allow for longer-range dependencies.
    /// Default is 3.
    /// </value>
    public int NumGNNLayers { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of feature encoder layers.
    /// </summary>
    /// <value>
    /// Number of layers in feature encoder.
    /// Default is 2.
    /// </value>
    public int NumFeatureLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of decoder layers.
    /// </summary>
    /// <value>
    /// Number of layers in task decoder.
    /// Default is 2.
    /// </value>
    public int NumDecoderLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>
    /// Number of parallel attention mechanisms.
    /// Only used when UseGraphAttention is true.
    /// Default is 8.
    /// </value>
    public int NumAttentionHeads { get; set; } = 8;

    /// <summary>
    /// Gets or sets the attention dimension per head.
    /// </summary>
    /// <value>
    /// Dimension of each attention head.
    /// Only used when UseGraphAttention is true.
    /// Default is 32.
    /// </value>
    public int AttentionDimension { get; set; } = 32;

    /// <summary>
    /// Gets or sets the maximum number of nodes in graph.
    /// </summary>
    /// <value>
    /// Maximum nodes to limit memory usage.
    /// Default is 1000.
    /// </value>
    public int MaxNodesInGraph { get; set; } = 1000;

    /// <summary>
    /// Gets or sets whether to use graph attention.
    /// </summary>
    /// <value>
    /// If true, uses attention mechanism in GNN layers.
    /// Allows nodes to attend to relevant neighbors.
    /// Default is true.
    /// </value>
    public bool UseGraphAttention { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use residual connections.
    /// </summary>
    /// <value>
    /// If true, adds residual connections in GNN layers.
    /// Helps with training deep GNNs.
    /// Default is true.
    /// </value>
    public bool UseResidualConnections { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to learn edge weights.
    /// </summary>
    /// <value>
    /// If true, learns edge weights from data.
    /// Otherwise uses fixed weights based on similarity.
    /// Default is false.
    /// </value>
    public bool LearnEdges { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to use inter-task edges.
    /// </summary>
    /// <value>
    /// If true, connects similar tasks with edges.
    /// Enables information sharing between tasks.
    /// Default is true.
    /// </value>
    public bool UseInterTaskEdges { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use intra-task edges.
    /// </summary>
    /// <value>
    /// If true, connects similar examples within tasks.
    /// Captures example relationships.
    /// Default is true.
    /// </value>
    public bool UseIntraTaskEdges { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use graph regularization.
    /// </summary>
    /// <value>
    /// If true, applies regularization on node embeddings.
    /// Prevents overfitting in graph space.
    /// Default is true.
    /// </value>
    public bool UseGraphRegularization { get; set; } = true;

    /// <summary>
    /// Gets or sets the pooling strategy for embeddings.
    /// </summary>
    /// <value>
    /// How to aggregate node embeddings to graph representation.
    /// Default is Attention.
    /// </value>
    public PoolingStrategy PoolingStrategy { get; set; } = PoolingStrategy.Attention;

    /// <summary>
    /// Gets or sets the similarity threshold for edges.
    /// </summary>
    /// <value>
    /// Minimum similarity to create an edge between nodes.
    /// Controls graph connectivity.
    /// Default is 0.7.
    /// </value>
    public double SimilarityThreshold { get; set; } = 0.7;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    /// <value>
    /// Dropout rate for regularization (0.0 to 1.0).
    /// Applied to encoder and decoder layers.
    /// Default is 0.1.
    /// </value>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the embedding regularization weight.
    /// </summary>
    /// <value>
    /// L2 regularization strength for node embeddings.
    /// Only used when UseGraphRegularization is true.
    /// Default is 1e-4.
    /// </value>
    public double EmbeddingRegularizationWeight { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the graph regularization weight.
    /// </summary>
    /// <value>
    /// Weight for graph-specific regularization terms.
    /// Default is 0.1.
    /// </value>
    public double GraphRegularizationWeight { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the edge prediction weight.
    /// </summary>
    /// <value>
    /// Weight for edge prediction loss.
    /// Only used when LearnEdges is true.
    /// Default is 0.01.
    /// </value>
    public double EdgePredictionWeight { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets whether to use hierarchical pooling.
    /// </summary>
    /// <value>
    /// If true, uses hierarchical graph pooling.
    /// Coarsens the graph hierarchically.
    /// Default is false.
    /// </value>
    public bool UseHierarchicalPooling { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of pooling levels.
    /// </summary>
    /// <value>
    /// Number of hierarchical pooling levels.
    /// Only used when UseHierarchicalPooling is true.
    /// Default is 3.
    /// </value>
    public int NumPoolingLevels { get; set; } = 3;

    /// <summary>
    /// Gets or sets whether to use edge type embeddings.
    /// </summary>
    /// <value>
    /// If true, learns embeddings for different edge types.
    /// Captures different relationship types.
    /// Default is true.
    /// </value>
    public bool UseEdgeTypeEmbeddings { get; set; } = true;

    /// <summary>
    /// Gets or sets the edge type embedding dimension.
    /// </summary>
    /// <value>
    /// Dimension of edge type embeddings.
    /// Only used when UseEdgeTypeEmbeddings is true.
    /// Default is 16.
    /// </value>
    public int EdgeTypeEmbeddingDimension { get; set; } = 16;

    /// <summary>
    /// Gets or sets whether to use temporal edges.
    /// </summary>
    /// <value>
    /// If true, adds temporal edges between consecutive episodes.
    /// Captures temporal dependencies.
    /// Default is false.
    /// </value>
    public bool UseTemporalEdges { get; set; } = false;

    /// <summary>
    /// Gets or sets the temporal window size.
    /// </summary>
    /// <value>
    /// Number of previous episodes to connect with temporal edges.
    /// Only used when UseTemporalEdges is true.
    /// Default is 5.
    /// </value>
    public int TemporalWindowSize { get; set; } = 5;

    /// <summary>
    /// Gets or sets whether to use graph normalization.
    /// </summary>
    /// <value>
    /// If true, normalizes node degrees and edge weights.
    /// Helps with training stability.
    /// Default is true.
    /// </value>
    public bool UseGraphNormalization { get; set; } = true;

    /// <summary>
    /// Gets or sets the message passing steps.
    /// </summary>
    /// <value>
    /// Number of message passing steps during adaptation.
    /// Affects information propagation.
    /// Default is 3.
    /// </value>
    public int MessagePassingSteps { get; set; } = 3;

    /// <summary>
    /// Gets or sets whether to cache graph structures.
    /// </summary>
    /// <value>
    /// If true, caches computed graphs for reuse.
    /// Improves efficiency for repeated tasks.
    /// Default is true.
    /// </value>
    public bool CacheGraphStructures { get; set; } = true;

    /// <summary>
    /// Creates a default GNN meta-learning configuration with standard values.
    /// </summary>
    /// <remarks>
    /// Default configuration based on GNN meta-learning literature:
    /// - Node embedding dimension: 64
    /// - 3 GNN layers with attention
    /// - Residual connections
    /// - Attention pooling
    /// - Inter and intra-task edges
    /// </remarks>
    public GNNMetaAlgorithmOptions()
    {
        // Set default values
        InnerLearningRate = NumOps.FromDouble(0.001);
        AdaptationSteps = 3; // Message passing steps
    }

    /// <summary>
    /// Creates a GNN meta-learning configuration with custom values.
    /// </summary>
    /// <param name="inputDimension">Input feature dimension.</param>
    /// <param name="outputDimension">Output prediction dimension.</param>
    /// <param name="nodeEmbeddingDimension">Node embedding dimension.</param>
    /// <param name="numGNNLayers">Number of GNN layers.</param>
    /// <param name="useGraphAttention">Whether to use graph attention.</param>
    /// <param name="poolingStrategy">Pooling strategy for embeddings.</param>
    /// <param name="learnEdges">Whether to learn edge weights.</param>
    /// <param name="useInterTaskEdges">Whether to use inter-task edges.</param>
    /// <param name="messagePassingSteps">Number of message passing steps.</param>
    /// <param name="innerLearningRate">Inner loop learning rate.</param>
    /// <param name="adaptationSteps">Number of adaptation steps.</param>
    /// <param name="numEpisodes">Number of training episodes.</param>
    public GNNMetaAlgorithmOptions(
        int inputDimension = 128,
        int outputDimension = 10,
        int nodeEmbeddingDimension = 64,
        int numGNNLayers = 3,
        bool useGraphAttention = true,
        PoolingStrategy poolingStrategy = PoolingStrategy.Attention,
        bool learnEdges = false,
        bool useInterTaskEdges = true,
        int messagePassingSteps = 3,
        double innerLearningRate = 0.001,
        int adaptationSteps = 3,
        int numEpisodes = 10000)
    {
        InputDimension = inputDimension;
        OutputDimension = outputDimension;
        NodeEmbeddingDimension = nodeEmbeddingDimension;
        NumGNNLayers = numGNNLayers;
        UseGraphAttention = useGraphAttention;
        PoolingStrategy = poolingStrategy;
        LearnEdges = learnEdges;
        UseInterTaskEdges = useInterTaskEdges;
        MessagePassingSteps = messagePassingSteps;
        InnerLearningRate = NumOps.FromDouble(innerLearningRate);
        AdaptationSteps = adaptationSteps;
        NumEpisodes = numEpisodes;
    }

    /// <summary>
    /// Validates the configuration parameters.
    /// </summary>
    /// <returns>True if all parameters are valid, false otherwise.</returns>
    public override bool IsValid()
    {
        // Check base class validation
        if (!base.IsValid())
            return false;

        // Check dimensions
        if (InputDimension <= 0 || InputDimension > 10000)
            return false;

        if (OutputDimension <= 0 || OutputDimension > 1000)
            return false;

        if (NodeEmbeddingDimension <= 0 || NodeEmbeddingDimension > 512)
            return false;

        if (GNNHiddenDimension <= 0 || GNNHiddenDimension > 1024)
            return false;

        if (FeatureHiddenDimension <= 0 || FeatureHiddenDimension > 2048)
            return false;

        if (DecoderHiddenDimension <= 0 || DecoderHiddenDimension > 1024)
            return false;

        if (EdgeHiddenDimension <= 0 || EdgeHiddenDimension > 512)
            return false;

        // Check layer counts
        if (NumGNNLayers <= 0 || NumGNNLayers > 20)
            return false;

        if (NumFeatureLayers <= 0 || NumFeatureLayers > 10)
            return false;

        if (NumDecoderLayers <= 0 || NumDecoderLayers > 10)
            return false;

        // Check attention parameters
        if (UseGraphAttention)
        {
            if (NumAttentionHeads <= 0 || NumAttentionHeads > 32)
                return false;

            if (AttentionDimension <= 0 || AttentionDimension > 256)
                return false;

            if (GNNHiddenDimension % NumAttentionHeads != 0)
                return false;
        }

        // Check graph constraints
        if (MaxNodesInGraph <= 0 || MaxNodesInGraph > 10000)
            return false;

        if (SimilarityThreshold < 0.0 || SimilarityThreshold > 1.0)
            return false;

        // Check dropout rate
        if (DropoutRate < 0.0 || DropoutRate >= 1.0)
            return false;

        // Check regularization weights
        if (EmbeddingRegularizationWeight < 0.0 || EmbeddingRegularizationWeight > 1.0)
            return false;

        if (GraphRegularizationWeight < 0.0 || GraphRegularizationWeight > 10.0)
            return false;

        if (EdgePredictionWeight < 0.0 || EdgePredictionWeight > 1.0)
            return false;

        // Check hierarchical pooling
        if (UseHierarchicalPooling)
        {
            if (NumPoolingLevels <= 0 || NumPoolingLevels > 10)
                return false;
        }

        // Check edge type embeddings
        if (UseEdgeTypeEmbeddings)
        {
            if (EdgeTypeEmbeddingDimension <= 0 || EdgeTypeEmbeddingDimension > 128)
                return false;
        }

        // Check temporal edges
        if (UseTemporalEdges)
        {
            if (TemporalWindowSize <= 0 || TemporalWindowSize > 100)
                return false;
        }

        // Check message passing
        if (MessagePassingSteps <= 0 || MessagePassingSteps > 20)
            return false;

        return true;
    }

    /// <summary>
    /// Gets the total number of GNN parameters.
    /// </summary>
    /// <returns>Total parameters in GNN layers.</returns>
    public int GetGNNParameterCount()
    {
        int paramsCount = 0;

        // Graph convolution layers
        for (int i = 0; i < NumGNNLayers; i++)
        {
            // Weight matrix: embedding_dim -> hidden_dim
            paramsCount += NodeEmbeddingDimension * GNNHiddenDimension + GNNHiddenDimension;

            // Attention layers
            if (UseGraphAttention)
            {
                // Multi-head attention parameters
                paramsCount += NumAttentionHeads * (
                    GNNHiddenDimension * AttentionDimension * 3 + // Q, K, V
                    AttentionDimension * AttentionDimension        // Output projection
                );
            }

            // Residual connections don't add parameters
        }

        return paramsCount;
    }

    /// <summary>
    /// Gets the total number of feature encoder parameters.
    /// </summary>
    /// <returns>Total parameters in feature encoder.</returns>
    public int GetFeatureEncoderParameterCount()
    {
        int paramsCount = 0;

        // Input layer
        paramsCount += InputDimension * FeatureHiddenDimension + FeatureHiddenDimension;

        // Hidden layers
        for (int i = 1; i < NumFeatureLayers; i++)
        {
            paramsCount += FeatureHiddenDimension * FeatureHiddenDimension + FeatureHiddenDimension;
        }

        // Output layer
        paramsCount += FeatureHiddenDimension * NodeEmbeddingDimension;

        return paramsCount;
    }

    /// <summary>
    /// Gets the total number of decoder parameters.
    /// </summary>
    /// <returns>Total parameters in decoder.</returns>
    public int GetDecoderParameterCount()
    {
        int paramsCount = 0;

        // Input layer
        paramsCount += NodeEmbeddingDimension * DecoderHiddenDimension + DecoderHiddenDimension;

        // Hidden layers
        for (int i = 1; i < NumDecoderLayers; i++)
        {
            paramsCount += DecoderHiddenDimension * DecoderHiddenDimension + DecoderHiddenDimension;
        }

        // Output layer
        paramsCount += DecoderHiddenDimension * OutputDimension;

        return paramsCount;
    }

    /// <summary>
    /// Gets the total number of edge predictor parameters.
    /// </summary>
    /// <returns>Total parameters in edge predictor.</returns>
    public int GetEdgePredictorParameterCount()
    {
        if (!LearnEdges)
        {
            return 0;
        }

        int paramsCount = 0;

        // Input layer (concatenated node embeddings)
        paramsCount += 2 * NodeEmbeddingDimension * EdgeHiddenDimension + EdgeHiddenDimension;

        // Output layer
        paramsCount += EdgeHiddenDimension * 1;  // Single output for edge probability

        return paramsCount;
    }

    /// <summary>
    /// Gets the total number of edge type embedding parameters.
    /// </summary>
    /// <returns>Total parameters for edge type embeddings.</returns>
    public int GetEdgeTypeEmbeddingParameterCount()
    {
        if (!UseEdgeTypeEmbeddings)
        {
            return 0;
        }

        // Number of edge types * embedding dimension
        var numEdgeTypes = 4; // Contains, Similar, Related, Temporal
        return numEdgeTypes * EdgeTypeEmbeddingDimension;
    }

    /// <summary>
    /// Gets the total number of parameters in the model.
    /// </summary>
    /// <returns>Total parameter count across all components.</returns>
    public int GetTotalParameterCount()
    {
        return GetGNNParameterCount() +
               GetFeatureEncoderParameterCount() +
               GetDecoderParameterCount() +
               GetEdgePredictorParameterCount() +
               GetEdgeTypeEmbeddingParameterCount();
    }

    /// <summary>
    /// Gets the estimated memory usage in MB.
    /// </summary>
    /// <returns>Estimated memory usage in megabytes.</returns>
    public double GetEstimatedMemoryUsageMB()
    {
        // Parameter memory (4 bytes per float32)
        var paramMemoryMB = (GetTotalParameterCount() * 4.0) / (1024 * 1024);

        // Graph structure memory
        var maxNodes = MaxNodesInGraph;
        var nodeEmbeddingMemoryMB = (maxNodes * NodeEmbeddingDimension * 4.0) / (1024 * 1024);

        // Edge memory (assuming sparse graph)
        var avgEdgesPerNode = 10;
        var edgeMemoryMB = (maxNodes * avgEdgesPerNode * 4.0) / (1024 * 1024);

        // Activation memory for message passing
        var activationMemoryMB = (MessagePassingSteps * maxNodes * GNNHiddenDimension * 4.0) / (1024 * 1024);

        return paramMemoryMB + nodeEmbeddingMemoryMB + edgeMemoryMB + activationMemoryMB;
    }

    /// <summary>
    /// Gets the computational complexity estimate.
    /// </summary>
    /// <returns>Estimated FLOPs per forward pass.</returns>
    public long GetComputationalComplexity()
    {
        // Feature encoding
        long flops = (long)MaxNodesInGraph * InputDimension * FeatureHiddenDimension;

        // GNN layers
        for (int i = 0; i < NumGNNLayers; i++)
        {
            // Message passing
            var avgDegree = 10;
            flops += (long)MaxNodesInGraph * avgDegree * NodeEmbeddingDimension * GNNHiddenDimension;

            // Attention
            if (UseGraphAttention)
            {
                flops += (long)MaxNodesInGraph * avgDegree * AttentionDimension * 3;
            }
        }

        // Pooling
        flops += (long)MaxNodesInGraph * NodeEmbeddingDimension;

        // Decoding
        flops += (long)MaxNodesInGraph * NodeEmbeddingDimension * DecoderHiddenDimension;

        return flops;
    }
}