using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for MTGNN (Multivariate Time-series Graph Neural Network).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// MTGNN is a graph neural network that simultaneously learns the graph structure
/// and performs spatio-temporal forecasting, without requiring a predefined adjacency matrix.
/// </para>
/// <para><b>For Beginners:</b> MTGNN is unique because it LEARNS how variables are connected:
///
/// <b>The Key Insight:</b>
/// Unlike other graph models that require you to define the graph structure upfront,
/// MTGNN automatically discovers which time series influence each other through
/// an adaptive graph learning module. This is powerful when relationships are unknown.
///
/// <b>What Problems Does MTGNN Solve?</b>
/// - Traffic prediction when road network is complex or unknown
/// - Multivariate financial forecasting with unknown correlations
/// - Sensor networks where dependencies change over time
/// - Any multivariate time series where inter-variable relationships matter
///
/// <b>How MTGNN Works:</b>
/// 1. <b>Graph Learning:</b> Learns node embeddings and computes their similarity
/// 2. <b>Mix-hop Propagation:</b> Aggregates information from different hop distances
/// 3. <b>Dilated Inception:</b> Captures multi-scale temporal patterns
/// 4. <b>Joint Learning:</b> Graph structure and predictions are learned together
///
/// <b>MTGNN Architecture:</b>
/// - Node Embeddings: Each node gets a learnable embedding vector
/// - Adaptive Adjacency: A = softmax(E1 * E2^T) computes learned graph
/// - Mix-hop Propagation: Combines 1-hop, 2-hop, ... K-hop neighbors
/// - Dilated Inception: Parallel dilated convolutions for temporal patterns
///
/// <b>Key Benefits:</b>
/// - No need to predefine graph structure
/// - Discovers hidden variable relationships
/// - Handles multivariate series with complex dependencies
/// - Combines best of GCN and TCN architectures
/// </para>
/// <para>
/// <b>Reference:</b> Wu et al., "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks", KDD 2020.
/// https://arxiv.org/abs/2005.11650
/// </para>
/// </remarks>
public class MTGNNOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="MTGNNOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default MTGNN configuration suitable for
    /// multivariate time series forecasting with automatic graph structure learning.
    /// </para>
    /// </remarks>
    public MTGNNOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public MTGNNOptions(MTGNNOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SequenceLength = other.SequenceLength;
        ForecastHorizon = other.ForecastHorizon;
        NumNodes = other.NumNodes;
        NumFeatures = other.NumFeatures;
        HiddenDimension = other.HiddenDimension;
        NodeEmbeddingDim = other.NodeEmbeddingDim;
        NumLayers = other.NumLayers;
        MixHopDepth = other.MixHopDepth;
        TemporalKernelSize = other.TemporalKernelSize;
        DilationFactor = other.DilationFactor;
        DropoutRate = other.DropoutRate;
        UsePredefinedGraph = other.UsePredefinedGraph;
        UseSubgraphSampling = other.UseSubgraphSampling;
        SubgraphSize = other.SubgraphSize;
        NumSamples = other.NumSamples;
    }

    /// <summary>
    /// Gets or sets the sequence length (input time steps).
    /// </summary>
    /// <value>The sequence length, defaulting to 12.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps to use as input.
    /// More steps provide more context but require more computation.
    /// </para>
    /// </remarks>
    public int SequenceLength { get; set; } = 12;

    /// <summary>
    /// Gets or sets the forecast horizon.
    /// </summary>
    /// <value>The forecast horizon, defaulting to 12.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many future time steps to predict.
    /// </para>
    /// </remarks>
    public int ForecastHorizon { get; set; } = 12;

    /// <summary>
    /// Gets or sets the number of nodes (variables/time series).
    /// </summary>
    /// <value>The number of nodes, defaulting to 207 (METR-LA dataset size).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many time series to model simultaneously.
    /// MTGNN learns the relationships between these series automatically.
    /// </para>
    /// </remarks>
    public int NumNodes { get; set; } = 207;

    /// <summary>
    /// Gets or sets the number of features per node per time step.
    /// </summary>
    /// <value>The number of features, defaulting to 1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many measurements at each node per time step.
    /// Often 1 (single time series), but can be more for multivariate nodes.
    /// </para>
    /// </remarks>
    public int NumFeatures { get; set; } = 1;

    /// <summary>
    /// Gets or sets the hidden dimension for the model.
    /// </summary>
    /// <value>The hidden dimension, defaulting to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The internal representation size.
    /// Larger values capture more complex patterns but need more computation.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 32;

    /// <summary>
    /// Gets or sets the dimension of node embeddings for graph learning.
    /// </summary>
    /// <value>The node embedding dimension, defaulting to 40.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each node gets a learnable vector of this size.
    /// The similarity between these vectors determines the learned graph structure.
    /// Higher dimensions can capture more nuanced relationships.
    /// </para>
    /// </remarks>
    public int NodeEmbeddingDim { get; set; } = 40;

    /// <summary>
    /// Gets or sets the number of layers in the model.
    /// </summary>
    /// <value>The number of layers, defaulting to 3.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many graph-temporal processing layers to stack.
    /// More layers capture more complex patterns but may overfit on small datasets.
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 3;

    /// <summary>
    /// Gets or sets the depth of mix-hop propagation.
    /// </summary>
    /// <value>The mix-hop depth, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How far to aggregate spatial information.
    /// Depth of 2 means combining direct neighbors with 2-hop neighbors.
    /// Higher depth captures longer-range spatial dependencies.
    /// </para>
    /// </remarks>
    public int MixHopDepth { get; set; } = 2;

    /// <summary>
    /// Gets or sets the kernel size for temporal convolutions.
    /// </summary>
    /// <value>The temporal kernel size, defaulting to 7.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The window size for temporal pattern detection.
    /// Larger kernels capture longer temporal patterns in a single layer.
    /// </para>
    /// </remarks>
    public int TemporalKernelSize { get; set; } = 7;

    /// <summary>
    /// Gets or sets the dilation factor for temporal convolutions.
    /// </summary>
    /// <value>The dilation factor, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls the exponentially growing receptive field.
    /// With dilation 2, layers see patterns at 1, 2, 4, 8... time steps apart.
    /// This efficiently captures both short and long-term temporal patterns.
    /// </para>
    /// </remarks>
    public int DilationFactor { get; set; } = 2;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.3.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Prevents overfitting by randomly dropping connections.
    /// Higher values = more regularization but potentially underfitting.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets whether to use a predefined adjacency matrix alongside learned graph.
    /// </summary>
    /// <value>True to combine predefined and learned graphs; false otherwise. Default: false.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> If you have prior knowledge of the graph structure,
    /// enabling this allows combining it with the learned structure. Pure adaptive
    /// learning (false) discovers structure entirely from data.
    /// </para>
    /// </remarks>
    public bool UsePredefinedGraph { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to use subgraph sampling during training.
    /// </summary>
    /// <value>True to use subgraph sampling; false otherwise. Default: false.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> For very large graphs, processing all nodes at once
    /// is expensive. Subgraph sampling trains on random subsets of nodes, making
    /// training feasible for graphs with thousands of nodes.
    /// </para>
    /// </remarks>
    public bool UseSubgraphSampling { get; set; } = false;

    /// <summary>
    /// Gets or sets the size of sampled subgraphs.
    /// </summary>
    /// <value>The subgraph size, defaulting to 20.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When using subgraph sampling, how many nodes to
    /// include in each training batch. Smaller = faster but noisier gradients.
    /// </para>
    /// </remarks>
    public int SubgraphSize { get; set; } = 20;

    /// <summary>
    /// Gets or sets the number of samples for uncertainty estimation.
    /// </summary>
    /// <value>The number of samples, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> For probabilistic forecasting, generate multiple
    /// predictions using dropout at inference time (MC Dropout).
    /// </para>
    /// </remarks>
    public int NumSamples { get; set; } = 100;
}
