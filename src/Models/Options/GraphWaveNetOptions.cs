using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for GraphWaveNet (Graph WaveNet for Deep Spatial-Temporal Modeling).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// GraphWaveNet combines graph convolution networks with WaveNet-style dilated causal
/// convolutions to capture both spatial and temporal dependencies in time series data.
/// </para>
/// <para><b>For Beginners:</b> GraphWaveNet achieves state-of-the-art traffic forecasting by combining:
///
/// <b>The Key Insight:</b>
/// Traffic patterns have two types of dependencies: spatial (how one road affects nearby roads)
/// and temporal (how patterns evolve over time). GraphWaveNet uses diffusion convolution
/// for spatial modeling and dilated causal convolutions for temporal modeling.
///
/// <b>What Problems Does GraphWaveNet Solve?</b>
/// - Traffic speed/flow prediction on road networks
/// - Air quality forecasting across sensor networks
/// - Electricity load prediction across power grids
/// - Any spatio-temporal forecasting with graph structure
///
/// <b>How GraphWaveNet Works:</b>
/// 1. <b>Adaptive Adjacency:</b> Learns graph structure from node embeddings
/// 2. <b>Diffusion Convolution:</b> Bidirectional message passing on the graph
/// 3. <b>Dilated Temporal Conv:</b> WaveNet-style gated convolutions with exponentially growing dilation
/// 4. <b>Skip Connections:</b> Collects outputs from all layers for final prediction
///
/// <b>GraphWaveNet Architecture:</b>
/// - Node Embeddings: Learnable E1, E2 for adaptive graph A = softmax(ReLU(E1*E2^T))
/// - Diffusion Convolution: P^k * X * W for forward/backward random walks
/// - Gated TCN: (X * W_f + b_f) ⊙ σ(X * W_g + b_g) with dilated convolutions
/// - Skip Connections: Residual + skip from each layer to output
///
/// <b>Key Benefits:</b>
/// - No need for predefined graph structure (adaptive learning)
/// - Captures long-range temporal dependencies via dilated convolutions
/// - Bidirectional spatial propagation captures complex graph patterns
/// - Efficient parallel training unlike recurrent models
/// </para>
/// <para>
/// <b>Reference:</b> Wu et al., "Graph WaveNet for Deep Spatial-Temporal Graph Modeling", IJCAI 2019.
/// https://arxiv.org/abs/1906.00121
/// </para>
/// </remarks>
public class GraphWaveNetOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="GraphWaveNetOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default GraphWaveNet configuration optimized for
    /// traffic forecasting on road networks with adaptive graph learning.
    /// </para>
    /// </remarks>
    public GraphWaveNetOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public GraphWaveNetOptions(GraphWaveNetOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SequenceLength = other.SequenceLength;
        ForecastHorizon = other.ForecastHorizon;
        NumNodes = other.NumNodes;
        NumFeatures = other.NumFeatures;
        ResidualChannels = other.ResidualChannels;
        DilationChannels = other.DilationChannels;
        SkipChannels = other.SkipChannels;
        EndChannels = other.EndChannels;
        NodeEmbeddingDim = other.NodeEmbeddingDim;
        NumBlocks = other.NumBlocks;
        LayersPerBlock = other.LayersPerBlock;
        DiffusionSteps = other.DiffusionSteps;
        DropoutRate = other.DropoutRate;
        UseAdaptiveGraph = other.UseAdaptiveGraph;
        UsePredefinedGraph = other.UsePredefinedGraph;
        NumSamples = other.NumSamples;
    }

    /// <summary>
    /// Gets or sets the sequence length (input time steps).
    /// </summary>
    /// <value>The sequence length, defaulting to 12.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps to use as input.
    /// For traffic data with 5-minute intervals, 12 steps = 1 hour of history.
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
    /// Gets or sets the number of nodes in the graph.
    /// </summary>
    /// <value>The number of nodes, defaulting to 207 (METR-LA dataset size).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many locations/sensors in the network.
    /// For traffic forecasting, this is the number of road sensors.
    /// </para>
    /// </remarks>
    public int NumNodes { get; set; } = 207;

    /// <summary>
    /// Gets or sets the number of input features per node.
    /// </summary>
    /// <value>The number of features, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many measurements at each node per time step.
    /// Common features: (speed, time_of_day) or (speed, day_of_week).
    /// </para>
    /// </remarks>
    public int NumFeatures { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of residual channels.
    /// </summary>
    /// <value>The residual channels, defaulting to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The channel dimension for residual connections.
    /// This is the internal representation size for most processing.
    /// </para>
    /// </remarks>
    public int ResidualChannels { get; set; } = 32;

    /// <summary>
    /// Gets or sets the number of dilation channels.
    /// </summary>
    /// <value>The dilation channels, defaulting to 32.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The channel dimension for dilated convolution layers.
    /// Matches residual channels in the original implementation.
    /// </para>
    /// </remarks>
    public int DilationChannels { get; set; } = 32;

    /// <summary>
    /// Gets or sets the number of skip connection channels.
    /// </summary>
    /// <value>The skip channels, defaulting to 256.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Skip connections from each layer are projected to this dimension
    /// before being summed. Higher values capture more fine-grained multi-scale information.
    /// </para>
    /// </remarks>
    public int SkipChannels { get; set; } = 256;

    /// <summary>
    /// Gets or sets the number of end (output) channels.
    /// </summary>
    /// <value>The end channels, defaulting to 512.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The channel dimension before final projection to output.
    /// Large end channels capture complex output patterns.
    /// </para>
    /// </remarks>
    public int EndChannels { get; set; } = 512;

    /// <summary>
    /// Gets or sets the dimension of node embeddings for adaptive graph learning.
    /// </summary>
    /// <value>The node embedding dimension, defaulting to 10.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each node gets a learnable vector of this size.
    /// The adaptive adjacency is computed as A = softmax(ReLU(E1 * E2^T)).
    /// Smaller than MTGNN because GraphWaveNet uses diffusion convolution.
    /// </para>
    /// </remarks>
    public int NodeEmbeddingDim { get; set; } = 10;

    /// <summary>
    /// Gets or sets the number of temporal convolution blocks.
    /// </summary>
    /// <value>The number of blocks, defaulting to 4.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many blocks of dilated convolutions to stack.
    /// More blocks = larger receptive field for longer temporal patterns.
    /// </para>
    /// </remarks>
    public int NumBlocks { get; set; } = 4;

    /// <summary>
    /// Gets or sets the number of layers per temporal convolution block.
    /// </summary>
    /// <value>The layers per block, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each block has multiple layers with increasing dilation.
    /// Total receptive field = sum of all dilations across all layers.
    /// </para>
    /// </remarks>
    public int LayersPerBlock { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of diffusion steps (K) for graph convolution.
    /// </summary>
    /// <value>The diffusion steps, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many hops of random walk diffusion to compute.
    /// K=2 means using A, A^2 for forward and backward diffusion.
    /// Higher K captures longer-range spatial dependencies.
    /// </para>
    /// </remarks>
    public int DiffusionSteps { get; set; } = 2;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.3.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Prevents overfitting by randomly dropping connections.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets whether to use adaptive graph learning.
    /// </summary>
    /// <value>True to learn graph structure; false otherwise. Default: true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When enabled, GraphWaveNet learns the graph structure
    /// from data using node embeddings. This is the key innovation of the model.
    /// </para>
    /// </remarks>
    public bool UseAdaptiveGraph { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to also use a predefined adjacency matrix.
    /// </summary>
    /// <value>True to combine predefined and learned graphs; false otherwise. Default: true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> You can use both the predefined road network graph
    /// AND the learned adaptive graph together. This often improves performance.
    /// </para>
    /// </remarks>
    public bool UsePredefinedGraph { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of samples for uncertainty estimation.
    /// </summary>
    /// <value>The number of samples, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> For probabilistic forecasting with MC Dropout.
    /// </para>
    /// </remarks>
    public int NumSamples { get; set; } = 100;
}
