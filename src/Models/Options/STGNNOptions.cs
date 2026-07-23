using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for STGNN (Spatio-Temporal Graph Neural Network).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// STGNN is a graph neural network designed for spatio-temporal forecasting that captures
/// both spatial dependencies (between nodes/locations) and temporal dynamics.
/// </para>
/// <para><b>For Beginners:</b> STGNN combines two types of learning for forecasting:
///
/// <b>The Key Insight:</b>
/// Many time series are not independent - they're connected in space. Traffic at one
/// intersection affects nearby intersections; stock prices affect related stocks.
/// STGNN models both the spatial connections (graph) and temporal patterns (time series).
///
/// <b>What is a Spatio-Temporal Graph?</b>
/// - Nodes: Locations or entities (sensors, stocks, cities)
/// - Edges: Connections between nodes (roads, correlations, trade routes)
/// - Node Features: Time series at each node
/// - The graph captures "who affects whom"
///
/// <b>How STGNN Works:</b>
/// 1. <b>Spatial Aggregation:</b> Each node gathers information from its neighbors
/// 2. <b>Temporal Modeling:</b> Process the time series at each node
/// 3. <b>Spatio-Temporal Fusion:</b> Combine spatial and temporal information
/// 4. <b>Prediction:</b> Forecast future values for all nodes
///
/// <b>STGNN Architecture:</b>
/// - Graph Convolution: Aggregates neighbor information weighted by edge strength
/// - Temporal Convolution: Captures patterns in time using 1D convolutions
/// - Gated Mechanism: Controls information flow between spatial and temporal
/// - Skip Connections: Preserves input information through deep networks
///
/// <b>Key Benefits:</b>
/// - Models complex spatial dependencies
/// - Captures multi-scale temporal patterns
/// - Handles irregular graph structures
/// - Scalable to large networks
/// </para>
/// <para>
/// <b>Reference:</b> Yu et al., "Spatio-Temporal Graph Convolutional Networks", IJCAI 2018.
/// https://arxiv.org/abs/1709.04875
/// </para>
/// </remarks>
public class STGNNOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="STGNNOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default STGNN configuration suitable for
    /// spatio-temporal forecasting on graph-structured data.
    /// </para>
    /// </remarks>
    public STGNNOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public STGNNOptions(STGNNOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SequenceLength = other.SequenceLength;
        ForecastHorizon = other.ForecastHorizon;
        NumNodes = other.NumNodes;
        NumFeatures = other.NumFeatures;
        HiddenDimension = other.HiddenDimension;
        NumSpatialLayers = other.NumSpatialLayers;
        NumTemporalLayers = other.NumTemporalLayers;
        GraphConvType = other.GraphConvType;
        TemporalKernelSize = other.TemporalKernelSize;
        UseGatedFusion = other.UseGatedFusion;
        DropoutRate = other.DropoutRate;
        UseResidualConnections = other.UseResidualConnections;
        NumSamples = other.NumSamples;
    }

    /// <summary>
    /// Gets or sets the sequence length (input time steps).
    /// </summary>
    /// <value>The sequence length, defaulting to 12.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps to use for prediction.
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
    /// <para><b>For Beginners:</b> How many locations/entities in the spatial network.
    /// For traffic forecasting, this is the number of sensors.
    /// For financial networks, this could be the number of assets.
    /// </para>
    /// </remarks>
    public int NumNodes { get; set; } = 207;

    /// <summary>
    /// Gets or sets the number of features per node.
    /// </summary>
    /// <value>The number of features, defaulting to 1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many measurements at each node per time step.
    /// Often 1 (e.g., traffic speed), but can be more (speed, volume, occupancy).
    /// </para>
    /// </remarks>
    public int NumFeatures { get; set; } = 1;

    /// <summary>
    /// Gets or sets the hidden dimension for graph and temporal convolutions.
    /// </summary>
    /// <value>The hidden dimension, defaulting to 64.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The internal representation size.
    /// Larger values capture more complex patterns but need more computation.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of spatial (graph convolution) layers.
    /// </summary>
    /// <value>The number of spatial layers, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many hops of spatial information to aggregate.
    /// 1 layer = immediate neighbors, 2 layers = neighbors of neighbors, etc.
    /// More layers capture longer-range spatial dependencies.
    /// </para>
    /// </remarks>
    public int NumSpatialLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of temporal convolution layers.
    /// </summary>
    /// <value>The number of temporal layers, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Depth of temporal processing.
    /// More layers capture more complex temporal patterns.
    /// </para>
    /// </remarks>
    public int NumTemporalLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the type of graph convolution to use.
    /// </summary>
    /// <value>The graph convolution type, defaulting to "chebyshev".</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How to aggregate neighbor information:
    /// - "chebyshev": Uses Chebyshev polynomials (spectral, efficient)
    /// - "gcn": Standard Graph Convolutional Network (simple, popular)
    /// - "gat": Graph Attention (learns which neighbors matter more)
    /// - "diffusion": Bidirectional diffusion (captures directed flow)
    /// </para>
    /// </remarks>
    public string GraphConvType { get; set; } = "chebyshev";

    /// <summary>
    /// Gets or sets the kernel size for temporal convolutions.
    /// </summary>
    /// <value>The temporal kernel size, defaulting to 3.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The window size for temporal patterns.
    /// Kernel of 3 looks at 3 consecutive time steps at once.
    /// </para>
    /// </remarks>
    public int TemporalKernelSize { get; set; } = 3;

    /// <summary>
    /// Gets or sets whether to use gated fusion of spatial and temporal features.
    /// </summary>
    /// <value>True to use gated fusion; false otherwise. Default: true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Gating lets the model learn how to balance
    /// spatial vs temporal information. Like LSTM gates, it controls information flow.
    /// </para>
    /// </remarks>
    public bool UseGatedFusion { get; set; } = true;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.3.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Prevents overfitting by randomly dropping connections
    /// during training. Higher values = more regularization.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets whether to use residual (skip) connections.
    /// </summary>
    /// <value>True to use residual connections; false otherwise. Default: true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Skip connections add the input directly to the output,
    /// making it easier to train deep networks and preserving input information.
    /// </para>
    /// </remarks>
    public bool UseResidualConnections { get; set; } = true;

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
