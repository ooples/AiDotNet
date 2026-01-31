using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TemporalGCN (Temporal Graph Convolutional Network).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// TemporalGCN combines graph convolutional networks with recurrent neural networks
/// to model both spatial and temporal dependencies in graph-structured time series data.
/// </para>
/// <para><b>For Beginners:</b> TemporalGCN captures two types of patterns simultaneously:
///
/// <b>The Key Insight:</b>
/// Many real-world systems have both spatial structure (like a road network) and
/// temporal dynamics (like traffic patterns over time). TemporalGCN learns both
/// by alternating between spatial and temporal processing layers.
///
/// <b>What is Temporal Graph Convolution?</b>
/// - Standard GCN: Aggregates information from neighboring nodes at one time step
/// - TemporalGCN: Also captures how patterns evolve over time
/// - Combines spatial GCN layers with temporal recurrent (GRU/LSTM) layers
/// - Creates a "video" view of the graph, not just a "snapshot"
///
/// <b>How TemporalGCN Works:</b>
/// 1. <b>Graph Convolution:</b> Each node aggregates features from neighbors
/// 2. <b>Temporal Recurrence:</b> GRU/LSTM processes the sequence at each node
/// 3. <b>Spatial-Temporal Stacking:</b> Alternate spatial and temporal layers
/// 4. <b>Prediction:</b> Output future values at each node
///
/// <b>TemporalGCN Architecture:</b>
/// - GCN layers with Chebyshev polynomial approximation
/// - GRU cells for temporal modeling
/// - Batch normalization for stability
/// - Residual connections for gradient flow
///
/// <b>Key Benefits:</b>
/// - Jointly learns spatial and temporal patterns
/// - Handles dynamic graphs where edges change
/// - Scales to large graphs with sparse operations
/// - Works with irregular spatial structures
/// </para>
/// <para>
/// <b>Reference:</b> Zhao et al., "T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction", 2019.
/// https://arxiv.org/abs/1811.05320
/// </para>
/// </remarks>
public class TemporalGCNOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TemporalGCNOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default TemporalGCN configuration suitable for
    /// traffic prediction and other spatio-temporal forecasting tasks on graph-structured data.
    /// </para>
    /// </remarks>
    public TemporalGCNOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public TemporalGCNOptions(TemporalGCNOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SequenceLength = other.SequenceLength;
        ForecastHorizon = other.ForecastHorizon;
        NumNodes = other.NumNodes;
        NumFeatures = other.NumFeatures;
        HiddenDimension = other.HiddenDimension;
        NumGCNLayers = other.NumGCNLayers;
        NumTemporalLayers = other.NumTemporalLayers;
        ChebyshevOrder = other.ChebyshevOrder;
        TemporalCellType = other.TemporalCellType;
        DropoutRate = other.DropoutRate;
        UseResidualConnections = other.UseResidualConnections;
        UseBatchNormalization = other.UseBatchNormalization;
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
    /// <para><b>For Beginners:</b> How many locations/entities in the spatial network.
    /// For traffic forecasting, this is the number of sensors/intersections.
    /// For social networks, this could be users or communities.
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
    /// Gets or sets the hidden dimension for GCN and temporal layers.
    /// </summary>
    /// <value>The hidden dimension, defaulting to 64.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The internal representation size.
    /// Larger values capture more complex patterns but need more computation.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of GCN layers.
    /// </summary>
    /// <value>The number of GCN layers, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many graph convolution layers to stack.
    /// Each layer aggregates information from k-hop neighbors (k = layer depth).
    /// 2 layers = direct neighbors + neighbors of neighbors.
    /// </para>
    /// </remarks>
    public int NumGCNLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of temporal (recurrent) layers.
    /// </summary>
    /// <value>The number of temporal layers, defaulting to 1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many recurrent layers for temporal modeling.
    /// Usually 1-2 is sufficient; more layers increase capacity but risk overfitting.
    /// </para>
    /// </remarks>
    public int NumTemporalLayers { get; set; } = 1;

    /// <summary>
    /// Gets or sets the Chebyshev polynomial order for graph convolution.
    /// </summary>
    /// <value>The Chebyshev order, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Order of Chebyshev polynomials for approximating
    /// graph convolution. Higher order captures longer-range spatial dependencies
    /// but increases computation. Order K means aggregating from K-hop neighbors.
    /// </para>
    /// </remarks>
    public int ChebyshevOrder { get; set; } = 2;

    /// <summary>
    /// Gets or sets the type of recurrent cell for temporal modeling.
    /// </summary>
    /// <value>The temporal cell type, defaulting to "gru".</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Which recurrent architecture to use:
    /// - "gru": Gated Recurrent Unit (simpler, faster, often sufficient)
    /// - "lstm": Long Short-Term Memory (more capacity, handles longer sequences)
    /// - "rnn": Simple RNN (basic, may struggle with long sequences)
    /// </para>
    /// </remarks>
    public string TemporalCellType { get; set; } = "gru";

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.3.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Prevents overfitting by randomly dropping connections
    /// during training. Higher values = more regularization but slower learning.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets whether to use residual (skip) connections.
    /// </summary>
    /// <value>True to use residual connections; false otherwise. Default: true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Skip connections add the input directly to the output,
    /// helping gradients flow through deep networks and preserving original signal.
    /// </para>
    /// </remarks>
    public bool UseResidualConnections { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use batch normalization.
    /// </summary>
    /// <value>True to use batch normalization; false otherwise. Default: true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Batch normalization stabilizes training by normalizing
    /// layer outputs. Helps train faster and reduces sensitivity to initialization.
    /// </para>
    /// </remarks>
    public bool UseBatchNormalization { get; set; } = true;

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
