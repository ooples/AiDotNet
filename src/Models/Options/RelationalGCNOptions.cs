using System;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for RelationalGCN (Relational Graph Convolutional Network).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// RelationalGCN extends Graph Convolutional Networks to handle multi-relational data
/// where different types of edges (relations) exist between nodes.
/// </para>
/// <para><b>For Beginners:</b> RelationalGCN is designed for knowledge graphs and multi-relational data:
///
/// <b>The Key Insight:</b>
/// Standard GCN treats all edges equally, but in financial networks, different types of
/// relationships matter differently. A "supplier-of" relationship is different from a
/// "competitor-of" relationship. R-GCN learns separate transformations for each relation type.
///
/// <b>What Problems Does RelationalGCN Solve?</b>
/// - Entity classification in knowledge graphs (company type, sector classification)
/// - Link prediction in multi-relational networks (predicting missing relationships)
/// - Financial network analysis with multiple relationship types
/// - Supply chain modeling with different connection types
///
/// <b>How RelationalGCN Works:</b>
/// 1. <b>Relation-Specific Weights:</b> Learns different weights for each relation type
/// 2. <b>Basis Decomposition:</b> Efficiently shares parameters across relations
/// 3. <b>Block Decomposition:</b> Alternative parameter sharing using block-diagonal matrices
/// 4. <b>Self-Connections:</b> Special weight for node's own features
///
/// <b>RelationalGCN Architecture:</b>
/// - For each relation r: H^(l+1) = sum_r (A_r * H^(l) * W_r) where A_r is the adjacency for relation r
/// - Basis decomposition: W_r = sum_b (a_rb * B_b) with shared bases B
/// - Block decomposition: W_r = diag(W_r1, W_r2, ..., W_rB) with smaller matrices
///
/// <b>Key Benefits:</b>
/// - Handles heterogeneous graphs with multiple edge types
/// - Parameter efficient through basis or block decomposition
/// - Captures relation-specific patterns in the data
/// - Effective for both entity classification and link prediction
/// </para>
/// <para>
/// <b>Reference:</b> Schlichtkrull et al., "Modeling Relational Data with Graph Convolutional Networks", ESWC 2018.
/// https://arxiv.org/abs/1703.06103
/// </para>
/// </remarks>
public class RelationalGCNOptions<T> : TimeSeriesRegressionOptions<T>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="RelationalGCNOptions{T}"/> class with default values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a default RelationalGCN configuration suitable for
    /// multi-relational graph learning with basis decomposition for efficiency.
    /// </para>
    /// </remarks>
    public RelationalGCNOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public RelationalGCNOptions(RelationalGCNOptions<T> other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SequenceLength = other.SequenceLength;
        ForecastHorizon = other.ForecastHorizon;
        NumNodes = other.NumNodes;
        NumFeatures = other.NumFeatures;
        NumRelations = other.NumRelations;
        HiddenDimension = other.HiddenDimension;
        NumLayers = other.NumLayers;
        NumBases = other.NumBases;
        NumBlocks = other.NumBlocks;
        Regularization = other.Regularization;
        DropoutRate = other.DropoutRate;
        UseBasisDecomposition = other.UseBasisDecomposition;
        UseBlockDecomposition = other.UseBlockDecomposition;
        UseSelfLoop = other.UseSelfLoop;
        Aggregation = other.Aggregation;
        NumSamples = other.NumSamples;
    }

    /// <summary>
    /// Gets or sets the sequence length (input time steps).
    /// </summary>
    /// <value>The sequence length, defaulting to 12.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many past time steps to use as input when combining
    /// with temporal modeling.
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
    /// Gets or sets the number of nodes (entities) in the graph.
    /// </summary>
    /// <value>The number of nodes, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many entities in the knowledge graph.
    /// For example, number of companies, securities, or geographic regions.
    /// </para>
    /// </remarks>
    public int NumNodes { get; set; } = 100;

    /// <summary>
    /// Gets or sets the number of input features per node.
    /// </summary>
    /// <value>The number of features, defaulting to 16.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many attributes each entity has.
    /// Common features: market cap, sector encoding, recent returns, etc.
    /// </para>
    /// </remarks>
    public int NumFeatures { get; set; } = 16;

    /// <summary>
    /// Gets or sets the number of relation types.
    /// </summary>
    /// <value>The number of relations, defaulting to 10.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many different types of relationships exist.
    /// Examples: "supplies-to", "competes-with", "same-sector", "co-owned-by".
    /// </para>
    /// </remarks>
    public int NumRelations { get; set; } = 10;

    /// <summary>
    /// Gets or sets the hidden dimension for the model.
    /// </summary>
    /// <value>The hidden dimension, defaulting to 64.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The internal representation size.
    /// Each node's features are projected to this dimension.
    /// </para>
    /// </remarks>
    public int HiddenDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of R-GCN layers.
    /// </summary>
    /// <value>The number of layers, defaulting to 2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many layers of relational graph convolution.
    /// More layers capture longer-range multi-hop relationships.
    /// </para>
    /// </remarks>
    public int NumLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of bases for basis decomposition.
    /// </summary>
    /// <value>The number of bases, defaulting to 5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> For basis decomposition, relation weights are
    /// linear combinations of shared basis matrices. Fewer bases = more parameter sharing.
    /// If NumBases == NumRelations, there's no sharing (full weights per relation).
    /// </para>
    /// </remarks>
    public int NumBases { get; set; } = 5;

    /// <summary>
    /// Gets or sets the number of blocks for block decomposition.
    /// </summary>
    /// <value>The number of blocks, defaulting to 4.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> For block decomposition, weights are block-diagonal.
    /// More blocks = smaller individual weight matrices = fewer parameters.
    /// </para>
    /// </remarks>
    public int NumBlocks { get; set; } = 4;

    /// <summary>
    /// Gets or sets the regularization strength.
    /// </summary>
    /// <value>The regularization strength, defaulting to 0.01.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls overfitting by penalizing large weights.
    /// Higher values = simpler model but potentially underfitting.
    /// </para>
    /// </remarks>
    public double Regularization { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the dropout rate for regularization.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.2.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Prevents overfitting by randomly dropping connections.
    /// </para>
    /// </remarks>
    public double DropoutRate { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets whether to use basis decomposition.
    /// </summary>
    /// <value>True to use basis decomposition; false otherwise. Default: true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Basis decomposition reduces parameters by sharing
    /// basis matrices across relations. Recommended when you have many relation types.
    /// </para>
    /// </remarks>
    public bool UseBasisDecomposition { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use block decomposition.
    /// </summary>
    /// <value>True to use block decomposition; false otherwise. Default: false.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Alternative to basis decomposition. Uses block-diagonal
    /// weight matrices. Typically only one decomposition is used (basis OR block).
    /// </para>
    /// </remarks>
    public bool UseBlockDecomposition { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to add self-loops to the graph.
    /// </summary>
    /// <value>True to add self-loops; false otherwise. Default: true.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Self-loops allow nodes to consider their own features
    /// during message passing. Usually helpful for maintaining node identity.
    /// </para>
    /// </remarks>
    public bool UseSelfLoop { get; set; } = true;

    /// <summary>
    /// Gets or sets the aggregation method for neighbor messages.
    /// </summary>
    /// <value>The aggregation method, defaulting to "sum".</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> How to combine messages from neighbors.
    /// - "sum": Add all neighbor messages
    /// - "mean": Average neighbor messages (normalized)
    /// - "max": Take element-wise maximum
    /// </para>
    /// </remarks>
    public string Aggregation { get; set; } = "sum";

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
