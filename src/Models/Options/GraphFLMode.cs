namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies the federated graph learning task type.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Graphs can be analyzed at different levels of granularity.
/// This enum tells the system what kind of graph task each client is performing:</para>
/// <list type="bullet">
/// <item><description><b>SubgraphLevel:</b> Each client holds a subgraph of a larger graph and trains a GNN on it.
/// The server aggregates GNN parameters. Most common for social/transaction networks.</description></item>
/// <item><description><b>NodeLevel:</b> Each client contributes node features/labels for node classification.</description></item>
/// <item><description><b>LinkPrediction:</b> Clients train to predict missing edges (recommendations, knowledge graphs).</description></item>
/// <item><description><b>GraphClassification:</b> Each client holds complete graphs (e.g., molecules) and classifies them.</description></item>
/// </list>
/// </remarks>
public enum GraphFLMode
{
    /// <summary>Each client holds a subgraph of a larger graph.</summary>
    SubgraphLevel,

    /// <summary>Node-level classification across federated subgraphs.</summary>
    NodeLevel,

    /// <summary>Federated link prediction (edge existence).</summary>
    LinkPrediction,

    /// <summary>Each client classifies complete graphs (e.g., molecular property prediction).</summary>
    GraphClassification
}
