using AiDotNet.Data.Structures;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for data loaders that provide graph-structured data for graph neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This interface is for loading graph-structured data where:
/// - Nodes have features (attributes for each entity)
/// - Edges define connections between nodes
/// - Labels can be per-node (node classification) or per-graph (graph classification)
/// </para>
/// <para><b>For Beginners:</b> Graphs represent relationships between things:
///
/// **Example: Social Network**
/// - Nodes: People
/// - Edges: Friendships
/// - Node Features: Age, interests, location
/// - Task: Predict user interests based on their friends
///
/// **Example: Molecular Structure**
/// - Nodes: Atoms
/// - Edges: Chemical bonds
/// - Node Features: Atom type, charge
/// - Task: Predict molecular properties (toxicity, activity)
///
/// The adjacency matrix tells the GNN which nodes are connected so it can
/// aggregate information from neighbors during message passing.
/// </para>
/// </remarks>
public interface IGraphDataLoader<T>
{
    /// <summary>
    /// Gets the number of graphs in the dataset (1 for single-graph datasets like citation networks).
    /// </summary>
    int NumGraphs { get; }

    /// <summary>
    /// Gets the batch size (number of graphs per batch).
    /// </summary>
    int BatchSize { get; }

    /// <summary>
    /// Gets whether the data loader has more batches available.
    /// </summary>
    bool HasNext { get; }

    /// <summary>
    /// Loads and returns the next graph or batch of graphs.
    /// </summary>
    /// <returns>A GraphData instance containing the loaded graph(s).</returns>
    GraphData<T> GetNextBatch();

    /// <summary>
    /// Resets the data loader to the beginning of the dataset.
    /// </summary>
    void Reset();
}

/// <summary>
/// Extended interface for graph data loaders with full IDataLoader composition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This interface extends the base IGraphDataLoader with additional capabilities:
/// - Async loading and unloading
/// - Progress tracking through ICountable
/// - Node/edge/graph property accessors
/// - Task creation methods for different graph learning tasks
/// </para>
/// <para><b>For Beginners:</b> Use this interface when you need:
/// - Full lifecycle management (LoadAsync/Unload)
/// - Direct access to graph properties (NodeFeatures, AdjacencyMatrix, etc.)
/// - Built-in task creation (node classification, graph classification, link prediction)
/// </para>
/// </remarks>
public interface IGraphDataLoaderEx<T> : IGraphDataLoader<T>, IDataLoader<T>, IBatchIterable<GraphData<T>>
{
    /// <summary>
    /// Gets the node feature tensor of shape [numNodes, numFeatures].
    /// </summary>
    Tensor<T> NodeFeatures { get; }

    /// <summary>
    /// Gets the adjacency matrix of shape [numNodes, numNodes].
    /// </summary>
    Tensor<T> AdjacencyMatrix { get; }

    /// <summary>
    /// Gets the edge index tensor in COO format [numEdges, 2].
    /// </summary>
    Tensor<T> EdgeIndex { get; }

    /// <summary>
    /// Gets node labels for node classification tasks, or null if not available.
    /// </summary>
    Tensor<T>? NodeLabels { get; }

    /// <summary>
    /// Gets graph labels for graph classification tasks, or null if not available.
    /// </summary>
    Tensor<T>? GraphLabels { get; }

    /// <summary>
    /// Gets the number of node features.
    /// </summary>
    int NumNodeFeatures { get; }

    /// <summary>
    /// Gets the number of nodes in the graph (or total across all graphs).
    /// </summary>
    int NumNodes { get; }

    /// <summary>
    /// Gets the number of edges in the graph (or total across all graphs).
    /// </summary>
    int NumEdges { get; }

    /// <summary>
    /// Gets the number of classes for classification tasks.
    /// </summary>
    int NumClasses { get; }

    /// <summary>
    /// Creates a node classification task with train/val/test split.
    /// </summary>
    NodeClassificationTask<T> CreateNodeClassificationTask(
        double trainRatio = 0.1,
        double valRatio = 0.1,
        int? seed = null);

    /// <summary>
    /// Creates a graph classification task for datasets with multiple graphs.
    /// </summary>
    GraphClassificationTask<T> CreateGraphClassificationTask(
        double trainRatio = 0.8,
        double valRatio = 0.1,
        int? seed = null);

    /// <summary>
    /// Creates a link prediction task for predicting missing edges.
    /// </summary>
    LinkPredictionTask<T> CreateLinkPredictionTask(
        double trainRatio = 0.85,
        double negativeRatio = 1.0,
        int? seed = null);
}
