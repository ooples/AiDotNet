using AiDotNet.LinearAlgebra;

namespace AiDotNet.Data.Structures;

/// <summary>
/// Represents a single graph with nodes, edges, features, and optional labels.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// GraphData encapsulates all information about a graph structure including:
/// - Node features (attributes for each node)
/// - Edge indices (connections between nodes)
/// - Edge features (optional attributes for edges)
/// - Adjacency matrix (graph structure in matrix form)
/// - Labels (for supervised learning tasks)
/// </para>
/// <para><b>For Beginners:</b> Think of a graph as a social network:
/// - **Nodes**: People in the network
/// - **Edges**: Friendships or connections between people
/// - **Node Features**: Each person's attributes (age, interests, etc.)
/// - **Edge Features**: Relationship attributes (how long they've been friends, interaction frequency)
/// - **Labels**: What we want to predict (e.g., will this person like a product?)
///
/// This class packages all this information together for graph neural network training.
/// </para>
/// </remarks>
public class GraphData<T>
{
    /// <summary>
    /// Node feature matrix of shape [num_nodes, num_features].
    /// </summary>
    /// <remarks>
    /// Each row represents one node's feature vector. For example, in a molecular graph,
    /// features might include atom type, charge, hybridization, etc.
    /// </remarks>
    public Tensor<T> NodeFeatures { get; set; } = new Tensor<T>([0, 0]);

    /// <summary>
    /// Edge index tensor of shape [2, num_edges] or [num_edges, 2].
    /// Format: [source_nodes; target_nodes] or [[src, tgt], [src, tgt], ...].
    /// </summary>
    /// <remarks>
    /// <para>
    /// Stores graph connectivity in COO (Coordinate) format. Each edge is represented by
    /// a (source, target) pair of node indices.
    /// </para>
    /// <para><b>For Beginners:</b> If node 0 connects to node 1, and node 1 connects to node 2:
    /// EdgeIndex = [[0, 1], [1, 2]] or transposed as [[0, 1], [1, 2]]
    /// This is a compact way to store which nodes are connected.
    /// </para>
    /// </remarks>
    public Tensor<T> EdgeIndex { get; set; } = new Tensor<T>([0, 2]);

    /// <summary>
    /// Optional edge feature matrix of shape [num_edges, num_edge_features].
    /// </summary>
    /// <remarks>
    /// Each row contains features for one edge. In molecular graphs, this could be
    /// bond type, bond length, stereochemistry, etc.
    /// </remarks>
    public Tensor<T>? EdgeFeatures { get; set; }

    /// <summary>
    /// Adjacency matrix of shape [num_nodes, num_nodes] or [batch_size, num_nodes, num_nodes].
    /// </summary>
    /// <remarks>
    /// Square matrix where A[i,j] = 1 if edge exists from node i to j, 0 otherwise.
    /// Can be weighted for graphs with edge weights.
    /// </remarks>
    public Tensor<T>? AdjacencyMatrix { get; set; }

    /// <summary>
    /// Node labels for node-level tasks (e.g., node classification).
    /// Shape: [num_nodes] or [num_nodes, num_classes].
    /// </summary>
    public Tensor<T>? NodeLabels { get; set; }

    /// <summary>
    /// Graph-level label for graph-level tasks (e.g., graph classification).
    /// Shape: [1] or [num_classes].
    /// </summary>
    public Tensor<T>? GraphLabel { get; set; }

    /// <summary>
    /// Mask indicating which nodes are in the training set.
    /// </summary>
    public Tensor<T>? TrainMask { get; set; }

    /// <summary>
    /// Mask indicating which nodes are in the validation set.
    /// </summary>
    public Tensor<T>? ValMask { get; set; }

    /// <summary>
    /// Mask indicating which nodes are in the test set.
    /// </summary>
    public Tensor<T>? TestMask { get; set; }

    /// <summary>
    /// Number of nodes in the graph.
    /// </summary>
    public int NumNodes => NodeFeatures.Shape[0];

    /// <summary>
    /// Number of edges in the graph.
    /// Handles both [2, num_edges] and [num_edges, 2] EdgeIndex formats.
    /// </summary>
    public int NumEdges
    {
        get
        {
            if (EdgeIndex.Shape.Length < 2) return 0;
            // If shape is [2, X], then X is num_edges (COO format with source/target rows)
            // If shape is [X, 2], then X is num_edges (edge list format with [src, tgt] pairs)
            return EdgeIndex.Shape[0] == 2 ? EdgeIndex.Shape[1] : EdgeIndex.Shape[0];
        }
    }

    /// <summary>
    /// Number of node features.
    /// </summary>
    public int NumNodeFeatures => NodeFeatures.Shape.Length > 1 ? NodeFeatures.Shape[1] : 0;

    /// <summary>
    /// Number of edge features (0 if no edge features).
    /// </summary>
    public int NumEdgeFeatures => EdgeFeatures?.Shape[1] ?? 0;

    /// <summary>
    /// Metadata for heterogeneous graphs (optional).
    /// </summary>
    public Dictionary<string, object>? Metadata { get; set; }
}
