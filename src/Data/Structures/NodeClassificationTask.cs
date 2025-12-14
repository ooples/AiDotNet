using AiDotNet.LinearAlgebra;

namespace AiDotNet.Data.Structures;

/// <summary>
/// Represents a node classification task where the goal is to predict labels for individual nodes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// Node classification is a fundamental graph learning task where each node in a graph has a label,
/// and the goal is to predict labels for unlabeled nodes based on:
/// - Node features
/// - Graph structure (connections between nodes)
/// - Labels of neighboring nodes
/// </para>
/// <para><b>For Beginners:</b> Node classification is like categorizing people in a social network.
///
/// **Real-world examples:**
///
/// **Social Networks:**
/// - Nodes: Users
/// - Task: Predict user interests/communities
/// - How: Use profile features + friend connections
/// - Example: "Is this user interested in sports?"
///
/// **Citation Networks:**
/// - Nodes: Research papers
/// - Task: Classify paper topics
/// - How: Use paper abstracts + citation links
/// - Example: Papers citing each other often share topics
///
/// **Fraud Detection:**
/// - Nodes: Financial accounts
/// - Task: Detect fraudulent accounts
/// - How: Use transaction patterns + account relationships
/// - Example: Fraudsters often form connected clusters
///
/// **Key Insight:** Node classification leverages the graph structure. Connected nodes often
/// share similar properties (homophily), so a node's neighbors provide valuable information
/// for prediction.
/// </para>
/// </remarks>
public class NodeClassificationTask<T>
{
    /// <summary>
    /// The graph data containing nodes, edges, and features.
    /// </summary>
    /// <remarks>
    /// This is the complete graph structure. In semi-supervised node classification,
    /// some nodes have known labels (training set) and others don't (test set).
    /// </remarks>
    public GraphData<T> Graph { get; set; } = new GraphData<T>();

    /// <summary>
    /// Node labels for all nodes in the graph.
    /// Shape: [num_nodes] for single-label or [num_nodes, num_classes] for multi-label.
    /// </summary>
    /// <remarks>
    /// In semi-supervised settings, labels for test nodes are only used for evaluation,
    /// not during training.
    /// </remarks>
    public Tensor<T> Labels { get; set; } = new Tensor<T>([0]);

    /// <summary>
    /// Indices of nodes to use for training.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> In semi-supervised node classification, we typically have:
    /// - Small set of labeled nodes for training (5-20% of nodes)
    /// - Larger set of unlabeled nodes for testing
    ///
    /// This split simulates real-world scenarios where getting labels is expensive.
    /// For example, manually labeling research papers by topic requires expert knowledge.
    /// </para>
    /// </remarks>
    public int[] TrainIndices { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Indices of nodes to use for validation.
    /// </summary>
    public int[] ValIndices { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Indices of nodes to use for testing.
    /// </summary>
    public int[] TestIndices { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Number of classes in the classification task.
    /// </summary>
    public int NumClasses { get; set; }

    /// <summary>
    /// Whether this is a multi-label classification task.
    /// </summary>
    /// <remarks>
    /// - False: Each node has exactly one label (e.g., paper topic)
    /// - True: Each node can have multiple labels (e.g., user interests)
    /// </remarks>
    public bool IsMultiLabel { get; set; } = false;
}
