using AiDotNet.LinearAlgebra;

namespace AiDotNet.Data.Abstractions;

/// <summary>
/// Represents a link prediction task where the goal is to predict missing or future edges.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// Link prediction aims to predict whether an edge should exist between two nodes based on:
/// - Node features
/// - Graph structure
/// - Edge patterns in the existing graph
/// </para>
/// <para><b>For Beginners:</b> Link prediction is like recommending friendships or connections.
///
/// **Real-world examples:**
///
/// **Social Networks:**
/// - Task: Friend recommendation
/// - Question: "Will these two users become friends?"
/// - How: Analyze mutual friends, shared interests, interaction patterns
/// - Example: "You may know..." suggestions on Facebook/LinkedIn
///
/// **E-commerce:**
/// - Task: Product recommendation
/// - Question: "Will this user purchase this product?"
/// - Graph: Users and products as nodes, purchases as edges
/// - How: Users with similar purchase history likely buy similar products
///
/// **Citation Networks:**
/// - Task: Predict future citations
/// - Question: "Will paper A cite paper B?"
/// - How: Analyze topic similarity, author connections, citation patterns
///
/// **Drug Discovery:**
/// - Task: Predict drug-target interactions
/// - Question: "Will this drug bind to this protein?"
/// - Graph: Drugs and proteins as nodes, known interactions as edges
///
/// **Key Techniques:**
/// - **Negative sampling**: Create non-existent edges as negative examples
/// - **Edge splitting**: Hide some edges during training, predict them at test time
/// - **Node pair scoring**: Learn to score how likely two nodes should connect
/// </para>
/// </remarks>
public class LinkPredictionTask<T>
{
    /// <summary>
    /// The graph data with edges potentially removed for training.
    /// </summary>
    /// <remarks>
    /// In link prediction, we typically remove a portion of edges from the graph and try
    /// to predict them. The graph here contains the training edges only.
    /// </remarks>
    public GraphData<T> Graph { get; set; } = new GraphData<T>();

    /// <summary>
    /// Positive edge examples (edges that exist) for training.
    /// Shape: [num_train_edges, 2] where each row is [source_node, target_node].
    /// </summary>
    /// <remarks>
    /// These are edges that exist in the original graph and should be predicted as positive.
    /// </remarks>
    public Tensor<T> TrainPosEdges { get; set; } = new Tensor<T>([0, 2]);

    /// <summary>
    /// Negative edge examples (edges that don't exist) for training.
    /// Shape: [num_train_neg_edges, 2].
    /// </summary>
    /// <remarks>
    /// <para>
    /// These are sampled node pairs that don't have edges. They serve as negative examples
    /// to teach the model what connections are unlikely.
    /// </para>
    /// <para><b>For Beginners:</b> Why do we need negative examples?
    ///
    /// Imagine teaching someone to recognize friends vs strangers:
    /// - Positive examples: "These people ARE friends" (existing edges)
    /// - Negative examples: "These people are NOT friends" (non-existing edges)
    ///
    /// Without negatives, the model might predict everyone is friends with everyone!
    /// Negative sampling creates a balanced training set.
    /// </para>
    /// </remarks>
    public Tensor<T> TrainNegEdges { get; set; } = new Tensor<T>([0, 2]);

    /// <summary>
    /// Positive edge examples for validation.
    /// Shape: [num_val_edges, 2].
    /// </summary>
    public Tensor<T> ValPosEdges { get; set; } = new Tensor<T>([0, 2]);

    /// <summary>
    /// Negative edge examples for validation.
    /// Shape: [num_val_neg_edges, 2].
    /// </summary>
    public Tensor<T> ValNegEdges { get; set; } = new Tensor<T>([0, 2]);

    /// <summary>
    /// Positive edge examples for testing.
    /// Shape: [num_test_edges, 2].
    /// </summary>
    public Tensor<T> TestPosEdges { get; set; } = new Tensor<T>([0, 2]);

    /// <summary>
    /// Negative edge examples for testing.
    /// Shape: [num_test_neg_edges, 2].
    /// </summary>
    public Tensor<T> TestNegEdges { get; set; } = new Tensor<T>([0, 2]);

    /// <summary>
    /// Ratio of negative to positive edges for sampling.
    /// </summary>
    /// <remarks>
    /// Typically 1.0 (balanced) but can be adjusted. Higher ratios make the task harder
    /// but can improve model robustness.
    /// </remarks>
    public double NegativeSamplingRatio { get; set; } = 1.0;

    /// <summary>
    /// Whether the graph is directed (default: false).
    /// </summary>
    /// <remarks>
    /// - Directed: Edge from A to B doesn't imply edge from B to A (e.g., Twitter follows)
    /// - Undirected: Edge is bidirectional (e.g., Facebook friendships)
    /// </remarks>
    public bool IsDirected { get; set; } = false;
}
