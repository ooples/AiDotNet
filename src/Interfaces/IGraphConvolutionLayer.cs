namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for graph convolutional layers that process graph-structured data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// Graph convolutional layers process data that is organized as graphs (nodes and edges).
/// This interface extends the base layer interface with graph-specific functionality,
/// particularly the ability to work with adjacency matrices that define graph structure.
/// </para>
/// <para><b>For Beginners:</b> This interface defines what all graph layers must be able to do.
///
/// Graph layers are special because they work with data that has connections:
/// - Social networks (people connected to friends)
/// - Molecules (atoms connected by bonds)
/// - Transportation networks (cities connected by roads)
/// - Knowledge graphs (concepts connected by relationships)
///
/// The key difference from regular layers is that graph layers need to know
/// which nodes are connected to which other nodes. That's what the adjacency matrix provides.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("GraphConvolutionLayer")]
public interface IGraphConvolutionLayer<T> : ILayer<T>
{
    /// <summary>
    /// Sets the adjacency matrix that defines the graph structure.
    /// </summary>
    /// <param name="adjacencyMatrix">The adjacency matrix tensor representing node connections.</param>
    /// <remarks>
    /// <para>
    /// The adjacency matrix is a square matrix where element [i,j] indicates whether and how strongly
    /// node i is connected to node j. Common formats include:
    /// - Binary adjacency: 1 if connected, 0 otherwise
    /// - Weighted adjacency: connection strength as a value
    /// - Normalized adjacency: preprocessed for better training
    /// </para>
    /// <para><b>For Beginners:</b> This method tells the layer how nodes in the graph are connected.
    ///
    /// Think of the adjacency matrix as a map:
    /// - Each row represents a node
    /// - Each column represents a potential connection
    /// - The value at position [i,j] tells if node i connects to node j
    ///
    /// For example, in a social network:
    /// - adjacencyMatrix[Alice, Bob] = 1 means Alice is friends with Bob
    /// - adjacencyMatrix[Alice, Charlie] = 0 means Alice is not friends with Charlie
    ///
    /// This connectivity information is crucial for graph neural networks to propagate
    /// information between connected nodes.
    /// </para>
    /// </remarks>
    void SetAdjacencyMatrix(Tensor<T> adjacencyMatrix);

    /// <summary>
    /// Gets the adjacency matrix currently being used by this layer.
    /// </summary>
    /// <returns>The adjacency matrix tensor, or null if not set.</returns>
    /// <remarks>
    /// <para>
    /// This method retrieves the adjacency matrix that was set using SetAdjacencyMatrix.
    /// It may return null if the adjacency matrix has not been set yet.
    /// </para>
    /// <para><b>For Beginners:</b> This method lets you check what graph structure the layer is using.
    ///
    /// This can be useful for:
    /// - Verifying the correct graph was loaded
    /// - Debugging graph connectivity issues
    /// - Visualizing the graph structure
    /// </para>
    /// </remarks>
    Tensor<T>? GetAdjacencyMatrix();

    /// <summary>
    /// Gets the number of input features per node.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property indicates how many features each node in the graph has as input.
    /// For example, in a molecular graph, this might be properties of each atom.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many pieces of information each node starts with.
    ///
    /// Examples:
    /// - In a social network: age, location, interests (3 features)
    /// - In a molecule: atomic number, charge, mass (3 features)
    /// - In a citation network: word embeddings (300 features)
    ///
    /// Each node has the same number of input features.
    /// </para>
    /// </remarks>
    int InputFeatures { get; }

    /// <summary>
    /// Gets the number of output features per node.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This property indicates how many features each node will have after processing through this layer.
    /// The layer transforms each node's input features into output features through learned transformations.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you how many pieces of information each node will have after processing.
    ///
    /// The layer learns to:
    /// - Combine input features in useful ways
    /// - Extract important patterns
    /// - Create new representations that are better for the task
    ///
    /// For example, if you start with 10 features per node and the layer has 16 output features,
    /// each node's 10 numbers will be transformed into 16 numbers that hopefully capture
    /// more useful information for your specific task.
    /// </para>
    /// </remarks>
    int OutputFeatures { get; }
}
