namespace AiDotNet.InferenceOptimization.Core;

/// <summary>
/// Interface for a computation graph that represents the structure of neural network operations.
/// The graph can be optimized through various passes for improved inference performance.
/// </summary>
/// <typeparam name="T">The numeric type (double, float, decimal)</typeparam>
public interface IComputationGraph<T> where T : struct
{
    /// <summary>
    /// All nodes in the computation graph.
    /// </summary>
    List<ComputationNode<T>> Nodes { get; }

    /// <summary>
    /// Input nodes of the graph.
    /// </summary>
    List<ComputationNode<T>> InputNodes { get; }

    /// <summary>
    /// Output nodes of the graph.
    /// </summary>
    List<ComputationNode<T>> OutputNodes { get; }

    /// <summary>
    /// Adds a new node to the graph.
    /// </summary>
    void AddNode(ComputationNode<T> node);

    /// <summary>
    /// Removes a node from the graph.
    /// </summary>
    void RemoveNode(ComputationNode<T> node);

    /// <summary>
    /// Finds a node by its ID.
    /// </summary>
    ComputationNode<T>? FindNodeById(string id);

    /// <summary>
    /// Finds nodes by name.
    /// </summary>
    List<ComputationNode<T>> FindNodesByName(string name);

    /// <summary>
    /// Gets nodes in topological order (inputs to outputs).
    /// </summary>
    List<ComputationNode<T>> GetTopologicalOrder();

    /// <summary>
    /// Validates the graph structure.
    /// </summary>
    bool Validate();

    /// <summary>
    /// Creates a deep copy of the graph.
    /// </summary>
    IComputationGraph<T> Clone();

    /// <summary>
    /// Gets statistics about the graph.
    /// </summary>
    GraphStatistics GetStatistics();
}
