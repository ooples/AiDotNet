namespace AiDotNet.InferenceOptimization.Core;

/// <summary>
/// Interface for an optimization graph that represents the structure of neural network operations.
/// The graph can be optimized through various passes for improved inference performance.
/// </summary>
/// <remarks>
/// <para>
/// IOptimizationGraph is the core interface for the middle-layer IR in our two-tier architecture.
/// It provides graph manipulation capabilities needed for optimization passes.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type (double, float, decimal)</typeparam>
public interface IOptimizationGraph<T> where T : struct
{
    /// <summary>
    /// All nodes in the optimization graph.
    /// </summary>
    List<OptimizationNode<T>> Nodes { get; }

    /// <summary>
    /// Input nodes of the graph.
    /// </summary>
    List<OptimizationNode<T>> InputNodes { get; }

    /// <summary>
    /// Output nodes of the graph.
    /// </summary>
    List<OptimizationNode<T>> OutputNodes { get; }

    /// <summary>
    /// Adds a new node to the graph.
    /// </summary>
    void AddNode(OptimizationNode<T> node);

    /// <summary>
    /// Removes a node from the graph.
    /// </summary>
    void RemoveNode(OptimizationNode<T> node);

    /// <summary>
    /// Finds a node by its ID.
    /// </summary>
    OptimizationNode<T>? FindNodeById(string id);

    /// <summary>
    /// Finds nodes by name.
    /// </summary>
    List<OptimizationNode<T>> FindNodesByName(string name);

    /// <summary>
    /// Gets nodes in topological order (inputs to outputs).
    /// </summary>
    List<OptimizationNode<T>> GetTopologicalOrder();

    /// <summary>
    /// Validates the graph structure.
    /// </summary>
    bool Validate();

    /// <summary>
    /// Creates a deep copy of the graph.
    /// </summary>
    IOptimizationGraph<T> Clone();

    /// <summary>
    /// Gets statistics about the graph.
    /// </summary>
    GraphStatistics GetStatistics();
}
