namespace AiDotNet.Agentic.Graph;

/// <summary>
/// A streamed update emitted after a single node finishes executing during a graph run: the node's name
/// and the graph state as it stands after that node.
/// </summary>
/// <typeparam name="TState">The graph's state type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> When you stream a graph run, you receive one of these each time a step
/// completes — letting you watch the state evolve node by node (for progress UIs, logging, or
/// debugging). The last update's <see cref="State"/> is the final result.
/// </para>
/// </remarks>
public sealed class GraphStepUpdate<TState>
{
    /// <summary>
    /// Initializes a new step update.
    /// </summary>
    /// <param name="nodeName">The name of the node that just executed.</param>
    /// <param name="state">The graph state after the node executed.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="nodeName"/> is <c>null</c>.</exception>
    public GraphStepUpdate(string nodeName, TState state)
    {
        Guard.NotNull(nodeName);
        NodeName = nodeName;
        State = state;
    }

    /// <summary>
    /// Gets the name of the node that just executed.
    /// </summary>
    public string NodeName { get; }

    /// <summary>
    /// Gets the graph state after the node executed.
    /// </summary>
    public TState State { get; }
}
