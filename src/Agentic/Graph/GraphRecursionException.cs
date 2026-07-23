namespace AiDotNet.Agentic.Graph;

/// <summary>
/// Thrown when a graph run exceeds its configured maximum number of steps, which usually indicates a
/// cycle that never reaches the end node.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Graphs can loop (a node can route back to an earlier node). To stop an
/// accidental infinite loop, every run has a step budget. If the graph keeps going past that budget,
/// this exception is raised so you can fix the routing or raise the limit deliberately.
/// </para>
/// </remarks>
public sealed class GraphRecursionException : Exception
{
    /// <summary>
    /// Initializes a new instance of the <see cref="GraphRecursionException"/> class.
    /// </summary>
    /// <param name="maxSteps">The step budget that was exceeded.</param>
    public GraphRecursionException(int maxSteps)
        : base($"Graph exceeded its maximum of {maxSteps} steps without reaching the end node. " +
               "This usually means a cycle never terminates; check your conditional edges or raise GraphRunOptions.MaxSteps.")
    {
        MaxSteps = maxSteps;
    }

    /// <summary>
    /// Gets the step budget that was exceeded.
    /// </summary>
    public int MaxSteps { get; }
}
