namespace AiDotNet.Agentic.Graph;

/// <summary>
/// Reserved node names recognized by the graph runtime.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A graph finishes when flow reaches the special <see cref="End"/> node.
/// You route to it with <c>AddEdge("lastNode", GraphSpecialNodes.End)</c> (or return it from a
/// conditional router) to say "we're done".
/// </para>
/// </remarks>
public static class GraphSpecialNodes
{
    /// <summary>
    /// The terminal node name. When flow reaches this node, the run completes and returns the current state.
    /// </summary>
    public const string End = "__end__";
}
