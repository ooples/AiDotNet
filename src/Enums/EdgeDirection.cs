namespace AiDotNet.Enums;

/// <summary>
/// Specifies the direction of edges to retrieve when querying a knowledge graph.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In a directed graph, edges have a direction - they go FROM one node TO another.
///
/// Think of it like Twitter follows:
/// - If Alice follows Bob, the edge goes FROM Alice TO Bob
/// - Alice has an OUTGOING edge (she's following someone)
/// - Bob has an INCOMING edge (someone is following him)
///
/// When querying relationships:
/// - Outgoing: "Who does this person follow?" (edges starting from this node)
/// - Incoming: "Who follows this person?" (edges pointing to this node)
/// - Both: "All connections" (both directions)
/// </para>
/// </remarks>
public enum EdgeDirection
{
    /// <summary>
    /// Retrieve only outgoing edges (edges where the specified node is the source).
    /// </summary>
    Outgoing,

    /// <summary>
    /// Retrieve only incoming edges (edges where the specified node is the target).
    /// </summary>
    Incoming,

    /// <summary>
    /// Retrieve both outgoing and incoming edges.
    /// </summary>
    Both
}
