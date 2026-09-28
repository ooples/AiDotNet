namespace AiDotNet.Enums;

/// <summary>
/// Defines how a graph model combines its per-node representations into a single representation
/// for the whole graph.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A graph model produces one summary per node, but labelling the whole
/// graph needs a single summary. Pooling is the step that combines them. Averaging is the usual
/// choice and treats every node equally; the others emphasise the strongest signal, the total
/// amount of signal, or let the model learn which nodes matter.
/// </para>
/// <para>
/// Promoted out of <c>GraphClassificationModel&lt;T&gt;</c>, where it was nested. A type nested
/// in a generic class is a distinct type per type argument, so
/// <c>GraphClassificationModel&lt;double&gt;.GraphPooling</c> and the <c>&lt;float&gt;</c> one
/// were unrelated types and the value could not be named on a non-generic options class.
/// </para>
/// </remarks>
public enum GraphPooling
{
    /// <summary>Average all node embeddings. The usual choice.</summary>
    Mean,

    /// <summary>Take the maximum across all node embeddings.</summary>
    Max,

    /// <summary>Sum all node embeddings, so larger graphs produce larger values.</summary>
    Sum,

    /// <summary>Weighted average with learned attention over the nodes.</summary>
    Attention
}
