namespace AiDotNet.Enums;

/// <summary>
/// Defines how a link-prediction model scores a candidate edge from the two node embeddings at
/// its ends.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The model turns every node into a list of numbers. To guess whether two
/// nodes should be connected it has to turn that pair of lists into a single score. These are the
/// ways it can do that — multiply them together, measure the angle between them, combine them
/// element by element, or measure how far apart they are.
/// </para>
/// <para>
/// Promoted out of <c>LinkPredictionModel&lt;T&gt;</c>, where it was nested. A type nested in a
/// generic class is a distinct type per type argument, so the value could not be named on a
/// non-generic options class.
/// </para>
/// </remarks>
public enum LinkPredictionDecoder
{
    /// <summary>Dot product of the two embeddings. The usual choice, and the cheapest.</summary>
    DotProduct,

    /// <summary>Cosine similarity — the dot product normalised by length, so only direction counts.</summary>
    CosineSimilarity,

    /// <summary>Element-wise (Hadamard) product, fed to a learned scoring layer.</summary>
    Hadamard,

    /// <summary>Distance between the embeddings, so nearby nodes score as likely neighbours.</summary>
    Distance
}
