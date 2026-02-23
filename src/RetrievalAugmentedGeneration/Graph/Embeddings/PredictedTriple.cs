namespace AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings;

/// <summary>
/// Represents a predicted (head, relation, tail) triple with its plausibility score.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When the link predictor finds possible missing facts,
/// each prediction includes:
/// - The head entity, relation, and tail entity forming the predicted fact
/// - A score indicating how plausible the fact is
/// - A confidence value normalized between 0 and 1
/// </para>
/// </remarks>
public class PredictedTriple
{
    /// <summary>
    /// The head (source) entity ID.
    /// </summary>
    public string HeadId { get; set; } = string.Empty;

    /// <summary>
    /// The relation type connecting head to tail.
    /// </summary>
    public string RelationType { get; set; } = string.Empty;

    /// <summary>
    /// The tail (target) entity ID.
    /// </summary>
    public string TailId { get; set; } = string.Empty;

    /// <summary>
    /// Raw plausibility score from the embedding model.
    /// </summary>
    public double Score { get; set; }

    /// <summary>
    /// Normalized confidence value between 0 and 1.
    /// </summary>
    public double Confidence { get; set; }
}
