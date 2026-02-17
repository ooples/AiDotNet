namespace AiDotNet.RetrievalAugmentedGeneration.Graph.Construction;

/// <summary>
/// Represents a relation extracted from text between two entities.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> After finding entities in text, the next step is finding relationships
/// between them. For example, in "Einstein worked at Princeton University":
/// - SourceEntity: "Einstein" (PERSON)
/// - TargetEntity: "Princeton University" (ORGANIZATION)
/// - RelationType: "WORKED_AT"
/// - Confidence: 0.85 (fairly confident based on the verb "worked at")
/// </para>
/// </remarks>
public class ExtractedRelation
{
    /// <summary>
    /// The source (subject) entity name.
    /// </summary>
    public string SourceEntity { get; set; } = string.Empty;

    /// <summary>
    /// The target (object) entity name.
    /// </summary>
    public string TargetEntity { get; set; } = string.Empty;

    /// <summary>
    /// The relation type (e.g., WORKS_AT, BORN_IN, PART_OF).
    /// </summary>
    public string RelationType { get; set; } = string.Empty;

    /// <summary>
    /// Confidence score for this relation extraction (0.0 to 1.0).
    /// </summary>
    public double Confidence { get; set; }
}
