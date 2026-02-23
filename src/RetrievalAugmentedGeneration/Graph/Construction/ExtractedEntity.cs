namespace AiDotNet.RetrievalAugmentedGeneration.Graph.Construction;

/// <summary>
/// Represents an entity extracted from text during knowledge graph construction.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When building a knowledge graph from text, the first step is
/// identifying entities — the "things" mentioned in the text (people, places, organizations).
/// Each extracted entity has:
/// - Name: The text mention ("Albert Einstein")
/// - Label: The entity type ("PERSON", "ORGANIZATION", "LOCATION")
/// - Confidence: How sure we are this is a real entity (0.0 to 1.0)
/// - Offsets: Where in the text this entity was found
/// </para>
/// </remarks>
public class ExtractedEntity
{
    /// <summary>
    /// The entity's surface form (name as it appears in text).
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Entity type label (e.g., PERSON, ORGANIZATION, LOCATION, CONCEPT).
    /// </summary>
    public string Label { get; set; } = string.Empty;

    /// <summary>
    /// Confidence score for this extraction (0.0 to 1.0).
    /// </summary>
    public double Confidence { get; set; }

    /// <summary>
    /// Start character offset in the source text.
    /// </summary>
    public int StartOffset { get; set; }

    /// <summary>
    /// End character offset in the source text.
    /// </summary>
    public int EndOffset { get; set; }
}
