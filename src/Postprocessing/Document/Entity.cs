namespace AiDotNet.Postprocessing.Document;

/// <summary>
/// Represents an extracted entity from document text.
/// </summary>
public class Entity
{
    /// <summary>
    /// The original text of the entity.
    /// </summary>
    public string Text { get; set; } = "";

    /// <summary>
    /// The type of entity.
    /// </summary>
    public EntityType Type { get; set; }

    /// <summary>
    /// Start index in the original text.
    /// </summary>
    public int StartIndex { get; set; }

    /// <summary>
    /// End index in the original text.
    /// </summary>
    public int EndIndex { get; set; }

    /// <summary>
    /// Confidence score for the extraction.
    /// </summary>
    public double Confidence { get; set; }

    /// <summary>
    /// Normalized value (for dates, money, etc.).
    /// </summary>
    public string? NormalizedValue { get; set; }

    /// <summary>
    /// Canonical name for the entity.
    /// </summary>
    public string? CanonicalName { get; set; }

    /// <summary>
    /// Linked entity from knowledge base.
    /// </summary>
    public Entity? LinkedEntity { get; set; }

    /// <summary>
    /// Additional attributes for the entity.
    /// </summary>
    public Dictionary<string, object> Attributes { get; set; } = new();
}
