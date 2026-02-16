namespace AiDotNet.RetrievalAugmentedGeneration.Graph.Construction;

/// <summary>
/// Configuration options for automated knowledge graph construction from text.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These settings control how the KG construction pipeline works:
/// - MaxChunkSize: How many characters per text chunk (smaller = more precise entity extraction)
/// - ChunkOverlap: Overlap between chunks to avoid missing entities at boundaries
/// - EntityConfidenceThreshold: Minimum confidence to accept an extracted entity
/// - EnableEntityResolution: Whether to merge similar entity names (e.g., "Einstein" and "A. Einstein")
/// - EntitySimilarityThreshold: How similar two entity names must be to merge them
/// </para>
/// </remarks>
public class KGConstructionOptions
{
    /// <summary>
    /// Maximum characters per text chunk for entity extraction. Default: 500.
    /// </summary>
    public int? MaxChunkSize { get; set; }

    /// <summary>
    /// Character overlap between adjacent chunks. Default: 50.
    /// </summary>
    public int? ChunkOverlap { get; set; }

    /// <summary>
    /// Minimum confidence threshold to accept an extracted entity. Default: 0.5.
    /// </summary>
    public double? EntityConfidenceThreshold { get; set; }

    /// <summary>
    /// Whether to merge similar entity names via string similarity. Default: true.
    /// </summary>
    public bool? EnableEntityResolution { get; set; }

    /// <summary>
    /// Minimum string similarity (0-1) for merging entity names. Default: 0.85.
    /// </summary>
    public double? EntitySimilarityThreshold { get; set; }

    internal int GetEffectiveMaxChunkSize() => MaxChunkSize ?? 500;
    internal int GetEffectiveChunkOverlap() => ChunkOverlap ?? 50;
    internal double GetEffectiveEntityConfidenceThreshold() => EntityConfidenceThreshold ?? 0.5;
    internal bool GetEffectiveEnableEntityResolution() => EnableEntityResolution ?? true;
    internal double GetEffectiveEntitySimilarityThreshold() => EntitySimilarityThreshold ?? 0.85;
}
