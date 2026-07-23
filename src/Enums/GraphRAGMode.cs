namespace AiDotNet.Enums;

/// <summary>
/// Specifies the retrieval mode for enhanced GraphRAG.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These modes represent different strategies for querying a knowledge graph:
/// - Local: Start from specific entities and explore their neighborhoods (best for specific questions)
/// - Global: Use community summaries for broad, thematic answers (best for "what are all X?" questions)
/// - Drift: Start broad with community summaries, then iteratively refine locally (best for complex queries)
/// </para>
/// </remarks>
public enum GraphRAGMode
{
    /// <summary>
    /// Local search: entity-centric retrieval via graph traversal from matched entities.
    /// </summary>
    Local,

    /// <summary>
    /// Global search: community-level retrieval using pre-computed community summaries.
    /// </summary>
    Global,

    /// <summary>
    /// DRIFT search: starts with global community summaries, then iteratively refines via local exploration.
    /// </summary>
    Drift
}
