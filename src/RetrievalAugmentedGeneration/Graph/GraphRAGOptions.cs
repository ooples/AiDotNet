using AiDotNet.Enums;
using AiDotNet.RetrievalAugmentedGeneration.Graph.Communities;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph;

/// <summary>
/// Configuration options for the enhanced GraphRAG retrieval system.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These settings control how GraphRAG queries the knowledge graph:
/// - Mode: Local (entity-focused), Global (community summaries), or Drift (global→local refinement)
/// - MaxHops: How far to traverse from matched entities in local mode
/// - CommunityDetection: Settings for the Leiden algorithm used in Global/Drift modes
/// </para>
/// </remarks>
public class GraphRAGOptions
{
    /// <summary>
    /// Retrieval mode. Default: Local.
    /// </summary>
    public GraphRAGMode? Mode { get; set; }

    /// <summary>
    /// Maximum traversal hops from matched entities in Local mode. Default: 2.
    /// </summary>
    public int? MaxHops { get; set; }

    /// <summary>
    /// Options for Leiden community detection (used in Global and Drift modes).
    /// </summary>
    public LeidenOptions? CommunityDetection { get; set; }

    /// <summary>
    /// Maximum iterations for DRIFT mode's local refinement phase. Default: 3.
    /// </summary>
    public int? DriftMaxIterations { get; set; }

    internal GraphRAGMode GetEffectiveMode() => Mode ?? GraphRAGMode.Local;
    internal int GetEffectiveMaxHops() => MaxHops ?? 2;
    internal int GetEffectiveDriftMaxIterations() => DriftMaxIterations ?? 3;
}
