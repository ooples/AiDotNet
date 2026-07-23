using System.Collections.Generic;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph.Communities;

/// <summary>
/// Represents a summary of a detected community within a knowledge graph.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> After detecting communities, each one gets a summary describing:
/// - Which entities belong to it
/// - What the most important entities are (by connection count)
/// - What types of relationships dominate
/// - A human-readable description of what the community represents
/// </para>
/// </remarks>
public class CommunitySummary
{
    /// <summary>
    /// Unique identifier for this community.
    /// </summary>
    public int CommunityId { get; set; }

    /// <summary>
    /// IDs of all entities belonging to this community.
    /// </summary>
    public List<string> EntityIds { get; set; } = [];

    /// <summary>
    /// IDs of the most central/important entities in the community (by degree centrality).
    /// </summary>
    public List<string> KeyEntities { get; set; } = [];

    /// <summary>
    /// Most frequent relation types within the community.
    /// </summary>
    public List<string> KeyRelations { get; set; } = [];

    /// <summary>
    /// Structured text description of the community's content and themes.
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Hierarchy level at which this community was detected (0 = finest).
    /// </summary>
    public int Level { get; set; }
}
