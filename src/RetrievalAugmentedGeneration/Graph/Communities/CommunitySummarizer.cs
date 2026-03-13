using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph.Communities;

/// <summary>
/// Generates structured summaries for detected communities in a knowledge graph.
/// </summary>
/// <typeparam name="T">The numeric type used for graph operations.</typeparam>
/// <remarks>
/// <para>
/// For each community, the summarizer collects member entities, identifies central entities
/// by degree centrality, finds dominant relation types, and generates a structured description.
/// </para>
/// <para><b>For Beginners:</b> After finding communities, this class describes what each one is about.
/// For example, a community containing "Einstein", "Bohr", "Planck" connected by "collaborated_with"
/// and "influenced" relations might be summarized as:
/// "Physics pioneers community: 3 entities centered around Einstein, with key relations: collaborated_with, influenced"
/// </para>
/// </remarks>
public class CommunitySummarizer<T>
{
    /// <summary>
    /// Generates summaries for all communities in a Leiden result.
    /// </summary>
    /// <param name="graph">The knowledge graph.</param>
    /// <param name="leidenResult">The community detection result.</param>
    /// <param name="maxKeyEntities">Maximum number of key entities per community summary.</param>
    /// <param name="maxKeyRelations">Maximum number of key relations per community summary.</param>
    /// <returns>List of community summaries.</returns>
    public List<CommunitySummary> Summarize(
        KnowledgeGraph<T> graph,
        LeidenResult leidenResult,
        int maxKeyEntities = 5,
        int maxKeyRelations = 5)
    {
        if (graph == null) throw new ArgumentNullException(nameof(graph));
        if (leidenResult == null) throw new ArgumentNullException(nameof(leidenResult));

        return SummarizePartition(graph, leidenResult.Communities, 0, maxKeyEntities, maxKeyRelations);
    }

    /// <summary>
    /// Generates summaries for a specific partition (community assignment) at a given hierarchy level.
    /// </summary>
    /// <param name="graph">The knowledge graph.</param>
    /// <param name="partition">Mapping from original node IDs to community IDs.</param>
    /// <param name="level">The hierarchy level this partition represents.</param>
    /// <param name="maxKeyEntities">Maximum number of key entities per community summary.</param>
    /// <param name="maxKeyRelations">Maximum number of key relations per community summary.</param>
    /// <returns>List of community summaries for this level.</returns>
    public List<CommunitySummary> SummarizePartition(
        KnowledgeGraph<T> graph,
        Dictionary<string, int> partition,
        int level,
        int maxKeyEntities = 5,
        int maxKeyRelations = 5)
    {
        if (graph == null) throw new ArgumentNullException(nameof(graph));
        if (partition == null) throw new ArgumentNullException(nameof(partition));

        var summaries = new List<CommunitySummary>();

        // Group nodes by community
        var communityMembers = new Dictionary<int, List<string>>();
        foreach (var (nodeId, communityId) in partition)
        {
            if (!communityMembers.ContainsKey(communityId))
                communityMembers[communityId] = [];
            communityMembers[communityId].Add(nodeId);
        }

        foreach (var (communityId, members) in communityMembers)
        {
            var memberSet = new HashSet<string>(members);

            // Compute degree centrality within the community
            var degreeCounts = new Dictionary<string, int>();
            var relationCounts = new Dictionary<string, int>();

            foreach (var nodeId in members)
            {
                degreeCounts[nodeId] = 0;
                foreach (var edge in graph.GetOutgoingEdges(nodeId))
                {
                    if (memberSet.Contains(edge.TargetId))
                    {
                        degreeCounts[nodeId]++;
                        relationCounts.TryGetValue(edge.RelationType, out int rc);
                        relationCounts[edge.RelationType] = rc + 1;
                    }
                }

                foreach (var edge in graph.GetIncomingEdges(nodeId))
                {
                    if (memberSet.Contains(edge.SourceId))
                    {
                        degreeCounts[nodeId]++;
                    }
                }
            }

            var keyEntities = degreeCounts
                .OrderByDescending(kv => kv.Value)
                .Take(maxKeyEntities)
                .Select(kv => kv.Key)
                .ToList();

            var keyRelations = relationCounts
                .OrderByDescending(kv => kv.Value)
                .Take(maxKeyRelations)
                .Select(kv => kv.Key)
                .ToList();

            // Build description from node labels and key entities
            var labelCounts = new Dictionary<string, int>();
            foreach (var nodeId in members)
            {
                var node = graph.GetNode(nodeId);
                if (node == null) continue;
                labelCounts.TryGetValue(node.Label, out int lc);
                labelCounts[node.Label] = lc + 1;
            }

            var dominantLabel = labelCounts.OrderByDescending(kv => kv.Value).FirstOrDefault().Key ?? "mixed";
            var keyEntityNames = keyEntities.Select(id =>
            {
                var node = graph.GetNode(id);
                return node?.GetProperty<string>("name") ?? id;
            });

            string description = $"Community of {members.Count} entities (primarily {dominantLabel}), " +
                                 $"centered around: {string.Join(", ", keyEntityNames)}. " +
                                 (keyRelations.Count > 0
                                     ? $"Key relations: {string.Join(", ", keyRelations)}."
                                     : "No internal relations.");

            summaries.Add(new CommunitySummary
            {
                CommunityId = communityId,
                EntityIds = members,
                KeyEntities = keyEntities,
                KeyRelations = keyRelations,
                Description = description,
                Level = level
            });
        }

        return summaries;
    }
}
