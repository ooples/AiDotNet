using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph.Communities;

/// <summary>
/// Index structure that maps hierarchy levels and community IDs to their summaries,
/// enabling efficient community-based retrieval for GraphRAG.
/// </summary>
/// <typeparam name="T">The numeric type used for graph operations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> The community index is like a table of contents for your knowledge graph.
/// Instead of searching every node, you can search community summaries first to quickly find
/// which part of the graph is relevant to your query.
/// </para>
/// </remarks>
public class CommunityIndex<T>
{
    private readonly Dictionary<int, Dictionary<int, CommunitySummary>> _index = [];

    /// <summary>
    /// Builds the index from a knowledge graph and Leiden result.
    /// </summary>
    /// <param name="graph">The knowledge graph.</param>
    /// <param name="leidenResult">Community detection results.</param>
    /// <param name="maxKeyEntities">Maximum key entities per summary.</param>
    /// <param name="maxKeyRelations">Maximum key relations per summary.</param>
    public void Build(KnowledgeGraph<T> graph, LeidenResult leidenResult,
        int maxKeyEntities = 5, int maxKeyRelations = 5)
    {
        if (graph == null) throw new ArgumentNullException(nameof(graph));
        if (leidenResult == null) throw new ArgumentNullException(nameof(leidenResult));

        _index.Clear();

        var summarizer = new CommunitySummarizer<T>();

        // Build summaries for each hierarchy level
        if (leidenResult.HierarchicalPartitions != null && leidenResult.HierarchicalPartitions.Count > 0)
        {
            for (int level = 0; level < leidenResult.HierarchicalPartitions.Count; level++)
            {
                var partition = leidenResult.HierarchicalPartitions[level];
                var summaries = summarizer.SummarizePartition(graph, partition, level, maxKeyEntities, maxKeyRelations);

                var levelDict = new Dictionary<int, CommunitySummary>();
                foreach (var summary in summaries)
                {
                    levelDict[summary.CommunityId] = summary;
                }

                _index[level] = levelDict;
            }
        }
        else
        {
            // Fallback: use Communities directly as level 0
            var summaries = summarizer.Summarize(graph, leidenResult, maxKeyEntities, maxKeyRelations);
            var levelDict = new Dictionary<int, CommunitySummary>();
            foreach (var summary in summaries)
            {
                levelDict[summary.CommunityId] = summary;
            }

            _index[0] = levelDict;
        }
    }

    /// <summary>
    /// Gets a community summary by level and community ID.
    /// </summary>
    /// <param name="level">Hierarchy level (0 = finest).</param>
    /// <param name="communityId">Community identifier.</param>
    /// <returns>The community summary, or null if not found.</returns>
    public CommunitySummary? GetSummary(int level, int communityId)
    {
        if (_index.TryGetValue(level, out var levelDict) &&
            levelDict.TryGetValue(communityId, out var summary))
        {
            return summary;
        }

        return null;
    }

    /// <summary>
    /// Gets all community summaries at a given level.
    /// </summary>
    /// <param name="level">Hierarchy level (0 = finest).</param>
    /// <returns>All summaries at the specified level.</returns>
    public IEnumerable<CommunitySummary> GetSummariesAtLevel(int level)
    {
        if (_index.TryGetValue(level, out var levelDict))
            return levelDict.Values;
        return [];
    }

    /// <summary>
    /// Searches community summaries for those relevant to a query string.
    /// </summary>
    /// <param name="query">The search query.</param>
    /// <param name="level">Hierarchy level to search (default: 0).</param>
    /// <param name="topK">Maximum number of communities to return.</param>
    /// <returns>Communities whose descriptions or key entities match the query.</returns>
    public IEnumerable<CommunitySummary> SearchCommunities(string query, int level = 0, int topK = 5)
    {
        if (string.IsNullOrWhiteSpace(query))
            return [];

        var queryLower = query.ToLowerInvariant();
        var queryTerms = queryLower.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

        return GetSummariesAtLevel(level)
            .Select(s => (summary: s, score: ComputeRelevanceScore(s, queryTerms)))
            .Where(x => x.score > 0)
            .OrderByDescending(x => x.score)
            .Take(topK)
            .Select(x => x.summary);
    }

    private static double ComputeRelevanceScore(CommunitySummary summary, string[] queryTerms)
    {
        double score = 0.0;
        var descLower = summary.Description.ToLowerInvariant();

        foreach (var term in queryTerms)
        {
            if (descLower.Contains(term)) score += 1.0;

            foreach (var entity in summary.KeyEntities)
            {
                if (entity.ToLowerInvariant().Contains(term)) score += 2.0;
            }

            foreach (var relation in summary.KeyRelations)
            {
                if (relation.ToLowerInvariant().Contains(term)) score += 1.5;
            }
        }

        return score;
    }
}
