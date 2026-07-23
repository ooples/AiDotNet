using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

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
/// <para>
/// When an <see cref="IGenerator{T}"/> LLM text generator is injected, the community <b>Description</b>
/// is instead produced by prompting the model to write a natural-language report summarizing the
/// community's entities and relations (Microsoft GraphRAG "community reports"). If no generator is
/// injected, or if the model returns an empty response, the extractive template description is used
/// as a documented fallback.
/// </para>
/// </remarks>
[ComponentType(ComponentType.DocumentStore)]
[PipelineStage(PipelineStage.Indexing)]
public class CommunitySummarizer<T>
{
    /// <summary>
    /// Optional LLM text generator used to write community reports. When null, the extractive
    /// template description is used.
    /// </summary>
    private readonly IGenerator<T>? _generator;

    /// <summary>
    /// Initializes a new instance of the <see cref="CommunitySummarizer{T}"/> class using only the
    /// extractive (offline, no-LLM) description template.
    /// </summary>
    public CommunitySummarizer()
    {
        _generator = null;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CommunitySummarizer{T}"/> class with an optional
    /// LLM text generator. When <paramref name="generator"/> is non-null, community descriptions are
    /// generated as natural-language reports by the model; otherwise the extractive template is used.
    /// </summary>
    /// <param name="generator">The LLM text generator, or null to use the extractive fallback.</param>
    public CommunitySummarizer(IGenerator<T>? generator)
    {
        _generator = generator;
    }

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
            }).ToList();

            // Extractive template description (also used as the fallback when no LLM is present
            // or the LLM returns an empty report).
            string description = $"Community of {members.Count} entities (primarily {dominantLabel}), " +
                                 $"centered around: {string.Join(", ", keyEntityNames)}. " +
                                 (keyRelations.Count > 0
                                     ? $"Key relations: {string.Join(", ", keyRelations)}."
                                     : "No internal relations.");

            // LLM-based community report (Microsoft GraphRAG parity), with extractive fallback.
            if (_generator != null)
            {
                var report = TryGenerateReport(graph, members, keyEntityNames, keyRelations, dominantLabel);
                if (!string.IsNullOrWhiteSpace(report))
                    description = report!.Trim();
            }

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

    /// <summary>
    /// Attempts to generate a natural-language community report via the injected LLM generator.
    /// Returns null when no generator is present, the generator throws, or it returns empty text,
    /// signaling the caller to keep the extractive template description.
    /// </summary>
    private string? TryGenerateReport(
        KnowledgeGraph<T> graph,
        List<string> members,
        List<string> keyEntityNames,
        List<string> keyRelations,
        string dominantLabel)
    {
        if (_generator == null)
            return null;

        try
        {
            var prompt = BuildReportPrompt(graph, members, keyEntityNames, keyRelations, dominantLabel);
            var report = _generator.Generate(prompt);
            return string.IsNullOrWhiteSpace(report) ? null : report;
        }
        catch (Exception ex)
        {
            System.Diagnostics.Trace.TraceWarning(
                $"CommunitySummarizer: LLM report generation threw ({ex.GetType().Name}); using extractive fallback.");
            return null;
        }
    }

    /// <summary>
    /// Builds the community-report prompt describing the community's entities and internal relations.
    /// </summary>
    internal static string BuildReportPrompt(
        KnowledgeGraph<T> graph,
        List<string> members,
        List<string> keyEntityNames,
        List<string> keyRelations,
        string dominantLabel)
    {
        var memberSet = new HashSet<string>(members);
        var sb = new StringBuilder();
        sb.AppendLine("You are an analyst writing a concise report about a community of related entities");
        sb.AppendLine("in a knowledge graph. Summarize what this community is about, who/what the key");
        sb.AppendLine("entities are, and how they relate. Write 2-4 sentences of plain prose (no JSON).");
        sb.AppendLine();
        sb.AppendLine($"Community size: {members.Count} entities (predominant type: {dominantLabel}).");

        if (keyEntityNames.Count > 0)
            sb.AppendLine($"Key entities: {string.Join(", ", keyEntityNames)}.");

        if (keyRelations.Count > 0)
            sb.AppendLine($"Common relation types: {string.Join(", ", keyRelations)}.");

        // Include a sample of concrete relationships to ground the report.
        sb.AppendLine("Relationships:");
        int shown = 0;
        foreach (var nodeId in members)
        {
            if (shown >= 25) break;
            var sourceName = graph.GetNode(nodeId)?.GetProperty<string>("name") ?? nodeId;
            foreach (var edge in graph.GetOutgoingEdges(nodeId))
            {
                if (!memberSet.Contains(edge.TargetId)) continue;
                var targetName = graph.GetNode(edge.TargetId)?.GetProperty<string>("name") ?? edge.TargetId;
                sb.AppendLine($"- {sourceName} {edge.RelationType} {targetName}");
                if (++shown >= 25) break;
            }
        }

        sb.AppendLine();
        sb.AppendLine("Report:");
        return sb.ToString();
    }
}
