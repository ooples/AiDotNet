using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.RetrievalAugmentedGeneration.Graph.Communities;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph;

/// <summary>
/// Enhanced Graph-based RAG that integrates with <see cref="KnowledgeGraph{T}"/> and supports
/// Local, Global (Leiden community summaries), and DRIFT retrieval modes.
/// </summary>
/// <typeparam name="T">The numeric type used for graph operations.</typeparam>
/// <remarks>
/// <para>
/// Unlike the existing <see cref="AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns.GraphRAG{T}"/>,
/// this class works directly with the <see cref="KnowledgeGraph{T}"/> class (which delegates to IGraphStore),
/// rather than maintaining its own internal dictionary. It also adds Leiden-based community summarization
/// for global and DRIFT search modes.
/// </para>
/// <para><b>For Beginners:</b> EnhancedGraphRAG combines three retrieval strategies:
///
/// <b>Local Search:</b> Best for specific factual questions.
/// 1. Find entities in the graph matching the query
/// 2. Traverse their neighborhood (1-2 hops)
/// 3. Collect related entities and their connections as context
///
/// <b>Global Search:</b> Best for broad thematic questions ("What are all the research areas?").
/// 1. Pre-compute community summaries via Leiden algorithm
/// 2. Search community summaries for relevant communities
/// 3. Return community descriptions as context
///
/// <b>DRIFT Search:</b> Best for complex queries needing both breadth and depth.
/// 1. Start with Global search to identify relevant communities
/// 2. From top communities, pick key entities
/// 3. Run Local search from those entities to refine context
/// 4. Repeat refinement for N iterations
/// </para>
/// </remarks>
public class EnhancedGraphRAG<T>
{
    private readonly KnowledgeGraph<T> _graph;
    private readonly GraphRAGOptions _options;
    private CommunityIndex<T>? _communityIndex;
    private LeidenResult? _leidenResult;

    /// <summary>
    /// Gets the Leiden community detection result, if community detection has been run.
    /// </summary>
    public LeidenResult? CommunityStructure => _leidenResult;

    /// <summary>
    /// Creates a new EnhancedGraphRAG instance.
    /// </summary>
    /// <param name="graph">The knowledge graph to query.</param>
    /// <param name="options">Configuration options.</param>
    public EnhancedGraphRAG(KnowledgeGraph<T> graph, GraphRAGOptions? options = null)
    {
        _graph = graph ?? throw new ArgumentNullException(nameof(graph));
        _options = options ?? new GraphRAGOptions();
    }

    /// <summary>
    /// Builds the community index for Global and DRIFT search modes.
    /// Must be called before using Global or DRIFT mode.
    /// </summary>
    public void BuildCommunityIndex()
    {
        var detector = new LeidenCommunityDetector<T>();
        _leidenResult = detector.Detect(_graph, _options.CommunityDetection);

        _communityIndex = new CommunityIndex<T>();
        _communityIndex.Build(_graph, _leidenResult);
    }

    /// <summary>
    /// Retrieves context from the knowledge graph for a given query.
    /// </summary>
    /// <param name="query">The search query.</param>
    /// <param name="topK">Maximum number of context items to return.</param>
    /// <returns>List of context strings derived from graph traversal or community summaries.</returns>
    public List<string> Retrieve(string query, int topK = 10)
    {
        if (string.IsNullOrWhiteSpace(query))
            return [];

        return _options.GetEffectiveMode() switch
        {
            GraphRAGMode.Local => LocalSearch(query, topK),
            GraphRAGMode.Global => GlobalSearch(query, topK),
            GraphRAGMode.Drift => DriftSearch(query, topK),
            _ => LocalSearch(query, topK)
        };
    }

    /// <summary>
    /// Retrieves relevant graph nodes for a given query.
    /// </summary>
    /// <param name="query">The search query.</param>
    /// <param name="topK">Maximum number of nodes to return.</param>
    /// <returns>Matching graph nodes.</returns>
    public IEnumerable<GraphNode<T>> RetrieveNodes(string query, int topK = 10)
    {
        if (string.IsNullOrWhiteSpace(query))
            return [];

        var matchedNodes = _graph.FindRelatedNodes(query, topK);
        var results = new List<GraphNode<T>>();
        var visited = new HashSet<string>();
        int maxHops = _options.GetEffectiveMaxHops();

        foreach (var node in matchedNodes)
        {
            if (visited.Contains(node.Id)) continue;
            visited.Add(node.Id);
            results.Add(node);

            // Expand neighborhood
            foreach (var neighbor in _graph.BreadthFirstTraversal(node.Id, maxHops))
            {
                if (visited.Contains(neighbor.Id)) continue;
                visited.Add(neighbor.Id);
                results.Add(neighbor);

                if (results.Count >= topK) break;
            }

            if (results.Count >= topK) break;
        }

        return results.Take(topK);
    }

    private List<string> LocalSearch(string query, int topK)
    {
        var context = new List<string>();
        int maxHops = _options.GetEffectiveMaxHops();

        var matchedNodes = _graph.FindRelatedNodes(query, topK);
        var visited = new HashSet<string>();

        foreach (var node in matchedNodes)
        {
            if (visited.Contains(node.Id)) continue;
            visited.Add(node.Id);

            var nodeName = node.GetProperty<string>("name") ?? node.Id;
            context.Add($"Entity: {nodeName} (type: {node.Label})");

            // Traverse neighborhood
            foreach (var edge in _graph.GetOutgoingEdges(node.Id))
            {
                var target = _graph.GetNode(edge.TargetId);
                if (target == null) continue;
                var targetName = target.GetProperty<string>("name") ?? target.Id;
                context.Add($"  {nodeName} --[{edge.RelationType}]--> {targetName}");
            }

            foreach (var edge in _graph.GetIncomingEdges(node.Id))
            {
                var source = _graph.GetNode(edge.SourceId);
                if (source == null) continue;
                var sourceName = source.GetProperty<string>("name") ?? source.Id;
                context.Add($"  {sourceName} --[{edge.RelationType}]--> {nodeName}");
            }

            // Expand deeper hops
            if (maxHops > 1)
            {
                foreach (var neighbor in _graph.BreadthFirstTraversal(node.Id, maxHops).Skip(1).Take(5))
                {
                    if (visited.Contains(neighbor.Id)) continue;
                    visited.Add(neighbor.Id);
                    var name = neighbor.GetProperty<string>("name") ?? neighbor.Id;
                    context.Add($"  Related: {name} (type: {neighbor.Label})");
                }
            }

            if (context.Count >= topK * 3) break;
        }

        return context.Take(topK * 3).ToList();
    }

    private List<string> GlobalSearch(string query, int topK)
    {
        if (_communityIndex == null)
        {
            throw new InvalidOperationException(
                "Community index not built. Call BuildCommunityIndex() before using Global or DRIFT mode.");
        }

        var relevantCommunities = _communityIndex.SearchCommunities(query, level: 0, topK: topK);
        return relevantCommunities.Select(c => c.Description).ToList();
    }

    private List<string> DriftSearch(string query, int topK)
    {
        if (_communityIndex == null)
        {
            throw new InvalidOperationException(
                "Community index not built. Call BuildCommunityIndex() before using Global or DRIFT mode.");
        }

        var context = new List<string>();
        int driftIterations = _options.GetEffectiveDriftMaxIterations();

        // Phase 1: Global search for relevant communities
        var communities = _communityIndex.SearchCommunities(query, level: 0, topK: 3).ToList();
        foreach (var community in communities)
        {
            context.Add($"[Community] {community.Description}");
        }

        // Phase 2: Iterative local refinement from community key entities
        var exploredEntities = new HashSet<string>();
        for (int iter = 0; iter < driftIterations; iter++)
        {
            var entitiesToExplore = communities
                .SelectMany(c => c.KeyEntities)
                .Where(e => !exploredEntities.Contains(e))
                .Take(3)
                .ToList();

            if (entitiesToExplore.Count == 0) break;

            foreach (var entityId in entitiesToExplore)
            {
                exploredEntities.Add(entityId);
                var node = _graph.GetNode(entityId);
                if (node == null) continue;

                var nodeName = node.GetProperty<string>("name") ?? node.Id;
                context.Add($"[Drill-down iter {iter + 1}] Entity: {nodeName} ({node.Label})");

                foreach (var edge in _graph.GetOutgoingEdges(entityId))
                {
                    var target = _graph.GetNode(edge.TargetId);
                    if (target == null) continue;
                    var targetName = target.GetProperty<string>("name") ?? target.Id;
                    context.Add($"  {nodeName} --[{edge.RelationType}]--> {targetName}");
                }
            }

            if (context.Count >= topK * 3) break;
        }

        return context.Take(topK * 3).ToList();
    }
}
