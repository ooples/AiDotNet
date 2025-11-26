using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using AiDotNet.Interfaces;

namespace AiDotNet.RetrievalAugmentedGeneration.Graph;

/// <summary>
/// Hybrid retriever that combines vector similarity search with graph traversal for enhanced RAG.
/// </summary>
/// <typeparam name="T">The numeric type used for vector operations.</typeparam>
/// <remarks>
/// <para>
/// This retriever uses a two-stage approach:
/// 1. Vector similarity search to find initial candidate nodes
/// 2. Graph traversal to expand context with related nodes
/// </para>
/// <para><b>For Beginners:</b> Traditional RAG uses only vector similarity:
///
/// Query: "What is photosynthesis?"
/// Traditional RAG:
/// - Find documents similar to the query
/// - Return top-k matches
/// - Misses related context!
///
/// Hybrid Graph RAG:
/// - Find initial matches via vector similarity
/// - Walk the graph to find related concepts
/// - Example: photosynthesis → chlorophyll → plants → carbon dioxide
/// - Provides richer, more complete context
///
/// Real-world analogy:
/// - Traditional: Search "Paris" → get Paris documents
/// - Hybrid: Search "Paris" → get Paris + France + Eiffel Tower + Seine River
/// - Graph connections provide context vectors can't capture!
/// </para>
/// </remarks>
public class HybridGraphRetriever<T> where T : struct, INumber<T>
{
    private readonly KnowledgeGraph<T> _graph;
    private readonly IVectorDatabase<T> _vectorDb;
    private readonly ISimilarityMetric<T> _similarityMetric;

    /// <summary>
    /// Initializes a new instance of the <see cref="HybridGraphRetriever{T}"/> class.
    /// </summary>
    /// <param name="graph">The knowledge graph containing entity relationships.</param>
    /// <param name="vectorDb">The vector database for similarity search.</param>
    /// <param name="similarityMetric">The similarity metric to use (e.g., cosine similarity).</param>
    public HybridGraphRetriever(
        KnowledgeGraph<T> graph,
        IVectorDatabase<T> vectorDb,
        ISimilarityMetric<T> similarityMetric)
    {
        _graph = graph ?? throw new ArgumentNullException(nameof(graph));
        _vectorDb = vectorDb ?? throw new ArgumentNullException(nameof(vectorDb));
        _similarityMetric = similarityMetric ?? throw new ArgumentNullException(nameof(similarityMetric));
    }

    /// <summary>
    /// Retrieves relevant nodes using hybrid vector + graph approach.
    /// </summary>
    /// <param name="queryEmbedding">The query embedding vector.</param>
    /// <param name="topK">Number of initial candidates to retrieve via vector search.</param>
    /// <param name="expansionDepth">How many hops to traverse in the graph (0 = no expansion).</param>
    /// <param name="maxResults">Maximum total results to return after expansion.</param>
    /// <returns>List of retrieved nodes with their relevance scores.</returns>
    public List<RetrievalResult<T>> Retrieve(
        T[] queryEmbedding,
        int topK = 5,
        int expansionDepth = 1,
        int maxResults = 10)
    {
        if (queryEmbedding == null || queryEmbedding.Length == 0)
            throw new ArgumentException("Query embedding cannot be null or empty", nameof(queryEmbedding));
        if (topK <= 0)
            throw new ArgumentOutOfRangeException(nameof(topK), "topK must be positive");
        if (expansionDepth < 0)
            throw new ArgumentOutOfRangeException(nameof(expansionDepth), "expansionDepth cannot be negative");

        // Stage 1: Vector similarity search for initial candidates
        var initialCandidates = _vectorDb.SearchSimilar(queryEmbedding, topK);

        if (expansionDepth == 0)
        {
            // No graph expansion - return vector results only
            return initialCandidates
                .Take(maxResults)
                .Select(r => new RetrievalResult<T>
                {
                    NodeId = r.Id,
                    Score = r.Similarity,
                    Source = RetrievalSource.VectorSearch,
                    Embedding = r.Embedding
                })
                .ToList();
        }

        // Stage 2: Graph expansion
        var results = new Dictionary<string, RetrievalResult<T>>();
        var visited = new HashSet<string>();

        // Add initial candidates
        foreach (var candidate in initialCandidates)
        {
            var result = new RetrievalResult<T>
            {
                NodeId = candidate.Id,
                Score = candidate.Similarity,
                Source = RetrievalSource.VectorSearch,
                Embedding = candidate.Embedding,
                Depth = 0
            };
            results[candidate.Id] = result;
            visited.Add(candidate.Id);
        }

        // BFS expansion from initial candidates
        var queue = new Queue<(string nodeId, int depth)>();
        foreach (var candidate in initialCandidates)
        {
            queue.Enqueue((candidate.Id, 0));
        }

        while (queue.Count > 0)
        {
            var (currentId, currentDepth) = queue.Dequeue();

            if (currentDepth >= expansionDepth)
                continue;

            // Get neighbors from graph
            var neighbors = GetNeighbors(currentId);

            foreach (var neighborId in neighbors)
            {
                if (visited.Contains(neighborId))
                    continue;

                visited.Add(neighborId);

                // Get neighbor's embedding if available
                var neighborEmbedding = _vectorDb.GetEmbedding(neighborId);
                double score = 0.0;

                if (neighborEmbedding != null)
                {
                    // Calculate similarity to query
                    score = CalculateSimilarity(queryEmbedding, neighborEmbedding);

                    // Apply depth penalty (closer nodes are more relevant)
                    var depthPenalty = Math.Pow(0.8, currentDepth + 1); // 0.8^depth
                    score *= depthPenalty;
                }

                var result = new RetrievalResult<T>
                {
                    NodeId = neighborId,
                    Score = score,
                    Source = RetrievalSource.GraphTraversal,
                    Embedding = neighborEmbedding,
                    Depth = currentDepth + 1,
                    ParentNodeId = currentId
                };

                results[neighborId] = result;

                // Continue expanding
                if (currentDepth + 1 < expansionDepth)
                {
                    queue.Enqueue((neighborId, currentDepth + 1));
                }
            }
        }

        // Return top results sorted by score
        return results.Values
            .OrderByDescending(r => r.Score)
            .Take(maxResults)
            .ToList();
    }

    /// <summary>
    /// Retrieves relevant nodes asynchronously using hybrid approach.
    /// </summary>
    public async Task<List<RetrievalResult<T>>> RetrieveAsync(
        T[] queryEmbedding,
        int topK = 5,
        int expansionDepth = 1,
        int maxResults = 10)
    {
        // For now, just wrap the synchronous version
        // In a real implementation, you'd use async vector DB operations
        return await Task.Run(() => Retrieve(queryEmbedding, topK, expansionDepth, maxResults));
    }

    /// <summary>
    /// Retrieves nodes with relationship-aware scoring.
    /// </summary>
    /// <param name="queryEmbedding">The query embedding vector.</param>
    /// <param name="topK">Number of initial candidates.</param>
    /// <param name="relationshipWeights">Weights for different relationship types.</param>
    /// <param name="maxResults">Maximum results to return.</param>
    /// <returns>List of retrieved nodes with relationship-aware scores.</returns>
    public List<RetrievalResult<T>> RetrieveWithRelationships(
        T[] queryEmbedding,
        int topK = 5,
        Dictionary<string, double>? relationshipWeights = null,
        int maxResults = 10)
    {
        if (queryEmbedding == null || queryEmbedding.Length == 0)
            throw new ArgumentException("Query embedding cannot be null or empty", nameof(queryEmbedding));

        relationshipWeights ??= new Dictionary<string, double>();

        // Stage 1: Vector similarity search
        var initialCandidates = _vectorDb.SearchSimilar(queryEmbedding, topK);
        var results = new Dictionary<string, RetrievalResult<T>>();

        // Add initial candidates
        foreach (var candidate in initialCandidates)
        {
            results[candidate.Id] = new RetrievalResult<T>
            {
                NodeId = candidate.Id,
                Score = candidate.Similarity,
                Source = RetrievalSource.VectorSearch,
                Embedding = candidate.Embedding,
                Depth = 0
            };
        }

        // Stage 2: Expand via relationships with weighted scoring
        foreach (var candidate in initialCandidates)
        {
            var node = _graph.GetNode(candidate.Id);
            if (node == null)
                continue;

            var outgoingEdges = _graph.GetOutgoingEdges(candidate.Id);

            foreach (var edge in outgoingEdges)
            {
                if (results.ContainsKey(edge.TargetId))
                    continue;

                // Get relationship weight (default to 1.0)
                var weight = relationshipWeights.TryGetValue(edge.RelationType, out var w) ? w : 1.0;

                // Get target node's embedding
                var targetEmbedding = _vectorDb.GetEmbedding(edge.TargetId);
                double score = 0.0;

                if (targetEmbedding != null)
                {
                    score = CalculateSimilarity(queryEmbedding, targetEmbedding);
                    score *= weight; // Apply relationship weight
                    score *= 0.8; // One-hop penalty
                }

                results[edge.TargetId] = new RetrievalResult<T>
                {
                    NodeId = edge.TargetId,
                    Score = score,
                    Source = RetrievalSource.GraphTraversal,
                    Embedding = targetEmbedding,
                    Depth = 1,
                    ParentNodeId = candidate.Id,
                    RelationType = edge.RelationType
                };
            }
        }

        // Return top results
        return results.Values
            .OrderByDescending(r => r.Score)
            .Take(maxResults)
            .ToList();
    }

    /// <summary>
    /// Gets all neighbors (both incoming and outgoing) of a node.
    /// </summary>
    private HashSet<string> GetNeighbors(string nodeId)
    {
        var neighbors = new HashSet<string>();

        var outgoing = _graph.GetOutgoingEdges(nodeId);
        foreach (var edge in outgoing)
        {
            neighbors.Add(edge.TargetId);
        }

        var incoming = _graph.GetIncomingEdges(nodeId);
        foreach (var edge in incoming)
        {
            neighbors.Add(edge.SourceId);
        }

        return neighbors;
    }

    /// <summary>
    /// Calculates similarity between two embeddings.
    /// </summary>
    private double CalculateSimilarity(T[] embedding1, T[] embedding2)
    {
        if (embedding1.Length != embedding2.Length)
            return 0.0;

        return Convert.ToDouble(_similarityMetric.CalculateSimilarity(embedding1, embedding2));
    }
}

/// <summary>
/// Represents a retrieval result from the hybrid retriever.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class RetrievalResult<T>
{
    /// <summary>
    /// Gets or sets the node ID.
    /// </summary>
    public string NodeId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the relevance score (0-1, higher is more relevant).
    /// </summary>
    public double Score { get; set; }

    /// <summary>
    /// Gets or sets how this result was retrieved.
    /// </summary>
    public RetrievalSource Source { get; set; }

    /// <summary>
    /// Gets or sets the embedding vector.
    /// </summary>
    public T[]? Embedding { get; set; }

    /// <summary>
    /// Gets or sets the graph traversal depth (0 for initial candidates).
    /// </summary>
    public int Depth { get; set; }

    /// <summary>
    /// Gets or sets the parent node ID (for graph-traversed results).
    /// </summary>
    public string? ParentNodeId { get; set; }

    /// <summary>
    /// Gets or sets the relationship type (for graph-traversed results).
    /// </summary>
    public string? RelationType { get; set; }
}

/// <summary>
/// Indicates how a result was retrieved.
/// </summary>
public enum RetrievalSource
{
    /// <summary>
    /// Retrieved via vector similarity search.
    /// </summary>
    VectorSearch,

    /// <summary>
    /// Retrieved via graph traversal.
    /// </summary>
    GraphTraversal,

    /// <summary>
    /// Retrieved via both methods.
    /// </summary>
    Hybrid
}
