
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Generators;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;

namespace AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns;

/// <summary>
/// Graph-based RAG (Retrieval Augmented Generation) that combines knowledge graph traversal with vector search for enhanced retrieval.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// GraphRAG enhances traditional RAG by maintaining a knowledge graph of entities and relationships alongside
/// vector embeddings. When a query mentions entities in the graph, GraphRAG retrieves both directly related
/// documents (via graph traversal) and semantically similar documents (via vector search), then boosts scores
/// for documents that appear in both results. This leverages structured knowledge for more accurate retrieval.
/// </para>
/// <para><b>For Beginners:</b> Think of a knowledge graph like a mind map or family tree of facts.
/// 
/// Traditional RAG (vector-only):
/// - Question: "What did Einstein discover?"
/// - Search embeddings for documents about "Einstein" and "discovery"
/// - Problem: Might miss important connections or relationships
/// 
/// GraphRAG (graph + vector):
/// - Question: "What did Einstein discover?"
/// - Step 1: Extract entities → "Einstein"
/// - Step 2: Check knowledge graph:
///   * Einstein → DISCOVERED → Theory of Relativity
///   * Einstein → WORKED_AT → Princeton University
///   * Theory of Relativity → INFLUENCED → Quantum Mechanics
/// - Step 3: Vector search for "Einstein" and "discovery"
/// - Step 4: Boost documents that mention graph-connected entities
/// - Result: Prioritizes documents about his actual discoveries over generic biographical info
/// 
/// Real-world analogy:
/// - Regular search: Looking through books by reading every page
/// - GraphRAG: Using the index AND table of contents AND cross-references all together
/// </para>
/// <para><b>Example Usage:</b>
/// <code>
/// // Setup
/// var generator = new StubGenerator&lt;double&gt;(); // Or real LLM for entity extraction
/// var vectorRetriever = new DenseRetriever&lt;double&gt;(embeddingModel, documentStore);
/// 
/// // Create GraphRAG instance
/// var graphRAG = new GraphRAG&lt;double&gt;(generator, vectorRetriever);
/// 
/// // Build knowledge graph (can be done manually or from documents)
/// graphRAG.AddRelation("Albert Einstein", "DISCOVERED", "Theory of Relativity");
/// graphRAG.AddRelation("Albert Einstein", "BORN_IN", "Germany");
/// graphRAG.AddRelation("Theory of Relativity", "PUBLISHED", "1915");
/// graphRAG.AddRelation("Theory of Relativity", "INFLUENCED", "GPS Technology");
/// 
/// // Retrieve with graph-enhanced search
/// var documents = graphRAG.Retrieve("What did Einstein discover?", topK: 10);
/// 
/// // GraphRAG will:
/// // 1. Extract "Einstein" from query
/// // 2. Find graph neighbors: "Theory of Relativity", "Germany", etc.
/// // 3. Perform vector search for the query
/// // 4. Boost docs mentioning "Theory of Relativity" (graph-connected)
/// // 5. Return top-10 with graph-boosted scores
/// </code>
/// </para>
/// <para><b>How It Works:</b>
/// The retrieval process:
/// 1. Entity Extraction - Use LLM to extract entities from the query
/// 2. Graph Traversal - Find all entities connected to query entities in the knowledge graph
/// 3. Vector Retrieval - Perform standard semantic search for the query
/// 4. Score Boosting - Multiply scores by 1.5x for documents mentioning graph-connected entities
/// 5. Ranking - Sort all documents by boosted scores and return top-K
/// 
/// Current implementation uses:
/// - In-memory dictionary for knowledge graph (production should use Neo4j, GraphDB)
/// - Regex-based entity extraction (production should use NER models)
/// - Simple score boosting (production could use graph embeddings, PageRank, etc.)
/// </para>
/// <para><b>Benefits:</b>
/// - Structured reasoning - Leverages explicit relationships between entities
/// - Better precision - Prioritizes documents with known connections
/// - Explainable - Can trace why documents were selected via graph paths
/// - Handles multi-hop reasoning - Can traverse entity → relation → entity chains
/// - Complementary - Combines structured (graph) and unstructured (vector) knowledge
/// </para>
/// <para><b>Limitations:</b>
/// - Requires building/maintaining knowledge graph (initial overhead)
/// - Graph quality affects results (garbage in, garbage out)
/// - Entity extraction quality matters (missed entities = missed connections)
/// - Current implementation is in-memory only (not scalable for large graphs)
/// - Simple boosting strategy (more sophisticated approaches possible)
/// </para>
/// </remarks>
public class GraphRAG<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private const double GraphBoostFactor = 1.5;
    private readonly IGenerator<T> _generator;
    private readonly RetrieverBase<T> _vectorRetriever;
    private readonly Dictionary<string, List<(string relation, string target)>> _knowledgeGraph;
    private readonly Dictionary<string, string> _entityNormalizationMap;
    private readonly int _maxHops;

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphRAG{T}"/> class.
    /// </summary>
    /// <param name="generator">The LLM generator for entity extraction (use StubGenerator or real LLM).</param>
    /// <param name="vectorRetriever">Vector retriever for unstructured data.</param>
    /// <param name="maxHops">Maximum number of hops to traverse in the knowledge graph (default: 2).</param>
    public GraphRAG(
        IGenerator<T> generator,
        RetrieverBase<T> vectorRetriever,
        int maxHops = 2)
    {
        _generator = generator ?? throw new ArgumentNullException(nameof(generator));
        _vectorRetriever = vectorRetriever ?? throw new ArgumentNullException(nameof(vectorRetriever));

        if (maxHops < 1)
            throw new ArgumentOutOfRangeException(nameof(maxHops), "maxHops must be at least 1");

        _maxHops = maxHops;
        _knowledgeGraph = new Dictionary<string, List<(string relation, string target)>>(StringComparer.OrdinalIgnoreCase);
        _entityNormalizationMap = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Adds a relationship to the knowledge graph.
    /// </summary>
    /// <param name="entity">The source entity.</param>
    /// <param name="relation">The relationship type.</param>
    /// <param name="target">The target entity.</param>
    /// <exception cref="ArgumentException">Thrown when entity, relation, or target is null, empty, or whitespace.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a connection between two entities in the knowledge graph.
    /// 
    /// Example:
    /// <code>
    /// graphRAG.AddRelation("Einstein", "DISCOVERED", "Theory of Relativity");
    /// graphRAG.AddRelation("Theory of Relativity", "PUBLISHED_IN", "1915");
    /// </code>
    /// 
    /// This builds a graph structure that can be traversed during retrieval.
    /// </para>
    /// </remarks>
    public void AddRelation(string entity, string relation, string target)
    {
        if (string.IsNullOrWhiteSpace(entity))
            throw new ArgumentException("Entity cannot be null, empty, or whitespace", nameof(entity));
        if (string.IsNullOrWhiteSpace(relation))
            throw new ArgumentException("Relation cannot be null, empty, or whitespace", nameof(relation));
        if (string.IsNullOrWhiteSpace(target))
            throw new ArgumentException("Target cannot be null, empty, or whitespace", nameof(target));

        var normalizedEntity = entity.Trim();
        var normalizedRelation = relation.Trim().ToUpperInvariant();
        var normalizedTarget = target.Trim();

        // Store the original casing for display purposes
        if (!_entityNormalizationMap.ContainsKey(normalizedEntity))
        {
            _entityNormalizationMap[normalizedEntity] = normalizedEntity;
        }
        if (!_entityNormalizationMap.ContainsKey(normalizedTarget))
        {
            _entityNormalizationMap[normalizedTarget] = normalizedTarget;
        }

        if (!_knowledgeGraph.ContainsKey(normalizedEntity))
        {
            _knowledgeGraph[normalizedEntity] = new List<(string, string)>();
        }

        // Avoid duplicate relations (case-insensitive comparison via StringComparer.OrdinalIgnoreCase)
        if (!_knowledgeGraph[normalizedEntity].Any(r =>
            string.Equals(r.relation, normalizedRelation, StringComparison.OrdinalIgnoreCase) &&
            string.Equals(r.target, normalizedTarget, StringComparison.OrdinalIgnoreCase)))
        {
            _knowledgeGraph[normalizedEntity].Add((normalizedRelation, normalizedTarget));
        }
    }

    /// <summary>
    /// Retrieves documents using combined knowledge graph traversal and vector similarity search.
    /// </summary>
    /// <param name="query">The user's query to search for.</param>
    /// <param name="topK">Maximum number of documents to return.</param>
    /// <returns>Documents ranked by graph-boosted relevance scores.</returns>
    /// <exception cref="ArgumentException">Thrown when query is null or whitespace.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when topK is not positive.</exception>
    /// <remarks>
    /// <para>
    /// This method combines structured knowledge from the graph with unstructured vector search.
    /// Documents mentioning entities connected to the query in the knowledge graph receive
    /// higher relevance scores (1.5x boost).
    /// </para>
    /// <para><b>For Beginners:</b> This is the main retrieval method that uses both graph and vectors.
    /// 
    /// Process:
    /// 1. Extract Entities: Use LLM to find entities in query ("Einstein", "relativity", etc.)
    /// 2. Graph Traversal: Find all entities connected to query entities in knowledge graph
    /// 3. Vector Search: Perform regular semantic search for the query
    /// 4. Score Boosting: Multiply score by 1.5x for docs that mention graph-connected entities
    /// 5. Ranking: Sort by boosted scores and return top-K
    /// 
    /// Example:
    /// - Query: "What did Einstein discover?"
    /// - Extracted: ["Einstein"]
    /// - Graph neighbors: ["Theory of Relativity", "E=mc²", "Photoelectric Effect"]
    /// - Vector search returns 20 documents
    /// - Document about "Theory of Relativity" gets 1.5x boost (in graph)
    /// - Document about "Einstein's childhood" stays normal (not in graph neighbors)
    /// - Result: Theory of Relativity doc ranks higher due to graph connection
    /// </para>
    /// </remarks>
    public IEnumerable<Document<T>> Retrieve(string query, int topK)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or whitespace", nameof(query));

        if (topK < 1)
            throw new ArgumentOutOfRangeException(nameof(topK), "topK must be positive");

        // Step 1: Extract entities from query using LLM
        var entities = ExtractEntities(query);

        // Step 2: Traverse knowledge graph to find related entities (multi-hop)
        var relatedEntities = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var relationships = new List<string>();

        // Add initial entities
        foreach (var entity in entities)
        {
            relatedEntities.Add(entity);
        }

        // Multi-hop traversal using BFS
        // Track hop distance for each entity to use in document metadata
        var entityHopDistance = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var frontier = new Queue<string>(entities);
        var visited = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var hopCount = 0;

        // Initial entities are at hop distance 0
        foreach (var entity in entities)
        {
            entityHopDistance[entity] = 0;
        }

        while (frontier.Count > 0 && hopCount < _maxHops)
        {
            var nextFrontier = new Queue<string>();
            var currentLevelCount = frontier.Count;

            for (int i = 0; i < currentLevelCount; i++)
            {
                var currentEntity = frontier.Dequeue();

                if (visited.Contains(currentEntity))
                    continue;

                visited.Add(currentEntity);

                if (_knowledgeGraph.TryGetValue(currentEntity, out var relations))
                {
                    foreach (var (relation, target) in relations)
                    {
                        relatedEntities.Add(target);

                        // Track hop distance for newly discovered entities
                        if (!entityHopDistance.ContainsKey(target))
                        {
                            entityHopDistance[target] = hopCount + 1;
                        }

                        var displayEntity = _entityNormalizationMap.TryGetValue(currentEntity, out var originalEntity)
                            ? originalEntity
                            : currentEntity;
                        var displayTarget = _entityNormalizationMap.TryGetValue(target, out var originalTarget)
                            ? originalTarget
                            : target;
                        relationships.Add($"{displayEntity} {relation} {displayTarget}");

                        if (!visited.Contains(target))
                        {
                            nextFrontier.Enqueue(target);
                        }
                    }
                }
            }

            frontier = nextFrontier;
            hopCount++;
        }

        // Step 3: Use vector retriever for unstructured text (retrieve more than topK for better ranking after boost)
        var retrievalCount = Math.Min(topK * 3, 100); // Retrieve 3x topK to account for boosting, capped at 100
        var vectorResults = _vectorRetriever.Retrieve(query, retrievalCount).ToList();

        // Step 4: Enrich vector results with graph information
        var enrichedResults = new List<Document<T>>();

        foreach (var doc in vectorResults)
        {
            var enrichedContent = doc.Content;

            // Check if document mentions any of our graph entities using word boundary matching (case-insensitive)
            var mentionedEntities = relatedEntities
                .Where(entity => RegexHelper.IsMatch(
                    doc.Content,
                    @"\b" + RegexHelper.Escape(entity) + @"\b",
                    RegexOptions.IgnoreCase))
                .ToList();

            if (mentionedEntities.Count > 0)
            {
                // Add graph context to the document
                var graphContext = $"\n\nRelated knowledge: {string.Join("; ", relationships)}";
                enrichedContent = doc.Content + graphContext;

                // Boost relevance score by fixed 1.5x factor for documents that match graph entities
                var originalScore = doc.HasRelevanceScore
                    ? Convert.ToDouble(doc.RelevanceScore)
                    : 0.5;
                var boostedScore = Math.Min(1.0, originalScore * GraphBoostFactor);

                // Calculate the minimum hop distance among mentioned entities
                // This represents how closely related this document is to the query via the knowledge graph
                var minHopDistance = mentionedEntities
                    .Where(e => entityHopDistance.ContainsKey(e))
                    .Select(e => entityHopDistance[e])
                    .DefaultIfEmpty(0)
                    .Min();

                enrichedResults.Add(new Document<T>
                {
                    Id = doc.Id,
                    Content = enrichedContent,
                    Metadata = new Dictionary<string, object>(doc.Metadata)
                    {
                        ["graph_entities"] = mentionedEntities,
                        ["graph_boosted"] = true,
                        ["graph_hop_count"] = minHopDistance
                    },
                    RelevanceScore = NumOps.FromDouble(boostedScore),
                    HasRelevanceScore = true
                });
            }
            else
            {
                enrichedResults.Add(doc);
            }
        }

        // Step 5: Return top-K enriched results sorted by boosted scores
        return enrichedResults
            .OrderByDescending(d => d.HasRelevanceScore ? d.RelevanceScore : default(T))
            .Take(topK);
    }

    private List<string> ExtractEntities(string text)
    {
        var entities = new List<string>();

        // Extract capitalized phrases (simple proper noun detection)
        var capitalizedMatches = RegexHelper.Matches(text, @"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", RegexOptions.None);
        entities.AddRange(capitalizedMatches.Cast<Match>().Select(m => m.Value));

        // Extract quoted terms
        var quotedMatches = RegexHelper.Matches(text, @"""([^""]+)""", RegexOptions.None);
        entities.AddRange(quotedMatches.Cast<Match>().Select(m => m.Groups[1].Value));

        // Use LLM for more sophisticated extraction if needed
        if (entities.Count == 0)
        {
            try
            {
                var extractionPrompt = $"Extract the main entities (people, places, concepts) from: '{text}'\nList them separated by commas.";
                var llmResponse = _generator.Generate(extractionPrompt);

                if (!string.IsNullOrWhiteSpace(llmResponse))
                {
                    // Parse comma-separated or newline-separated entities from LLM
                    var llmEntities = llmResponse
                        .Split(new[] { ',', '\n', ';' }, StringSplitOptions.RemoveEmptyEntries)
                        .Select(e => e.Trim())
                        .Where(e => e.Length > 2 && e.Length < 100) // Reasonable entity length
                        .Where(e => !e.StartsWith("[") && !e.StartsWith("-")) // Filter out list markers
                        .Where(e => !e.All(char.IsDigit)) // Filter out pure numbers
                        .Take(10); // Limit to 10 entities to avoid noise

                    entities.AddRange(llmEntities);
                }
            }
            catch (Exception)
            {
                // If LLM fails, fall back to regex-only extraction
                // This ensures graceful degradation
            }
        }

        return entities.Distinct().ToList();
    }
}



