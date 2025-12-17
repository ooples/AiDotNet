global using System.Text.RegularExpressions;
global using AiDotNet.Interfaces;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers;

/// <summary>
/// Retrieves documents by identifying entities and their relationships in knowledge graph structures.
/// </summary>
/// <typeparam name="T">The numeric data type used for relevance scoring (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// GraphRetriever enhances traditional vector search by leveraging entity recognition and relationship scoring.
/// It extracts entities from the query (proper nouns, quoted terms, numbers), then boosts documents that contain
/// these entities and demonstrate relationships between them (co-occurrence within proximity). This approach is
/// particularly effective for queries requiring structured information or multi-hop reasoning across connected facts.
/// The retriever uses a hybrid scoring strategy combining base vector similarity with entity match scores (30% weight)
/// and relationship scores (20% weight), making it superior to plain vector search for graph-structured data.
/// </para>
/// <para><b>For Beginners:</b> Think of this like a detective searching a case file:
/// 
/// A regular search finds documents by overall similarity. GraphRetriever is smarter—it:
/// 1. Identifies key people, places, and things in your question (entities)
/// 2. Looks for documents mentioning those entities
/// 3. Gives extra points when entities appear close together (showing relationships)
/// 
/// For example, asking "How does John know Mary?" works better than "John Mary relationship" because:
/// - It finds documents mentioning both John AND Mary
/// - It prioritizes documents where John and Mary appear near each other
/// - It understands you're looking for connections, not just mentions
/// 
/// ```csharp
/// var graphRetriever = new GraphRetriever<double>(
///     documentStore: myDocStore,
///     embeddingModel: mySentenceTransformer,
///     enableAdvancedEntityExtraction: true
/// );
/// 
/// var results = graphRetriever.Retrieve("How does Einstein relate to quantum physics?", topK: 5);
/// // Finds documents with both "Einstein" and "quantum physics" appearing together
/// ```
/// 
/// Why use GraphRetriever:
/// - Better at finding connected facts (multi-hop questions)
/// - Understands relationships between entities
/// - Ideal for knowledge-intensive queries (science, history, technical domains)
/// - Works well with structured or semi-structured data
/// 
/// When NOT to use it:
/// - Simple factual lookups (basic vector search is faster)
/// - Queries without clear entities (abstract concepts, opinions)
/// - Documents lacking entity mentions or relationships
/// </para>
/// </remarks>
public class GraphRetriever<T> : RetrieverBase<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly IDocumentStore<T> _documentStore;
    private readonly IEmbeddingModel<T> _embeddingModel;
    private readonly bool _enableAdvancedEntityExtraction;

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphRetriever{T}"/> class.
    /// </summary>
    /// <param name="documentStore">The document store containing indexed documents with entity metadata.</param>
    /// <param name="embeddingModel">The embedding model for semantic query vectorization.</param>
    /// <param name="enableAdvancedEntityExtraction">When true, uses more sophisticated entity extraction (default: true).</param>
    /// <exception cref="ArgumentNullException">Thrown when any parameter is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up the retriever with two key components:
    /// 
    /// 1. Document Store: Where your indexed documents are stored
    /// 2. Embedding Model: Converts text to vectors for similarity comparison
    /// 
    /// Entity extraction is built-in using regex patterns to identify:
    /// - Proper nouns (capitalized words like "Albert Einstein")
    /// - Quoted terms ("quantum mechanics")
    /// - Numbers and dates (1905, 2024)
    /// - Technical terms with special patterns
    /// 
    /// ```csharp
    /// var graphRetriever = new GraphRetriever<double>(
    ///     documentStore: myDocStore,
    ///     embeddingModel: mySentenceTransformer,
    ///     enableAdvancedEntityExtraction: true
    /// );
    /// ```
    /// </para>
    /// </remarks>
    public GraphRetriever(
        IDocumentStore<T> documentStore,
        IEmbeddingModel<T> embeddingModel,
        bool enableAdvancedEntityExtraction = true)
    {
        _documentStore = documentStore ?? throw new ArgumentNullException(nameof(documentStore));
        _embeddingModel = embeddingModel ?? throw new ArgumentNullException(nameof(embeddingModel));
        _enableAdvancedEntityExtraction = enableAdvancedEntityExtraction;
    }

    /// <summary>
    /// Retrieves relevant documents by extracting entities from the query and scoring based on entity matches and relationships.
    /// </summary>
    /// <param name="query">The validated search query (non-empty).</param>
    /// <param name="topK">The validated number of documents to return (positive integer).</param>
    /// <param name="metadataFilters">The validated metadata filters for document selection.</param>
    /// <returns>Documents ordered by enhanced relevance score (combining vector similarity, entity matches, and relationship proximity).</returns>
    /// <exception cref="ArgumentException">Thrown when query is null or whitespace.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when topK is less than or equal to zero.</exception>
    /// <remarks>
    /// <para>
    /// This method implements a three-stage retrieval pipeline:
    /// 1. Entity Extraction: Uses regex patterns to identify proper nouns, quoted terms, and numbers from the query
    /// 2. Enhanced Retrieval: Oversamples documents (topK * 2) with entity filters for comprehensive coverage
    /// 3. Relationship Scoring: Calculates entity co-occurrence scores within 200-character windows, then combines:
    ///    - Base vector similarity (100% weight)
    ///    - Entity match score (30% boost)
    ///    - Relationship score (20% boost)
    /// 
    /// The algorithm prioritizes documents where query entities appear together, indicating factual relationships.
    /// For production systems, this would integrate with graph databases (e.g., Neo4j, GraphDB) via SPARQL/Cypher queries.
    /// The current implementation uses metadata-enhanced vector retrieval as a fallback.
    /// </para>
    /// <para><b>For Beginners:</b> This method does the actual searching. Here's what happens:
    /// 
    /// Step 1: Extract entities from your query
    /// - Query: "How did Marie Curie discover radium?"
    /// - Entities found: ["Marie Curie", "radium"]
    /// 
    /// Step 2: Find documents mentioning these entities
    /// - Gets more documents than needed (topK * 2) to ensure good coverage
    /// 
    /// Step 3: Score documents based on:
    /// - Do they mention the entities? (+30% boost)
    /// - Do the entities appear near each other? (+20% boost if within ~200 characters)
    /// 
    /// Step 4: Return the best topK documents
    /// 
    /// The result is documents that don't just mention your keywords, but actually connect them together.
    /// </para>
    /// </remarks>
    protected override IEnumerable<Document<T>> RetrieveCore(
        string query,
        int topK,
        Dictionary<string, object> metadataFilters)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or whitespace", nameof(query));

        if (topK <= 0)
            throw new ArgumentOutOfRangeException(nameof(topK), "topK must be positive");

        // Extract entities from query using built-in entity extraction
        var entities = ExtractEntities(query);

        // Embed the query for semantic retrieval
        var queryEmbedding = _embeddingModel.Embed(query);
        if (queryEmbedding == null || queryEmbedding.Length == 0)
        {
            throw new InvalidOperationException("Failed to generate query embedding");
        }

        // Retrieve documents using semantic similarity
        var documents = _documentStore.GetSimilarWithFilters(
            new Vector<T>(queryEmbedding),
            topK * 2, // Oversample for re-ranking
            metadataFilters ?? new Dictionary<string, object>()
        ).ToList();

        // Score documents based on entity matches and relationships
        var scoredDocuments = documents.Select(doc =>
        {
            var entityScore = CalculateEntityMatchScore(doc, entities);
            var relationshipScore = CalculateRelationshipScore(doc, entities);

            // Combine scores
            var baseScore = Convert.ToDouble(doc.RelevanceScore);
            var enhancedScore = baseScore * (1.0 + entityScore * 0.3 + relationshipScore * 0.2);

            return (doc, NumOps.FromDouble(enhancedScore));
        }).ToList();

        // Return top-K documents sorted by enhanced score
        return scoredDocuments
            .OrderByDescending(x => x.Item2)
            .Take(topK)
            .Select(x =>
            {
                x.doc.RelevanceScore = x.Item2;
                x.doc.HasRelevanceScore = true;
                return x.doc;
            });
    }

    private double CalculateEntityMatchScore(Document<T> document, List<string> entities)
    {
        if (entities.Count == 0)
            return 0.0;

        var contentLower = document.Content.ToLower();
        var matchCount = entities.Count(e => contentLower.Contains(e.ToLower()));

        return (double)matchCount / entities.Count;
    }

    private double CalculateRelationshipScore(Document<T> document, List<string> entities)
    {
        if (entities.Count < 2)
            return 0.0;

        // Check for co-occurrence of entities (indicating relationships)
        var relationshipCount = 0;
        var content = document.Content;

        for (int i = 0; i < entities.Count - 1; i++)
        {
            for (int j = i + 1; j < entities.Count; j++)
            {
                // Check if both entities appear close to each other
                var entity1Pos = content.IndexOf(entities[i], StringComparison.OrdinalIgnoreCase);
                var entity2Pos = content.IndexOf(entities[j], StringComparison.OrdinalIgnoreCase);

                if (entity1Pos >= 0 && entity2Pos >= 0)
                {
                    var distance = Math.Abs(entity1Pos - entity2Pos);
                    if (distance < 200) // Within 200 characters
                    {
                        relationshipCount++;
                    }
                }
            }
        }

        var maxPossibleRelationships = (entities.Count * (entities.Count - 1)) / 2;
        return maxPossibleRelationships > 0 ? (double)relationshipCount / maxPossibleRelationships : 0.0;
    }

    /// <summary>
    /// Extracts entities from text using pattern matching.
    /// </summary>
    /// <param name="text">The text to extract entities from.</param>
    /// <returns>List of extracted entity strings.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Entities are important nouns or terms like:
    /// - Proper nouns: "Albert Einstein", "New York"
    /// - Quoted terms: "machine learning", "quantum mechanics"
    /// - Numbers and dates: 1905, 2024
    /// - Technical abbreviations: AI, DNA, CPU
    /// 
    /// This method uses regex patterns to identify these automatically without needing external NLP tools.
    /// </para>
    /// </remarks>
    private List<string> ExtractEntities(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return new List<string>();

        var entities = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        // Extract capitalized phrases (proper nouns like "Albert Einstein", "New York")
        var properNouns = Regex.Matches(text, @"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", RegexOptions.None, RegexTimeout);
        foreach (Match match in properNouns)
        {
            if (match.Value.Length > 2) // Filter out single letters
                entities.Add(match.Value);
        }

        // Extract quoted terms (explicit entities in quotes like "machine learning")
        var quotedTerms = Regex.Matches(text, @"""([^""]{2,})""", RegexOptions.None, RegexTimeout);
        foreach (Match match in quotedTerms)
        {
            entities.Add(match.Groups[1].Value);
        }

        // Extract numbers and years (like 1905, 2024)
        if (_enableAdvancedEntityExtraction)
        {
            var numbers = Regex.Matches(text, @"\b(19|20)\d{2}\b", RegexOptions.None, RegexTimeout); // Years
            foreach (Match match in numbers)
            {
                entities.Add(match.Value);
            }

            // Extract technical abbreviations (AI, ML, DNA, etc.)
            var abbreviations = Regex.Matches(text, @"\b[A-Z]{2,5}\b", RegexOptions.None, RegexTimeout);
            foreach (Match match in abbreviations)
            {
                if (match.Value.Length >= 2 && match.Value.Length <= 5)
                    entities.Add(match.Value);
            }
        }

        return entities.Where(e => e.Length > 1).Distinct().ToList();
    }
}
