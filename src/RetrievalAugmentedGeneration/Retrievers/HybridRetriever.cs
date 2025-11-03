using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers;

/// <summary>
/// Combines dense and sparse retrievers to leverage both semantic and keyword-based search.
/// </summary>
/// <typeparam name="T">The numeric data type used for scoring and calculations.</typeparam>
/// <remarks>
/// <para>
/// Hybrid retrieval combines the strengths of different retrieval methods:
/// - Dense retrieval (vector search) excels at semantic similarity
/// - Sparse retrieval (BM25, TF-IDF) excels at exact keyword matching
/// By combining both, we get the best of both worlds with improved recall and precision.
/// </para>
/// <para><b>For Beginners:</b> This is like having two different search experts work together.
/// 
/// Imagine you're searching for "best Italian restaurants in Chicago":
/// 
/// **Dense Retriever (Vector Search):**
/// - Understands meaning: finds "top-rated pizzerias in Chicago"
/// - Finds: "authentic pasta places in the Windy City"
/// - Good at: Similar concepts, synonyms, related ideas
/// 
/// **Sparse Retriever (BM25):**
/// - Matches keywords: looks for exact words "Italian", "restaurants", "Chicago"
/// - Finds: "Chicago Italian restaurants", "Italian food Chicago"
/// - Good at: Exact terms, rare words, specific names
/// 
/// **Hybrid (This Class):**
/// - Gets results from both
/// - Combines and ranks them intelligently
/// - Result: More complete, accurate search results
/// 
/// Fusion strategies:
/// - **Reciprocal Rank Fusion (RRF)**: Combines rankings fairly (default, recommended)
/// - **Weighted Linear**: Mix scores with custom weights (e.g., 70% dense + 30% sparse)
/// - **Max Score**: Take the highest score from either retriever
/// 
/// Real-world example:
/// Query: "machine learning papers about transformers"
/// 
/// Dense finds: Papers about attention mechanisms, BERT, GPT (semantic match)
/// Sparse finds: Papers with exact term "transformer" in title
/// Hybrid combines: All relevant papers, properly ranked
/// 
/// This is the industry standard for production RAG systems!</para>
/// </remarks>
public class HybridRetriever<T> : RetrieverBase<T>
{
    private readonly IRetriever<T> _denseRetriever;
    private readonly IRetriever<T> _sparseRetriever;
    private readonly FusionStrategy _fusionStrategy;
    private readonly T _denseWeight;
    private readonly T _sparseWeight;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Strategies for combining results from multiple retrievers.
    /// </summary>
    public enum FusionStrategy
    {
        /// <summary>Reciprocal Rank Fusion - combines rankings fairly without requiring score normalization.</summary>
        ReciprocalRankFusion,
        
        /// <summary>Weighted linear combination of normalized scores.</summary>
        WeightedLinear,
        
        /// <summary>Takes the maximum score from either retriever for each document.</summary>
        MaxScore
    }

    /// <summary>
    /// Initializes a new instance of the HybridRetriever class.
    /// </summary>
    /// <param name="denseRetriever">The dense (vector-based) retriever.</param>
    /// <param name="sparseRetriever">The sparse (keyword-based) retriever.</param>
    /// <param name="numOps">Numeric operations for type T.</param>
    /// <param name="fusionStrategy">The strategy for combining results (default: ReciprocalRankFusion).</param>
    /// <param name="denseWeight">Weight for dense retriever scores (default: 0.5). Only used with WeightedLinear strategy.</param>
    /// <param name="sparseWeight">Weight for sparse retriever scores (default: 0.5). Only used with WeightedLinear strategy.</param>
    /// <param name="defaultTopK">The default number of documents to retrieve.</param>
    public HybridRetriever(
        IRetriever<T> denseRetriever,
        IRetriever<T> sparseRetriever,
        INumericOperations<T> numOps,
        FusionStrategy fusionStrategy = FusionStrategy.ReciprocalRankFusion,
        T? denseWeight = default,
        T? sparseWeight = default,
        int defaultTopK = 5) : base(defaultTopK)
    {
        _denseRetriever = denseRetriever ?? throw new ArgumentNullException(nameof(denseRetriever));
        _sparseRetriever = sparseRetriever ?? throw new ArgumentNullException(nameof(sparseRetriever));
        _numOps = numOps ?? throw new ArgumentNullException(nameof(numOps));
        _fusionStrategy = fusionStrategy;
        
        // Set default weights if not provided
        _denseWeight = denseWeight ?? _numOps.FromDouble(0.5);
        _sparseWeight = sparseWeight ?? _numOps.FromDouble(0.5);

        // Validate weights for WeightedLinear strategy
        if (fusionStrategy == FusionStrategy.WeightedLinear)
        {
            var weightSum = _numOps.Add(_denseWeight, _sparseWeight);
            var one = _numOps.One;
            var diff = _numOps.Abs(_numOps.Subtract(weightSum, one));
            var epsilon = _numOps.FromDouble(1e-6);
            
            if (_numOps.GreaterThan(diff, epsilon))
            {
                throw new ArgumentException("Dense and sparse weights must sum to 1.0 for WeightedLinear strategy.");
            }
        }
    }

    /// <summary>
    /// Core retrieval logic using hybrid fusion of dense and sparse results.
    /// </summary>
    /// <param name="query">The validated query text.</param>
    /// <param name="topK">The validated number of documents to retrieve.</param>
    /// <param name="metadataFilters">The validated metadata filters.</param>
    /// <returns>A collection of relevant documents ordered by fused relevance.</returns>
    protected override IEnumerable<Document<T>> RetrieveCore(string query, int topK, Dictionary<string, object> metadataFilters)
    {
        // Retrieve more results from each retriever to ensure good coverage after fusion
        var retrievalK = topK * 2;

        // Get results from both retrievers
        var denseResults = _denseRetriever.Retrieve(query, retrievalK, metadataFilters).ToList();
        var sparseResults = _sparseRetriever.Retrieve(query, retrievalK, metadataFilters).ToList();

        // Fuse the results based on the selected strategy
        var fusedResults = _fusionStrategy switch
        {
            FusionStrategy.ReciprocalRankFusion => FuseWithReciprocalRankFusion(denseResults, sparseResults),
            FusionStrategy.WeightedLinear => FuseWithWeightedLinear(denseResults, sparseResults),
            FusionStrategy.MaxScore => FuseWithMaxScore(denseResults, sparseResults),
            _ => throw new InvalidOperationException($"Unknown fusion strategy: {_fusionStrategy}")
        };

        // Return top K results
        return fusedResults.Take(topK);
    }

    /// <summary>
    /// Fuses results using Reciprocal Rank Fusion (RRF).
    /// </summary>
    /// <remarks>
    /// RRF score = sum(1 / (k + rank_i)) where k=60 is a constant and rank_i is the rank from each retriever.
    /// This method is robust and doesn't require score normalization.
    /// </remarks>
    private List<Document<T>> FuseWithReciprocalRankFusion(List<Document<T>> denseResults, List<Document<T>> sparseResults)
    {
        var k = _numOps.FromDouble(60.0); // Standard RRF constant
        var documentScores = new Dictionary<string, T>();

        // Process dense results
        for (int i = 0; i < denseResults.Count; i++)
        {
            var doc = denseResults[i];
            var rank = _numOps.FromDouble(i + 1);
            var score = _numOps.Divide(_numOps.One, _numOps.Add(k, rank));
            
            if (documentScores.ContainsKey(doc.Id))
            {
                documentScores[doc.Id] = _numOps.Add(documentScores[doc.Id], score);
            }
            else
            {
                documentScores[doc.Id] = score;
            }
        }

        // Process sparse results
        for (int i = 0; i < sparseResults.Count; i++)
        {
            var doc = sparseResults[i];
            var rank = _numOps.FromDouble(i + 1);
            var score = _numOps.Divide(_numOps.One, _numOps.Add(k, rank));
            
            if (documentScores.ContainsKey(doc.Id))
            {
                documentScores[doc.Id] = _numOps.Add(documentScores[doc.Id], score);
            }
            else
            {
                documentScores[doc.Id] = score;
            }
        }

        // Create a document lookup
        var allDocs = denseResults.Concat(sparseResults).GroupBy(d => d.Id).ToDictionary(g => g.Key, g => g.First());

        // Sort by fused score and return
        return documentScores
            .OrderByDescending(kvp => kvp.Value, Comparer<T>.Create((a, b) => {
                if (_numOps.GreaterThan(a, b)) return 1;
                if (_numOps.LessThan(a, b)) return -1;
                return 0;
            }))
            .Select(kvp =>
            {
                var doc = allDocs[kvp.Key];
                doc.RelevanceScore = kvp.Value;
                doc.HasRelevanceScore = true;
                return doc;
            })
            .ToList();
    }

    /// <summary>
    /// Fuses results using weighted linear combination of normalized scores.
    /// </summary>
    private List<Document<T>> FuseWithWeightedLinear(List<Document<T>> denseResults, List<Document<T>> sparseResults)
    {
        // Normalize scores to [0, 1] range
        var normalizedDense = NormalizeScores(denseResults);
        var normalizedSparse = NormalizeScores(sparseResults);

        var documentScores = new Dictionary<string, T>();

        // Weighted combination of dense scores
        foreach (var doc in normalizedDense)
        {
            var weightedScore = _numOps.Multiply(doc.RelevanceScore, _denseWeight);
            documentScores[doc.Id] = weightedScore;
        }

        // Add weighted sparse scores
        foreach (var doc in normalizedSparse)
        {
            var weightedScore = _numOps.Multiply(doc.RelevanceScore, _sparseWeight);
            
            if (documentScores.ContainsKey(doc.Id))
            {
                documentScores[doc.Id] = _numOps.Add(documentScores[doc.Id], weightedScore);
            }
            else
            {
                documentScores[doc.Id] = weightedScore;
            }
        }

        // Create document lookup
        var allDocs = normalizedDense.Concat(normalizedSparse).GroupBy(d => d.Id).ToDictionary(g => g.Key, g => g.First());

        // Sort and return
        return documentScores
            .OrderByDescending(kvp => kvp.Value, Comparer<T>.Create((a, b) => {
                if (_numOps.GreaterThan(a, b)) return 1;
                if (_numOps.LessThan(a, b)) return -1;
                return 0;
            }))
            .Select(kvp =>
            {
                var doc = allDocs[kvp.Key];
                doc.RelevanceScore = kvp.Value;
                doc.HasRelevanceScore = true;
                return doc;
            })
            .ToList();
    }

    /// <summary>
    /// Fuses results by taking the maximum score from either retriever.
    /// </summary>
    private List<Document<T>> FuseWithMaxScore(List<Document<T>> denseResults, List<Document<T>> sparseResults)
    {
        // Normalize scores
        var normalizedDense = NormalizeScores(denseResults);
        var normalizedSparse = NormalizeScores(sparseResults);

        var documentScores = new Dictionary<string, T>();

        // Process all documents and take max score
        foreach (var doc in normalizedDense)
        {
            documentScores[doc.Id] = doc.RelevanceScore;
        }

        foreach (var doc in normalizedSparse)
        {
            if (documentScores.ContainsKey(doc.Id))
            {
                var currentScore = documentScores[doc.Id];
                if (_numOps.GreaterThan(doc.RelevanceScore, currentScore))
                {
                    documentScores[doc.Id] = doc.RelevanceScore;
                }
            }
            else
            {
                documentScores[doc.Id] = doc.RelevanceScore;
            }
        }

        // Create document lookup
        var allDocs = normalizedDense.Concat(normalizedSparse).GroupBy(d => d.Id).ToDictionary(g => g.Key, g => g.First());

        // Sort and return
        return documentScores
            .OrderByDescending(kvp => kvp.Value, Comparer<T>.Create((a, b) => {
                if (_numOps.GreaterThan(a, b)) return 1;
                if (_numOps.LessThan(a, b)) return -1;
                return 0;
            }))
            .Select(kvp =>
            {
                var doc = allDocs[kvp.Key];
                doc.RelevanceScore = kvp.Value;
                doc.HasRelevanceScore = true;
                return doc;
            })
            .ToList();
    }

    /// <summary>
    /// Normalizes document scores to [0, 1] range using min-max normalization.
    /// </summary>
    private List<Document<T>> NormalizeScores(List<Document<T>> documents)
    {
        if (documents.Count == 0)
        {
            return new List<Document<T>>();
        }

        // Find min and max scores
        var docsWithScores = documents.Where(d => d.HasRelevanceScore).ToList();
        if (docsWithScores.Count == 0)
        {
            return documents;
        }

        var minScore = docsWithScores.Min(d => d.RelevanceScore);
        var maxScore = docsWithScores.Max(d => d.RelevanceScore);
        var range = _numOps.Subtract(maxScore, minScore);

        // If all scores are the same, return uniform scores
        if (_numOps.Equals(range, _numOps.Zero))
        {
            foreach (var doc in documents)
            {
                doc.RelevanceScore = _numOps.One;
                doc.HasRelevanceScore = true;
            }
            return documents;
        }

        // Normalize: (score - min) / (max - min)
        foreach (var doc in documents)
        {
            if (doc.HasRelevanceScore)
            {
                var normalized = _numOps.Divide(
                    _numOps.Subtract(doc.RelevanceScore, minScore),
                    range
                );
                doc.RelevanceScore = normalized;
            }
        }

        return documents;
    }
}
