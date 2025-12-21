
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Rerankers;

/// <summary>
/// Reranks documents to maximize diversity while maintaining relevance.
/// </summary>
/// <typeparam name="T">The numeric data type used for scoring.</typeparam>
/// <remarks>
/// <para>
/// This reranker addresses the problem of redundant results by explicitly promoting diversity.
/// It uses a greedy algorithm to select documents that are both relevant to the query and
/// dissimilar from already-selected documents. This is similar to Maximal Marginal Relevance (MMR)
/// but uses a simpler diversity metric based on text overlap.
/// </para>
/// <para><b>For Beginners:</b> This prevents showing the same information multiple times.
/// 
/// The Problem:
/// Imagine searching for "Python programming" and getting 10 results:
/// - Result 1: "Python is a programming language..."
/// - Result 2: "Python is a programming language used for..."
/// - Result 3: "Python programming language allows..."
/// - Result 4-10: More variations of the same thing
/// 
/// That's redundant! You want variety:
/// - Result 1: Python basics
/// - Result 2: Python web development
/// - Result 3: Python data science
/// - Result 4: Python machine learning
/// - Result 5: Python performance tips
/// 
/// How it works:
/// 1. Pick the most relevant document first
/// 2. For remaining docs, balance two factors:
///    a) Relevance to the query (should be useful)
///    b) Difference from already-picked docs (should be unique)
/// 3. Keep picking until you have enough results
/// 
/// Diversity calculation:
/// - Compares text overlap (how many words are shared)
/// - Higher overlap = less diverse = lower score
/// - Lower overlap = more diverse = higher score
/// 
/// Lambda parameter (0 to 1):
/// - lambda=1.0: Only care about relevance (might get duplicates)
/// - lambda=0.0: Only care about diversity (might get irrelevant docs)
/// - lambda=0.5: Balance both (recommended default)
/// 
/// Real example with lambda=0.5:
/// Query: "climate change effects"
/// 
/// Step 1: Pick most relevant → "Climate change causes rising temperatures" (relevance: 0.9)
/// Step 2: Next candidates:
///   - "Climate change leads to warmer weather" (relevance: 0.85, similarity to picked: 0.7)
///     → Score: 0.5 * 0.85 - 0.5 * 0.7 = 0.075
///   - "Ocean acidification from CO2" (relevance: 0.7, similarity: 0.2)
///     → Score: 0.5 * 0.7 - 0.5 * 0.2 = 0.25 ✓ Pick this!
/// 
/// Result: You get coverage of temperature AND ocean effects, not just temperature twice!
/// 
/// When to use this:
/// - Search results where redundancy is common
/// - Document recommendation systems
/// - Exploratory searches where breadth matters
/// - After initial retrieval that returns many similar docs
/// </para>
/// </remarks>
public class DiversityReranker<T> : RerankerBase<T>
{
    private readonly T _lambda;

    /// <summary>
    /// Gets a value indicating whether this reranker modifies relevance scores.
    /// </summary>
    public override bool ModifiesScores => true;

    /// <summary>
    /// Initializes a new instance of the DiversityReranker class.
    /// </summary>
    /// <param name="lambda">Trade-off parameter between relevance and diversity (0 to 1). Default: 0.5.
    /// Higher values favor relevance, lower values favor diversity.</param>
    public DiversityReranker(T? lambda = default) : base()
    {
        var lambdaValue = lambda ?? NumOps.FromDouble(0.5);

        // Validate lambda is in [0, 1]
        if (NumOps.LessThan(lambdaValue, NumOps.Zero) || NumOps.GreaterThan(lambdaValue, NumOps.One))
        {
            throw new ArgumentException("Lambda must be between 0 and 1.", nameof(lambda));
        }

        _lambda = lambdaValue;
    }

    /// <summary>
    /// Core reranking logic that maximizes diversity while maintaining relevance.
    /// </summary>
    /// <param name="query">The validated query text.</param>
    /// <param name="documents">The validated list of documents to rerank.</param>
    /// <returns>Documents reranked to balance relevance and diversity.</returns>
    protected override IEnumerable<Document<T>> RerankCore(string query, IList<Document<T>> documents)
    {
        var docList = documents.ToList();
        if (docList.Count <= 1)
        {
            return docList;
        }

        var rerankedDocs = new List<Document<T>>();
        var remainingDocs = new List<Document<T>>(docList);

        // Start with the most relevant document
        var firstDoc = remainingDocs.OrderByDescending(d => d.RelevanceScore,
            Comparer<T>.Create((a, b) =>
            {
                if (NumOps.GreaterThan(a, b)) return 1;
                if (NumOps.LessThan(a, b)) return -1;
                return 0;
            })).First();
        rerankedDocs.Add(firstDoc);
        remainingDocs.Remove(firstDoc);

        // Greedily select documents that maximize: lambda * relevance - (1 - lambda) * max_similarity
        while (remainingDocs.Count > 0)
        {
            Document<T>? bestDoc = null;
            var bestScore = NumOps.FromDouble(double.MinValue);

            foreach (var doc in remainingDocs)
            {
                // Calculate maximum similarity to already-selected documents
                var maxSimilarity = NumOps.Zero;
                foreach (var selectedDoc in rerankedDocs)
                {
                    var similarity = CalculateTextSimilarity(doc.Content, selectedDoc.Content);
                    if (NumOps.GreaterThan(similarity, maxSimilarity))
                    {
                        maxSimilarity = similarity;
                    }
                }

                // Calculate diversity-aware score: lambda * relevance - (1 - lambda) * max_similarity
                var relevancePart = NumOps.Multiply(_lambda, doc.RelevanceScore);
                var oneMinusLambda = NumOps.Subtract(NumOps.One, _lambda);
                var diversityPenalty = NumOps.Multiply(oneMinusLambda, maxSimilarity);
                var score = NumOps.Subtract(relevancePart, diversityPenalty);

                if (bestScore == null || NumOps.GreaterThan(score, bestScore))
                {
                    bestDoc = doc;
                    bestScore = score;
                }
            }

            if (bestDoc != null)
            {
                rerankedDocs.Add(bestDoc);
                remainingDocs.Remove(bestDoc);
            }
            else
            {
                break;
            }
        }

        // Update relevance scores based on final ranking
        for (int i = 0; i < rerankedDocs.Count; i++)
        {
            var rank = NumOps.FromDouble(i + 1);
            var totalDocs = NumOps.FromDouble(rerankedDocs.Count);
            // Score decreases with rank: (totalDocs - rank + 1) / totalDocs
            var numerator = NumOps.Add(NumOps.Subtract(totalDocs, rank), NumOps.One);
            var newScore = NumOps.Divide(numerator, totalDocs);

            rerankedDocs[i].RelevanceScore = newScore;
            rerankedDocs[i].HasRelevanceScore = true;
        }

        return rerankedDocs;
    }

    /// <summary>
    /// Calculates text similarity based on word overlap (Jaccard similarity of word sets).
    /// </summary>
    /// <param name="text1">First text.</param>
    /// <param name="text2">Second text.</param>
    /// <returns>Similarity score between 0 and 1.</returns>
    private T CalculateTextSimilarity(string text1, string text2)
    {
        if (string.IsNullOrWhiteSpace(text1) || string.IsNullOrWhiteSpace(text2))
        {
            return NumOps.Zero;
        }

        // Tokenize and create word sets
        var words1 = new HashSet<string>(
            text1.ToLowerInvariant()
                .Split(new[] { ' ', '\t', '\n', '\r', '.', ',', ';', ':', '!', '?' },
                    StringSplitOptions.RemoveEmptyEntries));

        var words2 = new HashSet<string>(
            text2.ToLowerInvariant()
                .Split(new[] { ' ', '\t', '\n', '\r', '.', ',', ';', ':', '!', '?' },
                    StringSplitOptions.RemoveEmptyEntries));

        // Calculate Jaccard similarity: |intersection| / |union|
        var intersection = words1.Intersect(words2).Count();
        var union = words1.Union(words2).Count();

        if (union == 0)
        {
            return NumOps.Zero;
        }

        var similarity = (double)intersection / union;
        return NumOps.FromDouble(similarity);
    }
}
