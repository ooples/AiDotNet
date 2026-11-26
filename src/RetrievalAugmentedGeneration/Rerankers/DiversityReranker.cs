
using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ComponentType(ComponentType.Reranker)]
[PipelineStage(PipelineStage.PostRetrieval)]
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

        // Greedily select documents that maximize: lambda * relevance - (1 - lambda) * max_similarity
        var relevance = new Tensor<T>(docList.Select(d => d.RelevanceScore).ToArray(), [docList.Count]);
        var order = MarginalRelevanceOrder(relevance, JaccardSimilarityMatrix(docList), _lambda);
        var rerankedDocs = order.Select(i => docList[i]).ToList();

        // Update relevance scores based on final ranking: (totalDocs - rank + 1) / totalDocs
        AssignRankScores(rerankedDocs);
        return rerankedDocs;
    }

    /// <summary>
    /// Jaccard similarity |A ∩ B| / |A ∪ B| of every pair of documents' lower-cased word sets
    /// (0 when either document has no words).
    /// </summary>
    private static Tensor<T> JaccardSimilarityMatrix(List<Document<T>> documents)
    {
        var separators = new[] { ' ', '\t', '\n', '\r', '.', ',', ';', ':', '!', '?' };
        var wordSets = documents
            .Select(d => string.IsNullOrWhiteSpace(d.Content)
                ? new HashSet<string>()
                : new HashSet<string>(d.Content.ToLowerInvariant().Split(separators, StringSplitOptions.RemoveEmptyEntries)))
            .ToList();

        int n = wordSets.Count;
        var intersections = new int[n * n];
        var unions = new int[n * n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                int shared = wordSets[i].Count(wordSets[j].Contains);
                intersections[i * n + j] = shared;

                // An empty union only happens for two empty sets, whose intersection is 0 as well.
                unions[i * n + j] = Math.Max(wordSets[i].Count + wordSets[j].Count - shared, 1);
            }
        }

        return Engine.TensorDivide(
            new Tensor<int>(intersections, [n, n]).Cast<T>(),
            new Tensor<int>(unions, [n, n]).Cast<T>());
    }
}
