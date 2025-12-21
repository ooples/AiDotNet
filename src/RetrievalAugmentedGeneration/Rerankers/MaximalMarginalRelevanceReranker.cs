
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Rerankers;

/// <summary>
/// Implements Maximal Marginal Relevance (MMR) reranking to balance relevance and diversity.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MMR reranking ensures that retrieved documents are not only relevant to the query but also
/// diverse from each other. This prevents redundancy where all top results say essentially the
/// same thing, providing users with a broader range of information. MMR is particularly valuable
/// for exploratory search, news aggregation, and research applications.
/// </para>
/// <para><b>For Beginners:</b> MMR prevents search results from being too similar to each other.
/// 
/// The problem MMR solves:
/// Imagine searching for "climate change" and getting:
/// 1. "Climate change threatens polar bears"
/// 2. "Polar bears endangered by climate change"
/// 3. "Climate change impact on polar ice affecting bears"
/// 4. "Global warming threatens polar bear habitats"
/// 5. "Arctic ice melting endangers polar bears"
/// 
/// All relevant, but they're all saying the same thing! You're getting one narrow aspect
/// repeated 5 times instead of a diverse view of climate change.
/// 
/// What MMR does instead:
/// 1. "Climate change threatens polar bears" (relevant: ✓, diverse: ✓ first result)
/// 2. "Rising sea levels threaten coastal cities" (relevant: ✓, different topic: ✓)
/// 3. "Carbon emissions reach record highs" (relevant: ✓, different aspect: ✓)
/// 4. "Renewable energy adoption accelerates globally" (relevant: ✓, solutions angle: ✓)
/// 5. "Climate refugees increase in developing nations" (relevant: ✓, human impact: ✓)
/// 
/// Now you get a comprehensive view with diverse perspectives!
/// 
/// How MMR works:
/// 1. Pick the most relevant document → Add to results
/// 2. For next pick, consider:
///    - Relevance to query (you want relevant docs)
///    - Dissimilarity to already-picked docs (you want diversity)
/// 3. Balance these two goals with a lambda parameter
/// 4. Repeat until you have K documents
/// 
/// The lambda parameter (λ):
/// - λ = 1.0: Only care about relevance (normal ranking, no diversity)
/// - λ = 0.0: Only care about diversity (might get irrelevant but diverse docs)
/// - λ = 0.7: Balanced (70% relevance, 30% diversity) ← Good default
/// 
/// When to use MMR:
/// - Research/exploratory queries: Users want comprehensive coverage
/// - News aggregation: Don't show 10 articles about the same event
/// - Product search: Show variety, not just variations of one product
/// - Question answering: Provide multiple perspectives
/// 
/// When NOT to use MMR:
/// - User wants very specific info: "iPhone 15 Pro Max price" (diversity not helpful)
/// - Transactional queries: "buy Nike Air Max" (user knows what they want)
/// - Fact lookups: "Paris population" (one correct answer)
/// </para>
/// </remarks>
public class MaximalMarginalRelevanceReranker<T> : RerankerBase<T>
{
    private readonly double _lambda;
    private readonly Func<Document<T>, Vector<T>> _getEmbedding;

    /// <summary>
    /// Gets a value indicating whether this reranker modifies relevance scores.
    /// </summary>
    public override bool ModifiesScores => true;

    /// <summary>
    /// Initializes a new instance of the MaximalMarginalRelevanceReranker class.
    /// </summary>
    /// <param name="getEmbedding">Function to get document embeddings for similarity calculation.</param>
    /// <param name="lambda">Balance between relevance and diversity (0-1, default: 0.7). Higher values prioritize relevance.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Configuration explained.
    /// 
    /// **getEmbedding Function:**
    /// This function should return the vector embedding of a document. If you don't have
    /// embeddings, you can't use MMR (it needs to measure similarity between documents).
    /// 
    /// Example:
    /// ```csharp
    /// Func&lt;Document&lt;double&gt;, Vector&lt;double&gt;&gt; getEmb = (doc) => {
    ///     // Return cached embedding if available
    ///     if (doc.Metadata.ContainsKey("embedding"))
    ///         return (Vector&lt;double&gt;)doc.Metadata["embedding"];
    ///     
    ///     // Otherwise compute it
    ///     return embeddingModel.Encode(doc.Content);
    /// };
    /// ```
    /// 
    /// **Lambda Parameter Guidelines:**
    /// - 1.0 = Pure relevance (same as no reranking)
    /// - 0.9 = Slight diversity boost
    /// - 0.7 = Balanced (recommended default)
    /// - 0.5 = Equal weight to relevance and diversity
    /// - 0.3 = Heavy diversity (may sacrifice relevance)
    /// - 0.0 = Pure diversity (probably too extreme)
    /// 
    /// Start with 0.7 and adjust based on user feedback!
    /// </para>
    /// </remarks>
    public MaximalMarginalRelevanceReranker(Func<Document<T>, Vector<T>> getEmbedding, double lambda = 0.7)
    {
        if (getEmbedding == null)
            throw new ArgumentNullException(nameof(getEmbedding));

        if (lambda < 0 || lambda > 1)
            throw new ArgumentException("Lambda must be between 0 and 1", nameof(lambda));

        _getEmbedding = getEmbedding;
        _lambda = lambda;
    }

    /// <summary>
    /// Reranks documents using Maximal Marginal Relevance.
    /// </summary>
    protected override IEnumerable<Document<T>> RerankCore(string query, IList<Document<T>> documents)
    {
        var docList = documents.ToList();

        if (docList.Count == 0)
            return Enumerable.Empty<Document<T>>();

        if (docList.Count == 1)
            return docList;

        // Get embeddings for all documents
        var embeddings = docList.Select(doc => _getEmbedding(doc)).ToList();

        // Track which documents have been selected
        var selected = new List<int>();
        var unselected = Enumerable.Range(0, docList.Count).ToList();

        // Select first document (most relevant)
        var firstIdx = unselected
            .OrderByDescending(i => docList[i].HasRelevanceScore ? Convert.ToDouble(docList[i].RelevanceScore) : 0.0)
            .First();

        selected.Add(firstIdx);
        unselected.Remove(firstIdx);

        // Iteratively select remaining documents using MMR
        while (unselected.Count > 0)
        {
            var bestIdx = -1;
            var bestScore = NumOps.FromDouble(double.MinValue);

            foreach (var i in unselected)
            {
                var doc = docList[i];

                // Relevance component (original score)
                var relevance = doc.HasRelevanceScore ? doc.RelevanceScore : NumOps.Zero;

                // Diversity component (max similarity to selected docs)
                var maxSimilarity = NumOps.Zero;
                foreach (var j in selected)
                {
                    var similarity = NumOps.FromDouble(CalculateCosineSimilarity(embeddings[i], embeddings[j]));
                    if (NumOps.GreaterThan(similarity, maxSimilarity))
                        maxSimilarity = similarity;
                }

                // MMR score = λ * relevance - (1 - λ) * maxSimilarity
                var lambdaT = NumOps.FromDouble(_lambda);
                var oneMinusLambda = NumOps.FromDouble(1.0 - _lambda);

                var mmrScore = NumOps.Subtract(
                    NumOps.Multiply(lambdaT, relevance),
                    NumOps.Multiply(oneMinusLambda, maxSimilarity)
                );

                if (bestScore == null || NumOps.GreaterThan(mmrScore, bestScore))
                {
                    bestScore = mmrScore;
                    bestIdx = i;
                }
            }

            if (bestIdx >= 0)
            {
                selected.Add(bestIdx);
                unselected.Remove(bestIdx);
            }
            else
            {
                break;
            }
        }

        // Return documents in MMR order
        var reranked = new List<Document<T>>();
        for (int i = 0; i < selected.Count; i++)
        {
            var doc = docList[selected[i]];

            // Update relevance score to reflect MMR position
            doc.RelevanceScore = NumOps.FromDouble(1.0 - (i / (double)selected.Count));
            doc.HasRelevanceScore = true;

            reranked.Add(doc);
        }

        return reranked;
    }

    /// <summary>
    /// Calculates cosine similarity between two vectors.
    /// </summary>
    private double CalculateCosineSimilarity(Vector<T> a, Vector<T> b)
    {
        if (a == null || b == null)
            throw new ArgumentNullException("Vectors cannot be null");

        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have the same length");

        var similarityT = StatisticsHelper<T>.CosineSimilarity(a, b);
        return Convert.ToDouble(similarityT);
    }
}
