
using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ComponentType(ComponentType.Reranker)]
[PipelineStage(PipelineStage.PostRetrieval)]
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

        // MMR = argmax[λ Sim1(D_i, Q) - (1 - λ) max_{D_j ∈ S} Sim2(D_i, D_j)] (Carbonell and Goldstein 1998),
        // with Sim1 the original relevance score (0 when absent) and Sim2 the embedding cosine similarity.
        var relevance = new Tensor<T>(
            docList.Select(doc => doc.HasRelevanceScore ? doc.RelevanceScore : NumOps.Zero).ToArray(), [docList.Count]);
        var similarity = CosineSimilarityMatrix(docList.Select(doc => _getEmbedding(doc)).ToList());
        var order = MarginalRelevanceOrder(relevance, similarity, NumOps.FromDouble(_lambda));

        // Return documents in MMR order; scores reflect MMR position: 1 - i / n
        var reranked = order.Select(i => docList[i]).ToList();
        AssignRankScores(reranked);
        return reranked;
    }

    /// <summary>
    /// Pairwise cosine similarity of the embeddings (0 for zero vectors), clamped to [-1, 1].
    /// Rows are divided by their largest magnitude before squaring so narrow T cannot overflow.
    /// </summary>
    private static Tensor<T> CosineSimilarityMatrix(List<Vector<T>> embeddings)
    {
        var matrix = Engine.TensorStack(embeddings.Select(e => Tensor<T>.FromVector(e)).ToArray(), 0);
        var one = NumOps.One;
        var scale = Engine.ReduceMax(Engine.TensorAbs(matrix), [1], true);
        var scaled = Engine.TensorDivide(matrix, NonZero(scale, one));
        var norms = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorSquare(scaled), [1], true));
        var unit = Engine.TensorDivide(scaled, NonZero(norms, one));
        var cosine = Engine.TensorMatMul(unit, Engine.TensorTranspose(unit));
        return Engine.TensorClamp(cosine, NumOps.Negate(one), one);
    }

    /// <summary>Replaces zero entries with <paramref name="fill"/> (a zero row stays zero after dividing).</summary>
    private static Tensor<T> NonZero(Tensor<T> values, T fill)
        => Engine.TensorWhere(
            Engine.TensorGreaterThan(values, NumOps.Zero), values, Tensor<T>.CreateDefault(values._shape, fill));

}
