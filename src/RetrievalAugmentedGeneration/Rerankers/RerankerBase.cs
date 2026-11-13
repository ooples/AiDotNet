
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Rerankers;

/// <summary>
/// Provides a base implementation for document rerankers with common functionality.
/// </summary>
/// <typeparam name="T">The numeric data type used for relevance scoring.</typeparam>
/// <remarks>
/// <para>
/// This abstract class implements the IReranker interface and provides common functionality
/// for reranking strategies. It handles validation, result limiting, and score normalization
/// while allowing derived classes to focus on implementing the core reranking algorithm.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation that all reranking methods build upon.
/// 
/// Think of it like a template for improving search results:
/// - It handles common tasks (checking inputs, limiting results, normalizing scores)
/// - Specific reranking methods (cross-encoder, LLM-based) just fill in how they score documents
/// - This ensures all rerankers work consistently
/// </para>
/// </remarks>
public abstract class RerankerBase<T> : IReranker<T>
{
    /// <summary>
    /// Provides mathematical operations for the numeric type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>The engine used for vectorized scoring.</summary>
    private protected static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Gets a value indicating whether this reranker modifies relevance scores.
    /// </summary>
    public abstract bool ModifiesScores { get; }

    /// <summary>
    /// Reranks a collection of documents based on their relevance to a query.
    /// </summary>
    /// <param name="query">The query text used to assess relevance.</param>
    /// <param name="documents">The documents to rerank.</param>
    /// <returns>The documents reordered by relevance, with updated relevance scores.</returns>
    public IEnumerable<Document<T>> Rerank(string query, IEnumerable<Document<T>> documents)
    {
        ValidateQuery(query);
        ValidateDocuments(documents);

        var documentList = documents.ToList();
        if (documentList.Count == 0)
            return documentList;

        return RerankCore(query, documentList);
    }

    /// <summary>
    /// Reranks documents and returns only the top-k highest scoring results.
    /// </summary>
    /// <param name="query">The query text used to assess relevance.</param>
    /// <param name="documents">The documents to rerank.</param>
    /// <param name="topK">The number of top-ranked documents to return.</param>
    /// <returns>The top-k documents ordered by relevance, with updated relevance scores.</returns>
    public IEnumerable<Document<T>> Rerank(string query, IEnumerable<Document<T>> documents, int topK)
    {
        ValidateTopK(topK);

        var reranked = Rerank(query, documents);
        return reranked.Take(topK).ToList();
    }

    /// <summary>
    /// Asynchronously reranks a collection of documents based on their relevance to a query.
    /// </summary>
    public virtual async Task<IEnumerable<Document<T>>> RerankAsync(
        string query,
        IEnumerable<Document<T>> documents,
        CancellationToken cancellationToken = default)
    {
        ValidateQuery(query);
        ValidateDocuments(documents);

        var documentList = documents.ToList();
        if (documentList.Count == 0)
            return documentList;

        cancellationToken.ThrowIfCancellationRequested();

        return await RerankCoreAsync(query, documentList, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Asynchronously reranks documents and returns only the top-k highest scoring results.
    /// </summary>
    public virtual async Task<IEnumerable<Document<T>>> RerankAsync(
        string query,
        IEnumerable<Document<T>> documents,
        int topK,
        CancellationToken cancellationToken = default)
    {
        ValidateTopK(topK);

        var reranked = await RerankAsync(query, documents, cancellationToken).ConfigureAwait(false);
        return reranked.Take(topK).ToList();
    }

    /// <summary>
    /// Core reranking logic to be implemented by derived classes.
    /// </summary>
    /// <param name="query">The validated query text.</param>
    /// <param name="documents">The validated and materialized list of documents.</param>
    /// <returns>The documents reordered by relevance with updated scores.</returns>
    /// <remarks>
    /// <para><b>For Implementers:</b> This is where you implement your specific reranking algorithm.
    /// 
    /// You don't need to:
    /// - Validate the query (already done)
    /// - Validate documents (already done)
    /// - Convert IEnumerable to List (already done)
    /// - Handle null/empty inputs (already validated)
    /// 
    /// Just focus on: Scoring documents and ordering them by relevance.
    /// Make sure to update each document's RelevanceScore property!
    /// </para>
    /// </remarks>
    protected abstract IEnumerable<Document<T>> RerankCore(string query, IList<Document<T>> documents);

    /// <summary>
    /// Core asynchronous reranking logic. The default implementation wraps the synchronous
    /// <see cref="RerankCore"/>, which is appropriate for CPU-bound rerankers. LLM- or cross-encoder-backed
    /// rerankers should override this to perform genuine non-blocking work.
    /// </summary>
    /// <param name="query">The validated query text.</param>
    /// <param name="documents">The validated and materialized list of documents.</param>
    /// <param name="cancellationToken">A token to observe for cancellation requests.</param>
    /// <returns>A task producing the documents reordered by relevance with updated scores.</returns>
    protected virtual Task<IEnumerable<Document<T>>> RerankCoreAsync(
        string query,
        IList<Document<T>> documents,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult(RerankCore(query, documents));
    }

    /// <summary>
    /// Validates the query string.
    /// </summary>
    /// <param name="query">The query to validate.</param>
    /// <remarks>
    /// <para><b>For Implementers:</b> Override this if you need custom query validation.
    /// </para>
    /// </remarks>
    protected virtual void ValidateQuery(string query)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or empty", nameof(query));
    }

    /// <summary>
    /// Validates the document collection.
    /// </summary>
    /// <param name="documents">The documents to validate.</param>
    /// <remarks>
    /// <para><b>For Implementers:</b> Override this if you need custom document validation.
    /// For example, checking that documents have content or required metadata.
    /// </para>
    /// </remarks>
    protected virtual void ValidateDocuments(IEnumerable<Document<T>> documents)
    {
        if (documents == null)
            throw new ArgumentNullException(nameof(documents));
    }

    /// <summary>
    /// Validates the topK parameter.
    /// </summary>
    /// <param name="topK">The topK value to validate.</param>
    /// <remarks>
    /// <para><b>For Implementers:</b> Override this if you need custom topK validation.
    /// </para>
    /// </remarks>
    protected virtual void ValidateTopK(int topK)
    {
        if (topK <= 0)
            throw new ArgumentException("TopK must be greater than zero", nameof(topK));
    }

    /// <summary>
    /// Normalizes relevance scores to the 0-1 range.
    /// </summary>
    /// <param name="documents">The documents with scores to normalize.</param>
    /// <returns>The documents with normalized scores.</returns>
    /// <remarks>
    /// <para>
    /// This helper method can be used by derived classes to normalize scores after reranking.
    /// It uses min-max normalization to scale scores to the 0-1 range.
    /// </para>
    /// <para><b>For Implementers:</b> Call this if your reranker produces scores in different ranges.
    /// 
    /// For example:
    /// - Your model outputs scores from -10 to 10
    /// - Call NormalizeScores() to convert them to 0-1 range
    /// - Makes scores consistent with other retrievers/rerankers
    /// </para>
    /// </remarks>
    protected IList<Document<T>> NormalizeScores(IList<Document<T>> documents)
    {
        var docsWithScores = documents.Where(d => d.HasRelevanceScore).ToList();
        if (docsWithScores.Count == 0)
            return documents;

        var scores = new Tensor<T>(docsWithScores.Select(d => d.RelevanceScore).ToArray(), [docsWithScores.Count]);
        var minScore = Engine.TensorMinValue(scores);
        var maxScore = Engine.TensorMaxValue(scores);

        var range = NumOps.Subtract(maxScore, minScore);
        var epsilon = NumOps.FromDouble(1e-8);
        var isZeroRange = EqualityComparer<T>.Default.Equals(range, NumOps.Zero);
        if (isZeroRange || NumOps.LessThan(NumOps.Abs(range), epsilon))
        {
            // All scores are the same, set them all to 1.0
            var one = NumOps.One;
            foreach (var doc in docsWithScores)
            {
                doc.RelevanceScore = one;
                doc.HasRelevanceScore = true;
            }
        }
        else
        {
            var normalized = Engine.TensorDivideScalar(Engine.TensorSubtractScalar(scores, minScore), range).ToArray();
            foreach (var (doc, score) in docsWithScores.Zip(normalized, (doc, score) => (doc, score)))
            {
                doc.RelevanceScore = score;
                doc.HasRelevanceScore = true;
            }
        }

        return documents;
    }

    /// <summary>
    /// Greedy Maximal Marginal Relevance order (Carbonell and Goldstein 1998): start from the most relevant
    /// document, then repeatedly pick argmax_i [λ rel_i - (1 - λ) max_{j selected} sim(i, j)].
    /// </summary>
    /// <param name="relevance">Relevance of each document, shape [n].</param>
    /// <param name="similarity">Pairwise document similarity, shape [n, n].</param>
    /// <param name="lambda">Trade-off between relevance (1) and diversity (0).</param>
    /// <returns>Document indices in selection order.</returns>
    private protected static int[] MarginalRelevanceOrder(Tensor<T> relevance, Tensor<T> similarity, T lambda)
    {
        int n = relevance.Length;
        var order = new int[n];
        if (n == 0) return order;

        var selected = new bool[n];
        var excluded = Tensor<T>.CreateDefault([n], NumOps.FromDouble(double.NegativeInfinity));
        var weightedRelevance = Engine.TensorMultiplyScalar(relevance, lambda);
        var diversityWeight = NumOps.Subtract(NumOps.One, lambda);

        int current = Engine.TensorArgMax(relevance, 0).GetFlat(0);
        order[0] = current;
        selected[current] = true;
        var maxSimilarity = SimilarityRow(similarity, current, n);

        for (int k = 1; k < n; k++)
        {
            var score = Engine.TensorSubtract(weightedRelevance, Engine.TensorMultiplyScalar(maxSimilarity, diversityWeight));
            current = Engine.TensorArgMax(Engine.TensorWhere(selected, excluded, score), 0).GetFlat(0);
            if (selected[current])
            {
                // Every remaining score is -inf or NaN: keep the original order for the rest.
                current = Array.IndexOf(selected, false);
            }

            order[k] = current;
            selected[current] = true;
            maxSimilarity = Engine.TensorMax(maxSimilarity, SimilarityRow(similarity, current, n));
        }

        return order;
    }

    /// <summary>
    /// Scores documents by final position: the document at rank i (0-based) of n gets (n - i) / n.
    /// </summary>
    private protected static void AssignRankScores(IList<Document<T>> ranked)
    {
        int n = ranked.Count;
        if (n == 0) return;

        var count = NumOps.FromDouble(n);
        var scores = Engine.TensorDivideScalar(
            Engine.ScalarMinusTensor(count, Engine.TensorArange<T>(0, n, 1)), count).ToArray();
        for (int i = 0; i < n; i++)
        {
            ranked[i].RelevanceScore = scores[i];
            ranked[i].HasRelevanceScore = true;
        }
    }

    private static Tensor<T> SimilarityRow(Tensor<T> similarity, int row, int n)
        => Engine.Reshape(Engine.TensorNarrow(similarity, 0, row, 1), [n]);
}
