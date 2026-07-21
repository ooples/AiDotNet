using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Rerankers;

/// <summary>
/// A ColBERT-style late-interaction reranker (Khattab &amp; Zaharia 2020) that scores each
/// (query, document) pair with the "MaxSim" rule and reorders candidates by descending score.
/// </summary>
/// <typeparam name="T">The numeric data type used for relevance scoring.</typeparam>
/// <remarks>
/// <para>
/// The reranker embeds the query and every candidate document into <i>per-token</i> vectors using an
/// <see cref="ITokenEmbeddingModel{T}"/>, then computes
/// <c>Score(Q, D) = Σ_q max_d (E_q · E_d)</c> — for each query token, take the maximum dot product
/// against any document token, then sum over query tokens. Token embeddings are L2-normalised before
/// scoring, so the dot product equals cosine similarity. Document token embeddings are cached per
/// <see cref="Rerank(string, System.Collections.Generic.IEnumerable{Document{T}})"/> call so a
/// document appearing more than once (by id) is embedded only once.
/// </para>
/// <para>
/// By default a deterministic offline <see cref="HashingTokenEmbeddingModel{T}"/> is used, which
/// makes the reranker fully testable without downloading any model. Supply a real contextual token
/// embedder via the constructor for production semantics.
/// </para>
/// <para><b>For Beginners:</b> This is a careful, word-by-word second pass over search results. For
/// every word in your query it finds the single best-matching word in each candidate document and
/// adds up those best matches. A document that has a strong match for <i>every</i> part of your
/// query scores highest and floats to the top.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Reranker)]
[PipelineStage(PipelineStage.PostRetrieval)]
public class ColbertReranker<T> : RerankerBase<T>
{
    private readonly ITokenEmbeddingModel<T> _tokenEmbeddingModel;

    /// <summary>
    /// Gets a value indicating whether this reranker modifies relevance scores. Always <c>true</c>.
    /// </summary>
    public override bool ModifiesScores => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="ColbertReranker{T}"/> class.
    /// </summary>
    /// <param name="tokenEmbeddingModel">
    /// The per-token embedding model used to produce query and document token vectors. Required —
    /// there is no default. ColBERT late interaction is only meaningful over real (semantic) token
    /// embeddings; a placeholder would silently produce meaningless scores. Supply a real model,
    /// e.g. an <see cref="EmbeddingModelTokenAdapter{T}"/> over any real <c>IEmbeddingModel&lt;T&gt;</c>
    /// (Word2Vec / GloVe / ONNX sentence-transformer / a hosted embedder).
    /// </param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="tokenEmbeddingModel"/> is null.</exception>
    public ColbertReranker(ITokenEmbeddingModel<T> tokenEmbeddingModel)
    {
        if (tokenEmbeddingModel is null)
            throw new ArgumentNullException(nameof(tokenEmbeddingModel),
                "ColbertReranker requires a real token-embedding model; late interaction over placeholder " +
                "embeddings is meaningless. Wrap a real IEmbeddingModel<T> in EmbeddingModelTokenAdapter<T>.");
        _tokenEmbeddingModel = tokenEmbeddingModel;
    }

    /// <summary>
    /// Creates a reranker whose token embeddings come from a real single-vector
    /// <see cref="IEmbeddingModel{T}"/>, adapted to per-token output via <see cref="EmbeddingModelTokenAdapter{T}"/>.
    /// </summary>
    /// <param name="embeddingModel">A real embedding model used to embed each token.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="embeddingModel"/> is null.</exception>
    public ColbertReranker(IEmbeddingModel<T> embeddingModel)
        : this(new EmbeddingModelTokenAdapter<T>(embeddingModel))
    {
    }

    /// <summary>
    /// Reranks documents using ColBERT MaxSim scoring.
    /// </summary>
    /// <param name="query">The validated query text.</param>
    /// <param name="documents">The validated, materialized list of candidate documents.</param>
    /// <returns>The documents ordered by descending MaxSim score, with updated relevance scores.</returns>
    protected override IEnumerable<Document<T>> RerankCore(string query, IList<Document<T>> documents)
    {
        var queryTokens = NormalizeTokens(_tokenEmbeddingModel.EmbedTokens(query));

        // Per-call cache of document token embeddings keyed by document identity.
        var docTokenCache = new Dictionary<string, Vector<T>[]>();

        var scored = new List<(Document<T> doc, T score)>(documents.Count);
        foreach (var doc in documents)
        {
            var docTokens = GetDocumentTokens(doc, docTokenCache);
            var score = MaxSim(queryTokens, docTokens);
            scored.Add((doc, score));
        }

        return scored
            .OrderByDescending(x => x.score)
            .Select(x =>
            {
                x.doc.RelevanceScore = x.score;
                x.doc.HasRelevanceScore = true;
                return x.doc;
            })
            .ToList();
    }

    // Async reranking is provided by RerankerBase<T>.RerankAsync, which delegates to RerankCore above;
    // no override is needed (ColBERT MaxSim is CPU-bound and synchronous).

    /// <summary>
    /// Computes the ColBERT MaxSim score: Σ over query tokens of the max over document tokens of the
    /// dot product between L2-normalised token embeddings.
    /// </summary>
    /// <param name="queryTokens">The L2-normalised query token embeddings.</param>
    /// <param name="docTokens">The L2-normalised document token embeddings.</param>
    /// <returns>The summed maximum-similarity score.</returns>
    internal static T MaxSim(IReadOnlyList<Vector<T>> queryTokens, IReadOnlyList<Vector<T>> docTokens)
    {
        var total = NumOps.Zero;
        if (queryTokens.Count == 0 || docTokens.Count == 0)
            return total;

        foreach (var q in queryTokens)
        {
            var best = NumOps.FromDouble(double.NegativeInfinity);
            foreach (var d in docTokens)
            {
                var dot = Dot(q, d);
                if (NumOps.GreaterThan(dot, best))
                    best = dot;
            }

            total = NumOps.Add(total, best);
        }

        return total;
    }

    private Vector<T>[] GetDocumentTokens(Document<T> doc, Dictionary<string, Vector<T>[]> cache)
    {
        // Prefer the document id as the cache key; fall back to content when no id is present.
        var key = !string.IsNullOrEmpty(doc.Id) ? "id:" + doc.Id : "content:" + (doc.Content ?? string.Empty);
        if (cache.TryGetValue(key, out var cached))
            return cached;

        var tokens = NormalizeTokens(_tokenEmbeddingModel.EmbedTokens(doc.Content ?? string.Empty));
        cache[key] = tokens;
        return tokens;
    }

    private static Vector<T>[] NormalizeTokens(Vector<T>[] tokens)
    {
        // Ensure every token vector is unit length so dot product == cosine similarity, even if the
        // supplied embedding model did not normalise its output.
        for (int i = 0; i < tokens.Length; i++)
        {
            tokens[i] = VectorHelper.Normalize(tokens[i]);
        }

        return tokens;
    }

    private static T Dot(Vector<T> a, Vector<T> b)
    {
        var length = Math.Min(a.Length, b.Length);
        var sum = NumOps.Zero;
        for (int i = 0; i < length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(a[i], b[i]));
        }

        return sum;
    }
}
