
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers;

/// <summary>
/// Retrieves documents using ColBERT's token-level late interaction mechanism
/// (Khattab &amp; Zaharia 2020, "ColBERT: Efficient and Effective Passage Search
/// via Contextualized Late Interaction over BERT").
/// </summary>
/// <typeparam name="T">The numeric data type used for relevance scoring.</typeparam>
/// <remarks>
/// <para>
/// ColBERT represents queries and documents as <i>multiple</i> contextualised
/// token embeddings rather than a single vector. The scoring rule
/// (paper §3.2 Eq. 1) is
/// <c>Score(Q, D) = Σ_q max_d cos(E_q, E_d)</c>
/// — for every query token, take the maximum cosine similarity against any
/// document token, then sum across query tokens. This "MaxSim" formulation
/// gives finer-grained matching than single-vector dense retrieval at a
/// fraction of the cost of full cross-encoder rerankers.
/// </para>
/// <para>
/// Production usage requires an <see cref="IColBertEmbedder{T}"/> that
/// embeds query and document strings into per-token tensor banks. Without an
/// embedder, the retriever throws <see cref="NotSupportedException"/>:
/// there is no defensible "fallback" for ColBERT because the entire point
/// of the architecture is the contextual token-level representation. A
/// lexical-overlap stand-in would silently produce a different relevance
/// signal under the same class name, which is exactly the kind of silent
/// behavioural divergence this codebase rejects.
/// </para>
/// <para><b>For Beginners:</b> Think of ColBERT like a detailed word-by-word
/// comparison. For each word in your query, ColBERT finds the single
/// best-matching word in each candidate document, then sums those best-match
/// scores. The result is a document score that captures whether
/// <i>every</i> aspect of your query has SOME good match in the document,
/// not just the overall topic.</para>
/// </remarks>
[ComponentType(ComponentType.Retriever)]
[PipelineStage(PipelineStage.Retrieval)]
public class ColBERTRetriever<T> : RetrieverBase<T>
{
    private readonly string _modelPath;
    private readonly int _maxDocLength;
    private readonly int _maxQueryLength;
    private readonly IDocumentStore<T> _documentStore;
    private readonly IColBertEmbedder<T>? _embedder;

    /// <summary>
    /// Initializes a new instance of the ColBERTRetriever class.
    /// </summary>
    /// <param name="documentStore">The document store containing indexed documents.</param>
    /// <param name="modelPath">Path to the ColBERT model file (e.g., ONNX). Kept as
    /// a hint for documentation and serialisation; the actual model is loaded
    /// via <paramref name="embedder"/>.</param>
    /// <param name="maxDocLength">Maximum document length in tokens (typically 180-512).</param>
    /// <param name="maxQueryLength">Maximum query length in tokens (typically 32-64).</param>
    /// <param name="embedder">
    /// Optional token-level embedder. When provided, <see cref="RetrieveCore"/>
    /// runs the paper's MaxSim scoring rule over the embedder's per-token
    /// output. When <c>null</c>, <see cref="RetrieveCore"/> throws
    /// <see cref="NotSupportedException"/> instead of silently falling back
    /// to lexical-overlap pseudo-ColBERT scoring.
    /// </param>
    public ColBERTRetriever(
        IDocumentStore<T> documentStore,
        string modelPath,
        int maxDocLength,
        int maxQueryLength,
        IColBertEmbedder<T>? embedder = null)
    {
        Guard.NotNull(documentStore);
        _documentStore = documentStore;
        Guard.NotNull(modelPath);
        _modelPath = modelPath;

        if (maxDocLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxDocLength), "Max document length must be positive");

        if (maxQueryLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxQueryLength), "Max query length must be positive");

        _maxDocLength = maxDocLength;
        _maxQueryLength = maxQueryLength;
        _embedder = embedder;
    }

    /// <summary>
    /// Retrieves documents using ColBERT MaxSim scoring (paper §3.2).
    /// </summary>
    /// <param name="query">The validated query text.</param>
    /// <param name="topK">The validated number of documents to retrieve.</param>
    /// <param name="metadataFilters">The validated metadata filters.</param>
    /// <returns>Documents ordered by ColBERT relevance score (highest first).</returns>
    /// <exception cref="NotSupportedException">
    /// Thrown when no <see cref="IColBertEmbedder{T}"/> was provided. ColBERT
    /// cannot run without per-token contextual embeddings; a lexical-overlap
    /// "fallback" is not ColBERT and would silently produce different relevance
    /// scores than the paper describes.
    /// </exception>
    protected override IEnumerable<Document<T>> RetrieveCore(
        string query,
        int topK,
        Dictionary<string, object> metadataFilters)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or whitespace", nameof(query));

        if (topK <= 0)
            throw new ArgumentOutOfRangeException(nameof(topK), "topK must be positive");

        if (_embedder is null)
            throw new NotSupportedException(
                "ColBERTRetriever requires an IColBertEmbedder<T> to score documents. " +
                $"The retriever was constructed without one (modelPath='{_modelPath}'). " +
                "Pass an embedder that loads your ColBERT / ColBERTv2 / PLAID checkpoint " +
                "via the constructor's `embedder` parameter — there is no defensible " +
                "fallback for ColBERT scoring because the entire architecture is the " +
                "token-level contextual representation.");

        // 1. Embed the query into per-token vectors [queryTokens, embedDim].
        //    Truncate to _maxQueryLength to stay within the embedder's window.
        var queryEmb = _embedder.EmbedQuery(query);
        int queryTokens = Math.Min(queryEmb.Shape[0], _maxQueryLength);
        int embedDim = queryEmb.Shape[1];

        // 2. Use the query embedding's [CLS]-like mean vector for the
        //    first-stage candidate selection. This is a real query-aware
        //    signal — not a placeholder zero vector — so the candidate set
        //    is genuinely scoped to the query. Paper §6.3 uses BM25 here;
        //    a dense mean-pool gets us paper-equivalent first-stage retrieval
        //    when a BM25 index isn't available.
        var meanQuery = new Vector<T>(_documentStore.VectorDimension);
        int effDim = Math.Min(embedDim, _documentStore.VectorDimension);
        for (int t = 0; t < queryTokens; t++)
            for (int d = 0; d < effDim; d++)
                meanQuery[d] = NumOps.Add(meanQuery[d], queryEmb[t, d]);
        for (int d = 0; d < effDim; d++)
            meanQuery[d] = NumOps.Divide(meanQuery[d], NumOps.FromDouble(queryTokens));

        // Oversample so MaxSim has room to reorder — Khattab & Zaharia
        // §4.2 fetch K' = ~1000 with K = 100; ratio ≈ 10×.
        int oversample = Math.Max(topK * 10, 50);
        var candidates = (metadataFilters == null || metadataFilters.Count == 0)
            ? _documentStore.GetSimilar(meanQuery, oversample)
            : _documentStore.GetSimilarWithFilters(meanQuery, oversample, metadataFilters);

        // 3. Score each candidate with MaxSim against the query's per-token
        //    embeddings.
        var scored = candidates.Select(doc =>
        {
            string content = doc.Content ?? string.Empty;
            var docEmb = _embedder.EmbedDocument(content);
            int docTokens = Math.Min(docEmb.Shape[0], _maxDocLength);
            // Score = Σ_q max_d cos(E_q, E_d). Embeddings are L2-normalised
            // by IColBertEmbedder contract, so cosine == dot product.
            T total = NumOps.Zero;
            for (int q = 0; q < queryTokens; q++)
            {
                T best = NumOps.FromDouble(double.NegativeInfinity);
                for (int d = 0; d < docTokens; d++)
                {
                    T dot = NumOps.Zero;
                    for (int k = 0; k < embedDim; k++)
                    {
                        dot = NumOps.Add(dot,
                            NumOps.Multiply(queryEmb[q, k], docEmb[d, k]));
                    }
                    if (NumOps.GreaterThan(dot, best)) best = dot;
                }
                total = NumOps.Add(total, best);
            }
            return (doc, score: total);
        }).ToList();

        return scored
            .OrderByDescending(x => x.score)
            .Take(topK)
            .Select(x =>
            {
                x.doc.RelevanceScore = x.score;
                x.doc.HasRelevanceScore = true;
                return x.doc;
            });
    }
}
