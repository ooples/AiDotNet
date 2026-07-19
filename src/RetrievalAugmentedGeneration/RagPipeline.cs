using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration
{
    /// <summary>
    /// End-to-end RAG orchestrator that composes the individual components into the two operations an
    /// application actually performs: <see cref="IngestAsync"/> (chunk → embed → store) and
    /// <see cref="QueryAsync"/> (retrieve → rerank → compress → generate). Previously callers had to wire
    /// these stages by hand; this is the single entry point that ties them together, fully async and
    /// cancellation-aware, with optional tenant/namespace isolation.
    /// </summary>
    /// <typeparam name="T">The numeric type used for embeddings and scoring.</typeparam>
    public sealed class RagPipeline<T>
    {
        private readonly IEmbeddingModel<T> _embedding;
        private readonly IDocumentStore<T> _store;
        private readonly IRetriever<T> _retriever;
        private readonly IChunkingStrategy? _chunking;
        private readonly IReranker<T>? _reranker;
        private readonly IContextCompressor<T>? _compressor;
        private readonly IGenerator<T>? _generator;
        private readonly string? _tenant;

        /// <param name="embedding">Embedding model used for ingestion (and dense retrieval upstream).</param>
        /// <param name="store">Vector store documents are upserted into.</param>
        /// <param name="retriever">Retriever used at query time.</param>
        /// <param name="chunking">Optional chunker; when null each document is ingested whole.</param>
        /// <param name="reranker">Optional reranker applied to retrieved documents.</param>
        /// <param name="compressor">Optional context compressor applied before generation.</param>
        /// <param name="generator">Optional generator; when null <see cref="QueryAsync"/> returns retrieved context only.</param>
        /// <param name="tenant">
        /// Optional tenant/namespace. When set it is stamped on every ingested document's metadata under
        /// <c>_tenant</c> and used as a retrieval filter, isolating this pipeline's data from other tenants.
        /// </param>
        public RagPipeline(
            IEmbeddingModel<T> embedding,
            IDocumentStore<T> store,
            IRetriever<T> retriever,
            IChunkingStrategy? chunking = null,
            IReranker<T>? reranker = null,
            IContextCompressor<T>? compressor = null,
            IGenerator<T>? generator = null,
            string? tenant = null)
        {
            _embedding = embedding ?? throw new ArgumentNullException(nameof(embedding));
            _store = store ?? throw new ArgumentNullException(nameof(store));
            _retriever = retriever ?? throw new ArgumentNullException(nameof(retriever));
            _chunking = chunking;
            _reranker = reranker;
            _compressor = compressor;
            _generator = generator;
            _tenant = string.IsNullOrWhiteSpace(tenant) ? null : tenant;
        }

        /// <summary>Metadata key under which the tenant/namespace is stored on ingested documents.</summary>
        public const string TenantMetadataKey = "_tenant";

        /// <summary>
        /// Ingests a document: optionally chunks it, embeds each chunk, and upserts it into the store.
        /// </summary>
        /// <returns>The number of chunks stored.</returns>
        public async Task<int> IngestAsync(
            string id, string content, IReadOnlyDictionary<string, object>? metadata = null, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrEmpty(id)) throw new ArgumentException("Document id is required.", nameof(id));
            if (string.IsNullOrEmpty(content)) throw new ArgumentException("Document content is required.", nameof(content));

            var chunks = _chunking != null ? _chunking.Chunk(content).ToList() : new List<string> { content };
            int index = 0;
            foreach (var chunk in chunks)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var meta = new Dictionary<string, object>();
                if (metadata != null)
                {
                    foreach (var kv in metadata) meta[kv.Key] = kv.Value;
                }
                if (_tenant != null) meta[TenantMetadataKey] = _tenant;

                var doc = new Document<T>(chunks.Count == 1 ? id : $"{id}::{index}", chunk) { Metadata = meta };
                var embedding = await _embedding.EmbedAsync(chunk).ConfigureAwait(false);
                await _store.AddAsync(new VectorDocument<T>(doc, embedding), cancellationToken).ConfigureAwait(false);
                index++;
            }

            return chunks.Count;
        }

        /// <summary>
        /// Answers a question: retrieve → (rerank) → (compress) → (generate).
        /// </summary>
        public async Task<RagResult<T>> QueryAsync(string question, int topK = 5, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrEmpty(question)) throw new ArgumentException("Question is required.", nameof(question));

            var filters = new Dictionary<string, object>();
            if (_tenant != null) filters[TenantMetadataKey] = _tenant;

            var retrieved = (await _retriever.RetrieveAsync(question, topK, filters, cancellationToken).ConfigureAwait(false)).ToList();

            if (_reranker != null)
            {
                retrieved = (await _reranker.RerankAsync(question, retrieved, cancellationToken).ConfigureAwait(false)).ToList();
            }

            var context = _compressor != null
                ? await _compressor.CompressAsync(retrieved, question, null, cancellationToken).ConfigureAwait(false)
                : retrieved;

            GroundedAnswer<T>? answer = null;
            if (_generator != null && context.Count > 0)
            {
                answer = await _generator.GenerateGroundedAsync(question, context, cancellationToken).ConfigureAwait(false);
            }

            return new RagResult<T>(question, context, answer);
        }
    }

    /// <summary>The outcome of a <see cref="RagPipeline{T}.QueryAsync"/> call.</summary>
    public sealed class RagResult<T>
    {
        public RagResult(string question, List<Document<T>> contexts, GroundedAnswer<T>? answer)
        {
            Question = question;
            Contexts = contexts;
            Answer = answer;
        }

        /// <summary>The original question.</summary>
        public string Question { get; }

        /// <summary>The retrieved (and reranked/compressed) context documents.</summary>
        public List<Document<T>> Contexts { get; }

        /// <summary>The generated grounded answer, or <c>null</c> when no generator was configured.</summary>
        public GroundedAnswer<T>? Answer { get; }
    }
}
