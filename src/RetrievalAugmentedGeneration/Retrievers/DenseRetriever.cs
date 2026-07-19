
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers
{
    /// <summary>
    /// Dense retrieval using vector similarity search.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    [ComponentType(ComponentType.Retriever)]
    [PipelineStage(PipelineStage.Retrieval)]
    public class DenseRetriever<T> : RetrieverBase<T>
    {
        private readonly IDocumentStore<T> _documentStore;
        private readonly IEmbeddingModel<T> _embeddingModel;

        public DenseRetriever(IDocumentStore<T> documentStore, IEmbeddingModel<T> embeddingModel, int defaultTopK = 5)
            : base(defaultTopK)
        {
            if (documentStore == null)
                throw new ArgumentNullException(nameof(documentStore));
            if (embeddingModel == null)
                throw new ArgumentNullException(nameof(embeddingModel));

            _documentStore = documentStore;
            _embeddingModel = embeddingModel;
        }

        protected override IEnumerable<Document<T>> RetrieveCore(string query, int topK, Dictionary<string, object> metadataFilters)
        {
            var queryEmbedding = _embeddingModel.Embed(query);
            return _documentStore.GetSimilarWithFilters(queryEmbedding, topK, metadataFilters);
        }

        /// <summary>
        /// Truly-async core retrieval: awaits the (possibly network-backed) embedding model and document
        /// store instead of blocking, and flows the cancellation token end to end.
        /// </summary>
        protected override async System.Threading.Tasks.Task<IEnumerable<Document<T>>> RetrieveCoreAsync(
            string query,
            int topK,
            Dictionary<string, object> metadataFilters,
            System.Threading.CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var queryEmbedding = await _embeddingModel.EmbedAsync(query).ConfigureAwait(false);
            cancellationToken.ThrowIfCancellationRequested();
            return await _documentStore
                .GetSimilarWithFiltersAsync(queryEmbedding, topK, metadataFilters, cancellationToken)
                .ConfigureAwait(false);
        }
    }
}
