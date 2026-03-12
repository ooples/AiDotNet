
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers
{
    /// <summary>
    /// Dense retrieval using vector similarity search.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
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
    }
}
