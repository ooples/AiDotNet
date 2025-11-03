using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;

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
        private readonly INumericOperations<T> _numOps;

        /// <summary>
        /// Initializes a new instance of the <see cref="DenseRetriever{T}"/> class.
        /// </summary>
        /// <param name="numericOperations">The numeric operations for type T.</param>
        /// <param name="documentStore">The document store for retrieval.</param>
        /// <param name="embeddingModel">The embedding model for query encoding.</param>
        public DenseRetriever(
            INumericOperations<T> numericOperations,
            IDocumentStore<T> documentStore,
            IEmbeddingModel<T> embeddingModel) : base(numericOperations)
        {
            _numOps = numericOperations ?? throw new ArgumentNullException(nameof(numericOperations));
            _documentStore = documentStore ?? throw new ArgumentNullException(nameof(documentStore));
            _embeddingModel = embeddingModel ?? throw new ArgumentNullException(nameof(embeddingModel));
        }

        /// <summary>
        /// Retrieves documents using dense vector similarity.
        /// </summary>
        /// <param name="query">The query string.</param>
        /// <param name="topK">The number of documents to retrieve.</param>
        /// <returns>A list of the most relevant documents.</returns>
        public override List<Document<T>> Retrieve(string query, int topK)
        {
            if (string.IsNullOrEmpty(query)) throw new ArgumentNullException(nameof(query));
            if (topK <= 0) throw new ArgumentOutOfRangeException(nameof(topK));

            var queryEmbedding = _embeddingModel.Embed(query);
            return _documentStore.Search(queryEmbedding, topK);
        }
    }
}
