using System;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores
{
    /// <summary>
    /// Milvus-based document store for scalable vector similarity search.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class MilvusDocumentStore<T> : DocumentStoreBase<T>
    {
        private readonly List<(Document<T> doc, Vector<T> embedding)> _documents = new List<(Document<T>, Vector<T>)>();
        private readonly INumericOperations<T> _numOps;
        private readonly string _collectionName;
        private readonly int _dimension;

        /// <summary>
        /// Initializes a new instance of the <see cref="MilvusDocumentStore{T}"/> class.
        /// </summary>
        /// <param name="numericOperations">The numeric operations for type T.</param>
        /// <param name="collectionName">The name of the Milvus collection.</param>
        /// <param name="dimension">The dimension of the embedding vectors.</param>
        public MilvusDocumentStore(INumericOperations<T> numericOperations, string collectionName, int dimension) : base(numericOperations)
        {
            _numOps = numericOperations ?? throw new ArgumentNullException(nameof(numericOperations));
            _collectionName = collectionName ?? throw new ArgumentNullException(nameof(collectionName));
            _dimension = dimension > 0 ? dimension : throw new ArgumentOutOfRangeException(nameof(dimension), "Dimension must be greater than 0");
        }

        /// <summary>
        /// Adds a document with its embedding to the store.
        /// </summary>
        /// <param name="document">The document to add.</param>
        /// <param name="embedding">The embedding vector for the document.</param>
        public override void AddDocument(Document<T> document, Vector<T> embedding)
        {
            if (document == null) throw new ArgumentNullException(nameof(document));
            if (embedding == null) throw new ArgumentNullException(nameof(embedding));
            if (embedding.Length != _dimension) throw new ArgumentException($"Embedding dimension mismatch. Expected {_dimension}, got {embedding.Length}");

            _documents.Add((document, embedding));
        }

        /// <summary>
        /// Searches for documents similar to the query embedding.
        /// </summary>
        /// <param name="queryEmbedding">The query embedding vector.</param>
        /// <param name="topK">The number of top results to return.</param>
        /// <returns>A list of the most similar documents.</returns>
        public override List<Document<T>> Search(Vector<T> queryEmbedding, int topK)
        {
            if (queryEmbedding == null) throw new ArgumentNullException(nameof(queryEmbedding));
            if (topK <= 0) throw new ArgumentOutOfRangeException(nameof(topK), "TopK must be greater than 0");

            var similarities = _documents.Select(d => new
            {
                Document = d.doc,
                Similarity = StatisticsHelper.CosineSimilarity(_numOps, queryEmbedding, d.embedding)
            }).OrderByDescending(x => _numOps.ToDouble(x.Similarity)).Take(topK);

            foreach (var result in similarities)
            {
                result.Document.RelevanceScore = result.Similarity;
            }

            return similarities.Select(x => x.Document).ToList();
        }

        /// <summary>
        /// Clears all documents from the store.
        /// </summary>
        public override void Clear()
        {
            _documents.Clear();
        }

        /// <summary>
        /// Gets the number of documents in the store.
        /// </summary>
        public override int Count => _documents.Count;
    }
}
