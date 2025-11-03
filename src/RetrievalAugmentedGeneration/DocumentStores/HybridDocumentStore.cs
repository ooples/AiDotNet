using System;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores
{
    /// <summary>
    /// Hybrid document store combining dense and sparse retrieval strategies.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class HybridDocumentStore<T> : DocumentStoreBase<T>
    {
        private readonly IDocumentStore<T> _denseStore;
        private readonly IDocumentStore<T> _sparseStore;
        private readonly INumericOperations<T> _numOps;
        private readonly T _denseWeight;
        private readonly T _sparseWeight;

        /// <summary>
        /// Initializes a new instance of the <see cref="HybridDocumentStore{T}"/> class.
        /// </summary>
        /// <param name="numericOperations">The numeric operations for type T.</param>
        /// <param name="denseStore">The dense retrieval document store.</param>
        /// <param name="sparseStore">The sparse retrieval document store.</param>
        /// <param name="denseWeight">The weight for dense retrieval scores.</param>
        /// <param name="sparseWeight">The weight for sparse retrieval scores.</param>
        public HybridDocumentStore(
            INumericOperations<T> numericOperations,
            IDocumentStore<T> denseStore,
            IDocumentStore<T> sparseStore,
            T denseWeight,
            T sparseWeight) : base(numericOperations)
        {
            _numOps = numericOperations ?? throw new ArgumentNullException(nameof(numericOperations));
            _denseStore = denseStore ?? throw new ArgumentNullException(nameof(denseStore));
            _sparseStore = sparseStore ?? throw new ArgumentNullException(nameof(sparseStore));
            _denseWeight = denseWeight;
            _sparseWeight = sparseWeight;
        }

        /// <summary>
        /// Adds a document with its embedding to both stores.
        /// </summary>
        /// <param name="document">The document to add.</param>
        /// <param name="embedding">The embedding vector for the document.</param>
        public override void AddDocument(Document<T> document, Vector<T> embedding)
        {
            if (document == null) throw new ArgumentNullException(nameof(document));
            if (embedding == null) throw new ArgumentNullException(nameof(embedding));

            _denseStore.AddDocument(document, embedding);
            _sparseStore.AddDocument(document, embedding);
        }

        /// <summary>
        /// Searches using both dense and sparse methods and combines the results.
        /// </summary>
        /// <param name="queryEmbedding">The query embedding vector.</param>
        /// <param name="topK">The number of top results to return.</param>
        /// <returns>A list of the most similar documents with combined scores.</returns>
        public override List<Document<T>> Search(Vector<T> queryEmbedding, int topK)
        {
            if (queryEmbedding == null) throw new ArgumentNullException(nameof(queryEmbedding));
            if (topK <= 0) throw new ArgumentOutOfRangeException(nameof(topK), "TopK must be greater than 0");

            var denseResults = _denseStore.Search(queryEmbedding, topK * 2);
            var sparseResults = _sparseStore.Search(queryEmbedding, topK * 2);

            var scoreMap = new Dictionary<string, T>();

            foreach (var doc in denseResults)
            {
                if (!scoreMap.ContainsKey(doc.Id))
                {
                    scoreMap[doc.Id] = _numOps.Zero;
                }
                scoreMap[doc.Id] = _numOps.Add(scoreMap[doc.Id], _numOps.Multiply(doc.RelevanceScore, _denseWeight));
            }

            foreach (var doc in sparseResults)
            {
                if (!scoreMap.ContainsKey(doc.Id))
                {
                    scoreMap[doc.Id] = _numOps.Zero;
                }
                scoreMap[doc.Id] = _numOps.Add(scoreMap[doc.Id], _numOps.Multiply(doc.RelevanceScore, _sparseWeight));
            }

            var allDocs = denseResults.Concat(sparseResults)
                .GroupBy(d => d.Id)
                .Select(g => g.First())
                .ToList();

            foreach (var doc in allDocs)
            {
                if (scoreMap.ContainsKey(doc.Id))
                {
                    doc.RelevanceScore = scoreMap[doc.Id];
                }
            }

            return allDocs.OrderByDescending(d => _numOps.ToDouble(d.RelevanceScore)).Take(topK).ToList();
        }

        /// <summary>
        /// Clears all documents from both stores.
        /// </summary>
        public override void Clear()
        {
            _denseStore.Clear();
            _sparseStore.Clear();
        }

        /// <summary>
        /// Gets the maximum count from both stores.
        /// </summary>
        public override int Count => Math.Max(_denseStore.Count, _sparseStore.Count);
    }
}
