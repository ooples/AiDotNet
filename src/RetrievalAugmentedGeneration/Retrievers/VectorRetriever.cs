using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers
{
    /// <summary>
    /// Dense vector-based retriever using embeddings
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations</typeparam>
    public class VectorRetriever<T> : RetrieverBase<T> where T : struct, IComparable, IComparable<T>, IConvertible, IEquatable<T>, IFormattable
    {
        private readonly IEmbeddingModel<T> _embeddingModel;
        private readonly IDocumentStore<T> _documentStore;

        public VectorRetriever(IEmbeddingModel<T> embeddingModel, IDocumentStore<T> documentStore)
        {
            _embeddingModel = embeddingModel ?? throw new ArgumentNullException(nameof(embeddingModel));
            _documentStore = documentStore ?? throw new ArgumentNullException(nameof(documentStore));
        }

        protected override async Task<List<Document<T>>> RetrieveCoreAsync(string query, int topK = 5)
        {
            var queryEmbedding = await _embeddingModel.GenerateEmbeddingAsync(query);
            return await _documentStore.SearchAsync(queryEmbedding, topK);
        }
    }
}
