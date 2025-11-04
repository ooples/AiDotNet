using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers
{
    /// <summary>
    /// Parent document retriever that retrieves full documents from chunk matches
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations</typeparam>
    public class ParentDocumentRetriever<T> : RetrieverBase<T> where T : struct, IComparable, IComparable<T>, IConvertible, IEquatable<T>, IFormattable
    {
        private readonly IEmbeddingModel<T> _embeddingModel;
        private readonly IDocumentStore<T> _chunkStore;
        private readonly IDocumentStore<T> _parentStore;

        public ParentDocumentRetriever(
            IEmbeddingModel<T> embeddingModel,
            IDocumentStore<T> chunkStore,
            IDocumentStore<T> parentStore)
        {
            _embeddingModel = embeddingModel ?? throw new ArgumentNullException(nameof(embeddingModel));
            _chunkStore = chunkStore ?? throw new ArgumentNullException(nameof(chunkStore));
            _parentStore = parentStore ?? throw new ArgumentNullException(nameof(parentStore));
        }

        protected override async Task<List<Document<T>>> RetrieveCoreAsync(string query, int topK = 5)
        {
            var queryEmbedding = await _embeddingModel.GenerateEmbeddingAsync(query);
            
            var relevantChunks = await _chunkStore.SearchAsync(queryEmbedding, topK * 2);

            var parentIds = new HashSet<string>();
            foreach (var chunk in relevantChunks)
            {
                if (chunk.Metadata.TryGetValue("parent_id", out var parentId))
                {
                    parentIds.Add(parentId);
                }
            }

            var parentDocuments = new List<Document<T>>();
            foreach (var parentId in parentIds.Take(topK))
            {
                var parent = await _parentStore.GetDocumentAsync(parentId);
                if (parent != null)
                {
                    parentDocuments.Add(parent);
                }
            }

            var scoredParents = new List<(Document<T> doc, T score)>();
            foreach (var parent in parentDocuments)
            {
                var score = StatisticsHelper.CosineSimilarity(queryEmbedding, parent.Embedding, NumOps);
                scoredParents.Add((parent, score));
            }

            return scoredParents
                .OrderByDescending(x => x.score)
                .Take(topK)
                .Select(x => x.doc)
                .ToList();
        }
    }
}
