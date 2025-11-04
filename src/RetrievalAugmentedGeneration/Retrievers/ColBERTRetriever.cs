using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers
{
    /// <summary>
    /// ColBERT-style late interaction retriever
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations</typeparam>
    public class ColBERTRetriever<T> : RetrieverBase<T> where T : struct, IComparable, IComparable<T>, IConvertible, IEquatable<T>, IFormattable
    {
        private readonly IEmbeddingModel<T> _embeddingModel;
        private readonly IDocumentStore<T> _documentStore;

        public ColBERTRetriever(IEmbeddingModel<T> embeddingModel, IDocumentStore<T> documentStore)
        {
            _embeddingModel = embeddingModel ?? throw new ArgumentNullException(nameof(embeddingModel));
            _documentStore = documentStore ?? throw new ArgumentNullException(nameof(documentStore));
        }

        protected override async Task<List<Document<T>>> RetrieveCoreAsync(string query, int topK = 5)
        {
            var queryTokens = query.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            var tokenEmbeddings = new List<Vector<T>>();

            foreach (var token in queryTokens)
            {
                var embedding = await _embeddingModel.GenerateEmbeddingAsync(token);
                tokenEmbeddings.Add(embedding);
            }

            var queryEmbedding = await _embeddingModel.GenerateEmbeddingAsync(query);
            var candidates = await _documentStore.SearchAsync(queryEmbedding, topK * 3);

            var scoredDocuments = new List<(Document<T> doc, T score)>();

            foreach (var doc in candidates)
            {
                var docTokens = doc.Content.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                var docTokenEmbeddings = new List<Vector<T>>();

                foreach (var token in docTokens.Take(100))
                {
                    var embedding = await _embeddingModel.GenerateEmbeddingAsync(token);
                    docTokenEmbeddings.Add(embedding);
                }

                var score = ComputeMaxSim(tokenEmbeddings, docTokenEmbeddings);
                scoredDocuments.Add((doc, score));
            }

            return scoredDocuments
                .OrderByDescending(x => x.score)
                .Take(topK)
                .Select(x => x.doc)
                .ToList();
        }

        private T ComputeMaxSim(List<Vector<T>> queryEmbeddings, List<Vector<T>> docEmbeddings)
        {
            var totalScore = NumOps.Zero;

            foreach (var queryEmb in queryEmbeddings)
            {
                var maxSim = NumOps.FromDouble(-1.0);

                foreach (var docEmb in docEmbeddings)
                {
                    var sim = StatisticsHelper.CosineSimilarity(queryEmb, docEmb, NumOps);
                    if (NumOps.GreaterThan(sim, maxSim))
                    {
                        maxSim = sim;
                    }
                }

                totalScore = NumOps.Add(totalScore, maxSim);
            }

            return NumOps.Divide(totalScore, NumOps.FromInt32(queryEmbeddings.Count));
        }
    }
}
