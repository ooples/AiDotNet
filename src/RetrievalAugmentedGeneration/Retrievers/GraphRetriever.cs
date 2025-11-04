using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers
{
    /// <summary>
    /// Graph-based retriever for traversing document relationships
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations</typeparam>
    public class GraphRetriever<T> : RetrieverBase<T> where T : struct, IComparable, IComparable<T>, IConvertible, IEquatable<T>, IFormattable
    {
        private readonly IEmbeddingModel<T> _embeddingModel;
        private readonly IDocumentStore<T> _documentStore;
        private readonly Dictionary<string, List<string>> _graph;
        private readonly int _maxDepth;

        public GraphRetriever(
            IEmbeddingModel<T> embeddingModel,
            IDocumentStore<T> documentStore,
            int maxDepth = 2)
        {
            _embeddingModel = embeddingModel ?? throw new ArgumentNullException(nameof(embeddingModel));
            _documentStore = documentStore ?? throw new ArgumentNullException(nameof(documentStore));
            _graph = new Dictionary<string, List<string>>();
            _maxDepth = maxDepth;
        }

        public void AddEdge(string fromDocId, string toDocId)
        {
            if (!_graph.ContainsKey(fromDocId))
            {
                _graph[fromDocId] = new List<string>();
            }
            if (!_graph[fromDocId].Contains(toDocId))
            {
                _graph[fromDocId].Add(toDocId);
            }
        }

        protected override async Task<List<Document<T>>> RetrieveCoreAsync(string query, int topK = 5)
        {
            var queryEmbedding = await _embeddingModel.GenerateEmbeddingAsync(query);
            var initialDocs = await _documentStore.SearchAsync(queryEmbedding, Math.Max(topK / 2, 1));

            var visited = new HashSet<string>();
            var results = new Dictionary<string, (Document<T> doc, T score)>();

            foreach (var doc in initialDocs)
            {
                var score = StatisticsHelper.CosineSimilarity(
                    queryEmbedding,
                    doc.Embedding,
                    NumOps);
                results[doc.Id] = (doc, score);
                visited.Add(doc.Id);
            }

            foreach (var initialDoc in initialDocs)
            {
                await TraverseGraph(initialDoc.Id, queryEmbedding, 0, visited, results);
            }

            return results.Values
                .OrderByDescending(x => x.score)
                .Take(topK)
                .Select(x => x.doc)
                .ToList();
        }

        private async Task TraverseGraph(
            string docId,
            Vector<T> queryEmbedding,
            int depth,
            HashSet<string> visited,
            Dictionary<string, (Document<T> doc, T score)> results)
        {
            if (depth >= _maxDepth || !_graph.ContainsKey(docId))
                return;

            foreach (var neighborId in _graph[docId])
            {
                if (visited.Contains(neighborId))
                    continue;

                visited.Add(neighborId);
                var neighbor = await _documentStore.GetDocumentAsync(neighborId);
                
                if (neighbor != null)
                {
                    var score = StatisticsHelper.CosineSimilarity(
                        queryEmbedding,
                        neighbor.Embedding,
                        NumOps);

                    var decayFactor = NumOps.FromDouble(Math.Pow(0.8, depth + 1));
                    var decayedScore = NumOps.Multiply(score, decayFactor);

                    results[neighborId] = (neighbor, decayedScore);
                    await TraverseGraph(neighborId, queryEmbedding, depth + 1, visited, results);
                }
            }
        }
    }
}
