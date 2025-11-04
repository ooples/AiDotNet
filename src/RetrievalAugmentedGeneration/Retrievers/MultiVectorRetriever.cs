using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers
{
    /// <summary>
    /// Multi-vector retriever that generates multiple embeddings per document
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations</typeparam>
    public class MultiVectorRetriever<T> : RetrieverBase<T> where T : struct, IComparable, IComparable<T>, IConvertible, IEquatable<T>, IFormattable
    {
        private readonly IEmbeddingModel<T> _embeddingModel;
        private readonly IDocumentStore<T> _documentStore;
        private readonly Dictionary<string, List<Vector<T>>> _documentVectors;

        public MultiVectorRetriever(IEmbeddingModel<T> embeddingModel, IDocumentStore<T> documentStore)
        {
            _embeddingModel = embeddingModel ?? throw new ArgumentNullException(nameof(embeddingModel));
            _documentStore = documentStore ?? throw new ArgumentNullException(nameof(documentStore));
            _documentVectors = new Dictionary<string, List<Vector<T>>>();
        }

        public async Task IndexDocumentAsync(Document<T> document)
        {
            if (document == null)
                throw new ArgumentNullException(nameof(document));

            var sentences = document.Content.Split(new[] { '.', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);
            var vectors = new List<Vector<T>>();

            foreach (var sentence in sentences.Where(s => !string.IsNullOrWhiteSpace(s)))
            {
                var embedding = await _embeddingModel.GenerateEmbeddingAsync(sentence.Trim());
                vectors.Add(embedding);
            }

            _documentVectors[document.Id] = vectors;
            await _documentStore.AddDocumentAsync(document);
        }

        protected override async Task<List<Document<T>>> RetrieveCoreAsync(string query, int topK = 5)
        {
            var queryEmbedding = await _embeddingModel.GenerateEmbeddingAsync(query);
            var scoredDocuments = new List<(Document<T> doc, T score)>();

            foreach (var (docId, vectors) in _documentVectors)
            {
                var maxScore = NumOps.FromDouble(-1.0);

                foreach (var vector in vectors)
                {
                    var score = StatisticsHelper.CosineSimilarity(queryEmbedding, vector, NumOps);
                    if (NumOps.GreaterThan(score, maxScore))
                    {
                        maxScore = score;
                    }
                }

                var document = await _documentStore.GetDocumentAsync(docId);
                if (document != null)
                {
                    scoredDocuments.Add((document, maxScore));
                }
            }

            return scoredDocuments
                .OrderByDescending(x => x.score)
                .Take(topK)
                .Select(x => x.doc)
                .ToList();
        }
    }
}
