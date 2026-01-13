using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.Retrievers
{
    /// <summary>
    /// Test helper classes for RAG retriever unit tests
    /// </summary>
    internal static class TestHelpers
    {
        /// <summary>
        /// Creates a simple in-memory document store for testing
        /// </summary>
        public static InMemoryDocumentStore<T> CreateDocumentStore<T>(int vectorDimension = 128)
        {
            return new InMemoryDocumentStore<T>(vectorDimension);
        }

        /// <summary>
        /// Creates a stub embedding model that returns deterministic embeddings
        /// </summary>
        public static StubEmbeddingModel<T> CreateEmbeddingModel<T>(int dimension = 128)
        {
            return new StubEmbeddingModel<T>(dimension);
        }

        /// <summary>
        /// Creates sample documents for testing
        /// </summary>
        public static List<Document<T>> CreateSampleDocuments<T>()
        {
            return new List<Document<T>>
            {
                new Document<T>("doc1", "machine learning algorithms") { Metadata = new Dictionary<string, object> { ["category"] = "AI" } },
                new Document<T>("doc2", "deep neural networks") { Metadata = new Dictionary<string, object> { ["category"] = "AI" } },
                new Document<T>("doc3", "cooking recipes pasta") { Metadata = new Dictionary<string, object> { ["category"] = "Food" } },
                new Document<T>("doc4", "artificial intelligence") { Metadata = new Dictionary<string, object> { ["category"] = "AI" } },
                new Document<T>("doc5", "italian cuisine pizza") { Metadata = new Dictionary<string, object> { ["category"] = "Food" } }
            };
        }
    }

    /// <summary>
    /// Simple in-memory document store for testing
    /// </summary>
    internal class InMemoryDocumentStore<T> : IDocumentStore<T>
    {
        private readonly Dictionary<string, VectorDocument<T>> _documents = new();
        private readonly int _vectorDimension;
        private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

        public int DocumentCount => _documents.Count;
        public int VectorDimension => _vectorDimension;

        public InMemoryDocumentStore(int vectorDimension)
        {
            _vectorDimension = vectorDimension;
        }

        public void Add(VectorDocument<T> vectorDocument)
        {
            if (vectorDocument == null)
                throw new ArgumentNullException(nameof(vectorDocument));
            if (vectorDocument.Embedding == null || vectorDocument.Embedding.Length != _vectorDimension)
                throw new ArgumentException($"Embedding must have dimension {_vectorDimension}");

            _documents[vectorDocument.Document.Id] = vectorDocument;
        }

        public void AddBatch(IEnumerable<VectorDocument<T>> vectorDocuments)
        {
            foreach (var doc in vectorDocuments)
            {
                Add(doc);
            }
        }

        public IEnumerable<Document<T>> GetSimilar(Vector<T> queryVector, int topK)
        {
            return GetSimilarWithFilters(queryVector, topK, new Dictionary<string, object>());
        }

        public IEnumerable<Document<T>> GetSimilarWithFilters(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters)
        {
            if (queryVector == null)
                throw new ArgumentNullException(nameof(queryVector));
            if (topK <= 0)
                throw new ArgumentException("TopK must be positive", nameof(topK));

            var candidates = _documents.Values
                .Where(d => MatchesFilters(d, metadataFilters))
                .Select(vd =>
                {
                    var similarity = CosineSimilarity(queryVector, vd.Embedding);
                    // Normalize cosine similarity from [-1, 1] to [0, 1] range
                    var normalizedScore = NumOps.Divide(
                        NumOps.Add(similarity, NumOps.One),
                        NumOps.FromDouble(2.0));
                    var doc = new Document<T>(vd.Document.Id, vd.Document.Content, vd.Document.Metadata)
                    {
                        Embedding = vd.Embedding,
                        RelevanceScore = normalizedScore,
                        HasRelevanceScore = true
                    };
                    return doc;
                })
                .OrderByDescending(d => d.RelevanceScore)
                .Take(topK);

            return candidates.ToList();
        }

        public Document<T>? GetById(string documentId)
        {
            if (_documents.TryGetValue(documentId, out var vd))
            {
                return new Document<T>(vd.Document.Id, vd.Document.Content, vd.Document.Metadata) { Embedding = vd.Embedding };
            }
            return null;
        }

        public bool Remove(string documentId)
        {
            return _documents.Remove(documentId);
        }

        public void Clear()
        {
            _documents.Clear();
        }

        public IEnumerable<Document<T>> GetAll()
        {
            return _documents.Values.Select(vd =>
                new Document<T>(vd.Document.Id, vd.Document.Content, vd.Document.Metadata) { Embedding = vd.Embedding }
            ).ToList();
        }

        private bool MatchesFilters(VectorDocument<T> document, Dictionary<string, object> filters)
        {
            if (filters == null || filters.Count == 0)
                return true;

            foreach (var filter in filters)
            {
                if (!document.Document.Metadata.TryGetValue(filter.Key, out var value))
                    return false;
                if (!filter.Value.Equals(value))
                    return false;
            }
            return true;
        }

        private T CosineSimilarity(Vector<T> v1, Vector<T> v2)
        {
            if (v1.Length != v2.Length)
                throw new ArgumentException("Vectors must have same dimension");

            var dotProduct = NumOps.Zero;
            var norm1 = NumOps.Zero;
            var norm2 = NumOps.Zero;

            for (int i = 0; i < v1.Length; i++)
            {
                dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(v1[i], v2[i]));
                norm1 = NumOps.Add(norm1, NumOps.Multiply(v1[i], v1[i]));
                norm2 = NumOps.Add(norm2, NumOps.Multiply(v2[i], v2[i]));
            }

            var denominator = NumOps.Multiply(
                NumOps.FromDouble(Math.Sqrt(Convert.ToDouble(norm1))),
                NumOps.FromDouble(Math.Sqrt(Convert.ToDouble(norm2)))
            );

            if (NumOps.Equals(denominator, NumOps.Zero))
                return NumOps.Zero;

            return NumOps.Divide(dotProduct, denominator);
        }
    }

    /// <summary>
    /// Stub embedding model that returns deterministic embeddings based on text hash
    /// </summary>
    internal class StubEmbeddingModel<T> : IEmbeddingModel<T>
    {
        private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

        public int EmbeddingDimension { get; }
        public int MaxTokens { get; } = 512;

        public StubEmbeddingModel(int embeddingDimension)
        {
            EmbeddingDimension = embeddingDimension;
        }

        public Vector<T> Embed(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                throw new ArgumentException("Text cannot be null or empty", nameof(text));

            // Create deterministic embedding based on text hash
            var hash = text.GetHashCode();
            var random = RandomHelper.CreateSeededRandom(hash);
            var values = new T[EmbeddingDimension];

            for (int i = 0; i < EmbeddingDimension; i++)
            {
                values[i] = NumOps.FromDouble(random.NextDouble() * 2 - 1); // Range: -1 to 1
            }

            // Normalize the vector
            var norm = NumOps.Zero;
            for (int i = 0; i < EmbeddingDimension; i++)
            {
                norm = NumOps.Add(norm, NumOps.Multiply(values[i], values[i]));
            }
            norm = NumOps.FromDouble(Math.Sqrt(Convert.ToDouble(norm)));

            if (!NumOps.Equals(norm, NumOps.Zero))
            {
                for (int i = 0; i < EmbeddingDimension; i++)
                {
                    values[i] = NumOps.Divide(values[i], norm);
                }
            }

            return new Vector<T>(values);
        }

        public Task<Vector<T>> EmbedAsync(string text)
        {
            return Task.FromResult(Embed(text));
        }

        public Matrix<T> EmbedBatch(IEnumerable<string> texts)
        {
            var textList = texts.ToList();
            var embeddings = new T[textList.Count, EmbeddingDimension];

            for (int i = 0; i < textList.Count; i++)
            {
                var embedding = Embed(textList[i]);
                for (int j = 0; j < EmbeddingDimension; j++)
                {
                    embeddings[i, j] = embedding[j];
                }
            }

            return new Matrix<T>(embeddings);
        }

        public Task<Matrix<T>> EmbedBatchAsync(IEnumerable<string> texts)
        {
            return Task.FromResult(EmbedBatch(texts));
        }
    }
}
