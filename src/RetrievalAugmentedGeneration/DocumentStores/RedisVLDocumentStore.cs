#if NETCOREAPP || NETSTANDARD2_1_OR_GREATER
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using StackExchange.Redis;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores
{
    /// <summary>
    /// Document store using Redis Vector Library for vector search
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations</typeparam>
    public class RedisVLDocumentStore<T> : DocumentStoreBase<T> where T : struct, IComparable, IComparable<T>, IConvertible, IEquatable<T>, IFormattable
    {
        private readonly ConnectionMultiplexer _redis;
        private readonly IDatabase _db;
        private readonly string _indexName;

        public RedisVLDocumentStore(string connectionString, string indexName = "documents")
        {
            if (string.IsNullOrEmpty(connectionString))
                throw new ArgumentException("Connection string cannot be null or empty", nameof(connectionString));

            _redis = ConnectionMultiplexer.Connect(connectionString);
            _db = _redis.GetDatabase();
            _indexName = indexName;
        }

        public override async Task AddDocumentAsync(Document<T> document)
        {
            if (document == null)
                throw new ArgumentNullException(nameof(document));

            var key = $"{_indexName}:{document.Id}";
            var hash = new HashEntry[]
            {
                new HashEntry("id", document.Id),
                new HashEntry("content", document.Content),
                new HashEntry("embedding", JsonSerializer.Serialize(ConvertVectorToDoubleArray(document.Embedding))),
                new HashEntry("metadata", JsonSerializer.Serialize(document.Metadata))
            };

            await _db.HashSetAsync(key, hash);
        }

        public override async Task<List<Document<T>>> SearchAsync(Vector<T> queryEmbedding, int topK = 5)
        {
            if (queryEmbedding == null)
                throw new ArgumentNullException(nameof(queryEmbedding));

            var pattern = $"{_indexName}:*";
            var server = _redis.GetServer(_redis.GetEndPoints()[0]);
            var keys = server.Keys(pattern: pattern).ToList();

            var results = new List<(Document<T> doc, T similarity)>();

            foreach (var key in keys)
            {
                var hash = await _db.HashGetAllAsync(key);
                var hashDict = hash.ToDictionary(x => x.Name.ToString(), x => x.Value.ToString());

                if (!hashDict.ContainsKey("embedding"))
                    continue;

                var embedding = JsonSerializer.Deserialize<double[]>(hashDict["embedding"]) ?? Array.Empty<double>();
                var embeddingVector = new Vector<T>(
                    embedding.Select(x => (T)Convert.ChangeType(x, typeof(T))).ToArray(),
                    NumOps);

                var similarity = StatisticsHelper.CosineSimilarity(queryEmbedding, embeddingVector, NumOps);

                var doc = new Document<T>
                {
                    Id = hashDict.GetValueOrDefault("id", string.Empty),
                    Content = hashDict.GetValueOrDefault("content", string.Empty),
                    Metadata = JsonSerializer.Deserialize<Dictionary<string, string>>(
                        hashDict.GetValueOrDefault("metadata", "{}")) ?? new Dictionary<string, string>()
                };

                results.Add((doc, similarity));
            }

            return results
                .OrderByDescending(x => x.similarity)
                .Take(topK)
                .Select(x => x.doc)
                .ToList();
        }

        public override async Task DeleteDocumentAsync(string documentId)
        {
            if (string.IsNullOrEmpty(documentId))
                throw new ArgumentException("Document ID cannot be null or empty", nameof(documentId));

            var key = $"{_indexName}:{documentId}";
            await _db.KeyDeleteAsync(key);
        }

        public override async Task<Document<T>?> GetDocumentAsync(string documentId)
        {
            if (string.IsNullOrEmpty(documentId))
                throw new ArgumentException("Document ID cannot be null or empty", nameof(documentId));

            var key = $"{_indexName}:{documentId}";
            var hash = await _db.HashGetAllAsync(key);

            if (hash.Length == 0)
                return null;

            var hashDict = hash.ToDictionary(x => x.Name.ToString(), x => x.Value.ToString());

            return new Document<T>
            {
                Id = hashDict.GetValueOrDefault("id", string.Empty),
                Content = hashDict.GetValueOrDefault("content", string.Empty),
                Metadata = JsonSerializer.Deserialize<Dictionary<string, string>>(
                    hashDict.GetValueOrDefault("metadata", "{}")) ?? new Dictionary<string, string>()
            };
        }

        private double[] ConvertVectorToDoubleArray(Vector<T> vector)
        {
            var result = new double[vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                result[i] = Convert.ToDouble(vector[i]);
            }
            return result;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _redis?.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}

#endif
