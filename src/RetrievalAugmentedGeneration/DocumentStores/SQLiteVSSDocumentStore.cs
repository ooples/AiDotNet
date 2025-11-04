using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System;
using System.Collections.Generic;
using System.Data.SQLite;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores
{
    /// <summary>
    /// Document store using SQLite VSS extension for vector search
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations</typeparam>
    public class SQLiteVSSDocumentStore<T> : DocumentStoreBase<T> where T : struct, IComparable, IComparable<T>, IConvertible, IEquatable<T>, IFormattable
    {
        private readonly string _connectionString;
        private readonly string _tableName;

        public SQLiteVSSDocumentStore(string databasePath, string tableName = "documents")
        {
            if (string.IsNullOrEmpty(databasePath))
                throw new ArgumentException("Database path cannot be null or empty", nameof(databasePath));

            _connectionString = $"Data Source={databasePath};Version=3;";
            _tableName = tableName;
            InitializeDatabase().Wait();
        }

        private async Task InitializeDatabase()
        {
            using var connection = new SQLiteConnection(_connectionString);
            await connection.OpenAsync();

            var createTableSql = $@"
                CREATE TABLE IF NOT EXISTS {_tableName} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    metadata TEXT
                )";

            using var command = new SQLiteCommand(createTableSql, connection);
            await command.ExecuteNonQueryAsync();
        }

        public override async Task AddDocumentAsync(Document<T> document)
        {
            if (document == null)
                throw new ArgumentNullException(nameof(document));

            using var connection = new SQLiteConnection(_connectionString);
            await connection.OpenAsync();

            var insertSql = $@"
                INSERT OR REPLACE INTO {_tableName} (id, content, embedding, metadata)
                VALUES (@id, @content, @embedding, @metadata)";

            using var command = new SQLiteCommand(insertSql, connection);
            command.Parameters.AddWithValue("@id", document.Id);
            command.Parameters.AddWithValue("@content", document.Content);
            command.Parameters.AddWithValue("@embedding",
                JsonSerializer.SerializeToUtf8Bytes(ConvertVectorToDoubleArray(document.Embedding)));
            command.Parameters.AddWithValue("@metadata", JsonSerializer.Serialize(document.Metadata));

            await command.ExecuteNonQueryAsync();
        }

        public override async Task<List<Document<T>>> SearchAsync(Vector<T> queryEmbedding, int topK = 5)
        {
            if (queryEmbedding == null)
                throw new ArgumentNullException(nameof(queryEmbedding));

            using var connection = new SQLiteConnection(_connectionString);
            await connection.OpenAsync();

            var selectSql = $"SELECT id, content, embedding, metadata FROM {_tableName}";
            using var command = new SQLiteCommand(selectSql, connection);
            using var reader = await command.ExecuteReaderAsync();

            var results = new List<(Document<T> doc, T similarity)>();

            while (await reader.ReadAsync())
            {
                var embeddingBytes = (byte[])reader["embedding"];
                var embedding = JsonSerializer.Deserialize<double[]>(embeddingBytes) ?? Array.Empty<double>();
                var embeddingVector = new Vector<T>(
                    embedding.Select(x => (T)Convert.ChangeType(x, typeof(T))).ToArray(),
                    NumOps);

                var similarity = StatisticsHelper.CosineSimilarity(queryEmbedding, embeddingVector, NumOps);

                var doc = new Document<T>
                {
                    Id = reader["id"].ToString() ?? string.Empty,
                    Content = reader["content"].ToString() ?? string.Empty,
                    Metadata = JsonSerializer.Deserialize<Dictionary<string, string>>(
                        reader["metadata"].ToString() ?? "{}") ?? new Dictionary<string, string>()
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

            using var connection = new SQLiteConnection(_connectionString);
            await connection.OpenAsync();

            var deleteSql = $"DELETE FROM {_tableName} WHERE id = @id";
            using var command = new SQLiteCommand(deleteSql, connection);
            command.Parameters.AddWithValue("@id", documentId);

            await command.ExecuteNonQueryAsync();
        }

        public override async Task<Document<T>?> GetDocumentAsync(string documentId)
        {
            if (string.IsNullOrEmpty(documentId))
                throw new ArgumentException("Document ID cannot be null or empty", nameof(documentId));

            using var connection = new SQLiteConnection(_connectionString);
            await connection.OpenAsync();

            var selectSql = $"SELECT id, content, embedding, metadata FROM {_tableName} WHERE id = @id";
            using var command = new SQLiteCommand(selectSql, connection);
            command.Parameters.AddWithValue("@id", documentId);

            using var reader = await command.ExecuteReaderAsync();
            if (!await reader.ReadAsync())
                return null;

            return new Document<T>
            {
                Id = reader["id"].ToString() ?? string.Empty,
                Content = reader["content"].ToString() ?? string.Empty,
                Metadata = JsonSerializer.Deserialize<Dictionary<string, string>>(
                    reader["metadata"].ToString() ?? "{}") ?? new Dictionary<string, string>()
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
    }
}
