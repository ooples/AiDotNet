using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Filtering;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.Tensors.LinearAlgebra;

using Newtonsoft.Json;
using Npgsql;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores;

/// <summary>
/// A real PostgreSQL + <c>pgvector</c> document store. Persists documents to a table
/// (<c>id text pk, content text, metadata jsonb, embedding vector(dim)</c>) and performs
/// approximate/exact nearest-neighbour search using pgvector's distance operators.
/// </summary>
/// <typeparam name="T">The numeric type used for vector operations.</typeparam>
/// <remarks>
/// <para>
/// Ships in the opt-in <c>AiDotNet.Storage.Postgres</c> package so the core package stays free of
/// the Npgsql dependency (mirroring <c>PostgresGraphCheckpointer</c>). The interface contract is
/// synchronous, so this store uses Npgsql's synchronous API.
/// </para>
/// <para>
/// Vectors are sent as pgvector text literals (<c>[v1,v2,...]</c>) cast to <c>vector</c>; metadata is
/// stored as <c>jsonb</c>. Metadata filtering is pushed into SQL via
/// <see cref="PostgresVectorFilterBuilder"/>. Nearest-neighbour ordering uses the operator chosen by
/// <see cref="DistanceMetricType"/> (see <see cref="PgVectorMetric"/>): <c>&lt;=&gt;</c> (cosine),
/// <c>&lt;-&gt;</c> (L2) or <c>&lt;+&gt;</c> (L1).
/// </para>
/// </remarks>
[ComponentType(ComponentType.DocumentStore)]
[PipelineStage(PipelineStage.Indexing)]
public class PostgresVectorDocumentStore<T> : DocumentStoreBase<T>
{
    private static readonly Regex IdentifierPattern = new("^[A-Za-z_][A-Za-z0-9_]*$", RegexOptions.Compiled);

    private readonly string _connectionString;
    private readonly string _tableName;
    private readonly DistanceMetricType _metric;
    private readonly string _distanceOperator;
    private readonly int _vectorDimension;
    private int _documentCount;
    private bool _schemaReady;

    /// <inheritdoc/>
    public override int DocumentCount => _documentCount;

    /// <inheritdoc/>
    public override int VectorDimension => _vectorDimension;

    /// <summary>Gets the table this store is bound to.</summary>
    public string TableName => _tableName;

    /// <summary>
    /// Initializes a new instance of the <see cref="PostgresVectorDocumentStore{T}"/> class and ensures
    /// the pgvector extension, backing table and index exist.
    /// </summary>
    /// <param name="connectionString">The Npgsql connection string.</param>
    /// <param name="tableName">
    /// The backing table name. Must be a plain SQL identifier (letters, digits and underscores, not
    /// starting with a digit); it is validated and interpolated into DDL/DML, so it cannot contain
    /// arbitrary SQL.
    /// </param>
    /// <param name="vectorDimension">The fixed embedding dimension for the <c>vector(dim)</c> column.</param>
    /// <param name="distanceMetric">The distance metric that selects the pgvector operator.</param>
    /// <exception cref="ArgumentException">Thrown when arguments are empty or the table name is not a valid identifier.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="vectorDimension"/> is not positive.</exception>
    public PostgresVectorDocumentStore(
        string connectionString,
        string tableName,
        int vectorDimension,
        DistanceMetricType distanceMetric = DistanceMetricType.Cosine)
    {
        if (string.IsNullOrWhiteSpace(connectionString))
            throw new ArgumentException("Connection string cannot be empty", nameof(connectionString));
        if (string.IsNullOrWhiteSpace(tableName))
            throw new ArgumentException("Table name cannot be empty", nameof(tableName));
        if (!IdentifierPattern.IsMatch(tableName))
            throw new ArgumentException(
                "Table name must be a valid SQL identifier (letters, digits, underscore; not starting with a digit).",
                nameof(tableName));
        if (vectorDimension <= 0)
            throw new ArgumentOutOfRangeException(nameof(vectorDimension), "Vector dimension must be positive");

        _connectionString = connectionString;
        _tableName = tableName;
        _metric = distanceMetric;
        _distanceOperator = PgVectorMetric.Operator(distanceMetric);
        _vectorDimension = vectorDimension;

        EnsureSchema();
    }

    private NpgsqlConnection Open()
    {
        var connection = new NpgsqlConnection(_connectionString);
        connection.Open();
        return connection;
    }

    private void EnsureSchema()
    {
        using var connection = Open();
        using (var command = connection.CreateCommand())
        {
            command.CommandText =
                "CREATE EXTENSION IF NOT EXISTS vector;" +
                $"CREATE TABLE IF NOT EXISTS {_tableName} (" +
                "id text PRIMARY KEY, " +
                "content text NOT NULL, " +
                "metadata jsonb NOT NULL DEFAULT '{}'::jsonb, " +
                $"embedding vector({_vectorDimension.ToString(CultureInfo.InvariantCulture)}) NOT NULL);";
            command.ExecuteNonQuery();
        }

        using (var countCommand = connection.CreateCommand())
        {
            countCommand.CommandText = $"SELECT count(*) FROM {_tableName};";
            _documentCount = Convert.ToInt32(countCommand.ExecuteScalar(), CultureInfo.InvariantCulture);
        }

        _schemaReady = true;
    }

    private static string FormatVector(Vector<T> embedding)
    {
        var sb = new StringBuilder("[");
        var array = embedding.ToArray();
        for (var i = 0; i < array.Length; i++)
        {
            if (i > 0)
                sb.Append(',');
            sb.Append(Convert.ToDouble(array[i], CultureInfo.InvariantCulture).ToString("R", CultureInfo.InvariantCulture));
        }

        sb.Append(']');
        return sb.ToString();
    }

    /// <inheritdoc/>
    protected override void AddCore(VectorDocument<T> vectorDocument)
    {
        if (vectorDocument.Embedding.Length != _vectorDimension)
            throw new ArgumentException(
                $"Document embedding dimension ({vectorDocument.Embedding.Length}) does not match the store's configured dimension ({_vectorDimension}).");

        using var connection = Open();
        using var command = connection.CreateCommand();
        command.CommandText =
            $"INSERT INTO {_tableName} (id, content, metadata, embedding) " +
            "VALUES (@id, @content, @metadata::jsonb, @embedding::vector) " +
            "ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content, " +
            "metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding " +
            "RETURNING (xmax = 0) AS inserted;";
        AddDocumentParameters(command, vectorDocument);

        var inserted = Convert.ToBoolean(command.ExecuteScalar(), CultureInfo.InvariantCulture);
        if (inserted)
            _documentCount++;
    }

    /// <inheritdoc/>
    protected override void AddBatchCore(IList<VectorDocument<T>> vectorDocuments)
    {
        if (vectorDocuments.Count == 0)
            return;

        using var connection = Open();
        using var transaction = connection.BeginTransaction();
        var insertedCount = 0;

        foreach (var vectorDocument in vectorDocuments)
        {
            if (vectorDocument.Embedding.Length != _vectorDimension)
                throw new ArgumentException(
                    $"Document embedding dimension ({vectorDocument.Embedding.Length}) does not match the store's configured dimension ({_vectorDimension}).");

            using var command = connection.CreateCommand();
            command.Transaction = transaction;
            command.CommandText =
                $"INSERT INTO {_tableName} (id, content, metadata, embedding) " +
                "VALUES (@id, @content, @metadata::jsonb, @embedding::vector) " +
                "ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content, " +
                "metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding " +
                "RETURNING (xmax = 0) AS inserted;";
            AddDocumentParameters(command, vectorDocument);
            if (Convert.ToBoolean(command.ExecuteScalar(), CultureInfo.InvariantCulture))
                insertedCount++;
        }

        transaction.Commit();
        _documentCount += insertedCount;
    }

    private static void AddDocumentParameters(NpgsqlCommand command, VectorDocument<T> vectorDocument)
    {
        command.Parameters.AddWithValue("id", vectorDocument.Document.Id);
        command.Parameters.AddWithValue("content", vectorDocument.Document.Content ?? string.Empty);
        command.Parameters.AddWithValue(
            "metadata",
            JsonConvert.SerializeObject(vectorDocument.Document.Metadata ?? new Dictionary<string, object>()));
        command.Parameters.AddWithValue("embedding", FormatVector(vectorDocument.Embedding));
    }

    /// <inheritdoc/>
    protected override IEnumerable<Document<T>> GetSimilarCore(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters)
    {
        var parameters = new Dictionary<string, object>();
        var whereClause = PostgresVectorFilterBuilder.Build(metadataFilters, parameters);
        return ExecuteSearch(queryVector, topK, whereClause, parameters);
    }

    /// <inheritdoc/>
    protected override IEnumerable<Document<T>> GetSimilarWithFilterCore(Vector<T> queryVector, MetadataFilter filter, int topK)
    {
        var parameters = new Dictionary<string, object>();
        var whereClause = PostgresVectorFilterBuilder.Build(filter, parameters);
        return ExecuteSearch(queryVector, topK, whereClause, parameters);
    }

    /// <inheritdoc/>
    protected override System.Threading.Tasks.Task<IEnumerable<Document<T>>> GetSimilarWithFilterCoreAsync(Vector<T> queryVector, MetadataFilter filter, int topK, System.Threading.CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return System.Threading.Tasks.Task.FromResult(GetSimilarWithFilterCore(queryVector, filter, topK));
    }

    private IEnumerable<Document<T>> ExecuteSearch(Vector<T> queryVector, int topK, string whereClause, Dictionary<string, object> parameters)
    {
        using var connection = Open();
        using var command = connection.CreateCommand();
        command.CommandText =
            $"SELECT id, content, metadata, embedding {_distanceOperator} @q::vector AS distance " +
            $"FROM {_tableName}{whereClause} " +
            $"ORDER BY embedding {_distanceOperator} @q::vector " +
            "LIMIT @k;";
        command.Parameters.AddWithValue("q", FormatVector(queryVector));
        command.Parameters.AddWithValue("k", topK);
        foreach (var parameter in parameters)
            command.Parameters.AddWithValue(parameter.Key, parameter.Value);

        var results = new List<Document<T>>();
        using var reader = command.ExecuteReader();
        while (reader.Read())
        {
            var document = ReadDocument(reader);
            var distance = Convert.ToDouble(reader.GetValue(3), CultureInfo.InvariantCulture);
            document.RelevanceScore = NumOps.FromDouble(PgVectorMetric.ToSimilarity(_metric, distance));
            document.HasRelevanceScore = true;
            results.Add(document);
        }

        return results;
    }

    /// <inheritdoc/>
    protected override Document<T>? GetByIdCore(string documentId)
    {
        using var connection = Open();
        using var command = connection.CreateCommand();
        command.CommandText = $"SELECT id, content, metadata FROM {_tableName} WHERE id = @id;";
        command.Parameters.AddWithValue("id", documentId);

        using var reader = command.ExecuteReader();
        return reader.Read() ? ReadDocument(reader) : null;
    }

    /// <inheritdoc/>
    protected override bool RemoveCore(string documentId)
    {
        using var connection = Open();
        using var command = connection.CreateCommand();
        command.CommandText = $"DELETE FROM {_tableName} WHERE id = @id;";
        command.Parameters.AddWithValue("id", documentId);

        var affected = command.ExecuteNonQuery();
        if (affected > 0 && _documentCount > 0)
        {
            _documentCount--;
            return true;
        }

        return affected > 0;
    }

    /// <inheritdoc/>
    protected override IEnumerable<Document<T>> GetAllCore()
    {
        using var connection = Open();
        using var command = connection.CreateCommand();
        command.CommandText = $"SELECT id, content, metadata FROM {_tableName};";

        var results = new List<Document<T>>();
        using var reader = command.ExecuteReader();
        while (reader.Read())
            results.Add(ReadDocument(reader));

        return results;
    }

    /// <inheritdoc/>
    public override void Clear()
    {
        if (!_schemaReady)
            EnsureSchema();

        using var connection = Open();
        using var command = connection.CreateCommand();
        command.CommandText = $"TRUNCATE TABLE {_tableName};";
        command.ExecuteNonQuery();
        _documentCount = 0;
    }

    private static Document<T> ReadDocument(NpgsqlDataReader reader)
    {
        var id = reader.GetString(0);
        var content = reader.GetString(1);
        var metadataJson = reader.IsDBNull(2) ? "{}" : reader.GetString(2);
        var metadata = JsonConvert.DeserializeObject<Dictionary<string, object>>(metadataJson)
                       ?? new Dictionary<string, object>();
        return new Document<T>(id, content, metadata);
    }
}
