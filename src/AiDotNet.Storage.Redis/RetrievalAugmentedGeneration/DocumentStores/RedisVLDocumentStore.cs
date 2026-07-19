using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;

using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.Tensors.LinearAlgebra;

using Newtonsoft.Json;
using StackExchange.Redis;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores;

/// <summary>
/// A real Redis + RediSearch (RedisVL-style) document store. Each document is a Redis HASH; a
/// RediSearch index provides KNN vector search with server-side metadata filtering.
/// </summary>
/// <typeparam name="T">The numeric type used for vector operations.</typeparam>
/// <remarks>
/// <para>
/// Ships in the opt-in <c>AiDotNet.Storage.Redis</c> package so the core package stays free of the
/// StackExchange.Redis dependency (mirroring <c>RedisGraphCheckpointer</c>). Requires a Redis server
/// with the RediSearch module (e.g. Redis Stack).
/// </para>
/// <para>
/// The index is created with <c>FT.CREATE ... ON HASH</c> declaring a <c>VECTOR</c> field (HNSW,
/// FLOAT32) plus any caller-declared <see cref="RedisVectorField"/> TAG/NUMERIC fields. Documents are
/// written with <c>HSET</c>; search uses <c>FT.SEARCH</c> KNN. Metadata filters on declared fields are
/// pushed into the query (<see cref="RedisVectorQueryBuilder"/>); filters on undeclared keys are
/// honoured by in-memory post-filtering of an overfetched candidate set. Returned documents are always
/// freshly reconstructed from Redis, so nothing cached is ever mutated.
/// </para>
/// </remarks>
[ComponentType(ComponentType.DocumentStore)]
[PipelineStage(PipelineStage.Indexing)]
public class RedisVLDocumentStore<T> : DocumentStoreBase<T>, IDisposable
{
    private const string IdField = "_id";
    private const string ContentField = "_content";
    private const string MetadataField = "metadata";
    private const string EmbeddingField = "embedding";
    private const string ScoreField = "__vec_score";
    private const int CandidateMultiplier = 10;

    private static readonly Regex IdentifierPattern = new("^[A-Za-z_][A-Za-z0-9_]*$", RegexOptions.Compiled);

    private readonly IConnectionMultiplexer _connection;
    private readonly bool _ownsConnection;
    private readonly string _indexName;
    private readonly string _keyPrefix;
    private readonly int _vectorDimension;
    private readonly DistanceMetricType _metric;
    private readonly string _distanceMetric;
    private readonly IReadOnlyList<RedisVectorField> _fields;
    private readonly Dictionary<string, RedisVectorFieldType> _fieldTypes;
    private int _documentCount;

    /// <inheritdoc/>
    public override int DocumentCount => _documentCount;

    /// <inheritdoc/>
    public override int VectorDimension => _vectorDimension;

    /// <summary>Gets the RediSearch index name this store is bound to.</summary>
    public string IndexName => _indexName;

    /// <summary>
    /// Initializes a new instance of the <see cref="RedisVLDocumentStore{T}"/> class from a connection
    /// string, creating the RediSearch index if it does not already exist.
    /// </summary>
    /// <param name="connectionString">The StackExchange.Redis connection string.</param>
    /// <param name="indexName">The RediSearch index name (a plain identifier).</param>
    /// <param name="vectorDimension">The fixed embedding dimension.</param>
    /// <param name="distanceMetric">The distance metric (Cosine or Euclidean).</param>
    /// <param name="filterableFields">Optional metadata fields to index so their filters push server-side.</param>
    /// <param name="keyPrefix">Optional Redis key prefix; defaults to <c>{indexName}:doc:</c>.</param>
    public RedisVLDocumentStore(
        string connectionString,
        string indexName,
        int vectorDimension,
        DistanceMetricType distanceMetric = DistanceMetricType.Cosine,
        IEnumerable<RedisVectorField>? filterableFields = null,
        string? keyPrefix = null)
        : this(Connect(connectionString), true, indexName, vectorDimension, distanceMetric, filterableFields, keyPrefix)
    {
    }

    /// <summary>
    /// Initializes a new instance over an existing connection multiplexer (not owned/disposed by this instance).
    /// </summary>
    public RedisVLDocumentStore(
        IConnectionMultiplexer connection,
        string indexName,
        int vectorDimension,
        DistanceMetricType distanceMetric = DistanceMetricType.Cosine,
        IEnumerable<RedisVectorField>? filterableFields = null,
        string? keyPrefix = null)
        : this(connection ?? throw new ArgumentNullException(nameof(connection)),
               false, indexName, vectorDimension, distanceMetric, filterableFields, keyPrefix)
    {
    }

    private RedisVLDocumentStore(
        IConnectionMultiplexer connection,
        bool ownsConnection,
        string indexName,
        int vectorDimension,
        DistanceMetricType distanceMetric,
        IEnumerable<RedisVectorField>? filterableFields,
        string? keyPrefix)
    {
        if (string.IsNullOrWhiteSpace(indexName) || !IdentifierPattern.IsMatch(indexName))
            throw new ArgumentException("Index name must be a valid identifier.", nameof(indexName));
        if (vectorDimension <= 0)
            throw new ArgumentException("Vector dimension must be positive", nameof(vectorDimension));

        _connection = connection;
        _ownsConnection = ownsConnection;
        _indexName = indexName;
        _keyPrefix = string.IsNullOrWhiteSpace(keyPrefix) ? indexName + ":doc:" : keyPrefix!;
        _vectorDimension = vectorDimension;
        _metric = distanceMetric;
        _distanceMetric = MapDistance(distanceMetric);
        _fields = (filterableFields ?? Enumerable.Empty<RedisVectorField>()).ToList();
        _fieldTypes = _fields.ToDictionary(f => f.Name, f => f.Type);

        EnsureIndex();
    }

    private static IConnectionMultiplexer Connect(string connectionString)
    {
        if (string.IsNullOrWhiteSpace(connectionString))
            throw new ArgumentException("Connection string cannot be empty", nameof(connectionString));
        return ConnectionMultiplexer.Connect(connectionString);
    }

    private static string MapDistance(DistanceMetricType metric)
    {
        switch (metric)
        {
            case DistanceMetricType.Cosine:
                return "COSINE";
            case DistanceMetricType.Euclidean:
                return "L2";
            default:
                throw new NotSupportedException(
                    $"Distance metric '{metric}' is not supported by RediSearch. Use Cosine or Euclidean.");
        }
    }

    private IDatabase Db => _connection.GetDatabase();

    private void EnsureIndex()
    {
        try
        {
            Db.Execute("FT.INFO", _indexName);
        }
        catch (RedisServerException)
        {
            CreateIndex();
        }

        _documentCount = CountFromServer();
    }

    private void CreateIndex()
    {
        var args = new List<object>
        {
            _indexName, "ON", "HASH", "PREFIX", "1", _keyPrefix, "SCHEMA",
            ContentField, "TEXT",
        };

        foreach (var field in _fields)
        {
            args.Add(field.Name);
            args.Add(field.Type == RedisVectorFieldType.Numeric ? "NUMERIC" : "TAG");
        }

        args.Add(EmbeddingField);
        args.Add("VECTOR");
        args.Add("HNSW");
        args.Add("6");
        args.Add("TYPE");
        args.Add("FLOAT32");
        args.Add("DIM");
        args.Add(_vectorDimension.ToString(CultureInfo.InvariantCulture));
        args.Add("DISTANCE_METRIC");
        args.Add(_distanceMetric);

        Db.Execute("FT.CREATE", args);
    }

    private int CountFromServer()
    {
        var result = Db.Execute("FT.SEARCH", _indexName, "*", "LIMIT", "0", "0");
        var array = (RedisResult[]?)result;
        if (array == null || array.Length == 0)
            return 0;
        return (int)(long)array[0];
    }

    private static byte[] ToBytes(Vector<T> vector)
    {
        var array = vector.ToArray();
        var bytes = new byte[array.Length * sizeof(float)];
        for (var i = 0; i < array.Length; i++)
        {
            var value = (float)Convert.ToDouble(array[i], CultureInfo.InvariantCulture);
            var encoded = BitConverter.GetBytes(value);
            Buffer.BlockCopy(encoded, 0, bytes, i * sizeof(float), sizeof(float));
        }

        return bytes;
    }

    private static string Stringify(object? value)
    {
        if (value is bool b)
            return b ? "true" : "false";
        if (value is IFormattable formattable)
            return formattable.ToString(null, CultureInfo.InvariantCulture);
        return value?.ToString() ?? string.Empty;
    }

    /// <inheritdoc/>
    protected override void AddCore(VectorDocument<T> vectorDocument)
    {
        if (vectorDocument.Embedding.Length != _vectorDimension)
            throw new ArgumentException(
                $"Document embedding dimension ({vectorDocument.Embedding.Length}) does not match the store's configured dimension ({_vectorDimension}).");

        var key = _keyPrefix + vectorDocument.Document.Id;
        var existed = Db.KeyExists(key);
        Db.HashSet(key, BuildEntries(vectorDocument));
        if (!existed)
            _documentCount++;
    }

    /// <inheritdoc/>
    protected override void AddBatchCore(IList<VectorDocument<T>> vectorDocuments)
    {
        if (vectorDocuments.Count == 0)
            return;

        var db = Db;
        foreach (var vd in vectorDocuments)
        {
            if (vd.Embedding.Length != _vectorDimension)
                throw new ArgumentException(
                    $"Document embedding dimension ({vd.Embedding.Length}) does not match the store's configured dimension ({_vectorDimension}).");
        }

        // Determine which ids are new (for accurate counting) before the pipelined writes.
        var existsTasks = new List<Task<bool>>(vectorDocuments.Count);
        foreach (var vd in vectorDocuments)
            existsTasks.Add(db.KeyExistsAsync(_keyPrefix + vd.Document.Id));

        var writeBatch = db.CreateBatch();
        var writeTasks = new List<Task>(vectorDocuments.Count);
        foreach (var vd in vectorDocuments)
            writeTasks.Add(writeBatch.HashSetAsync(_keyPrefix + vd.Document.Id, BuildEntries(vd)));

        // existsTasks were created on 'db' directly (already dispatched); await them, then run writes.
        var existed = existsTasks.Select(t => t.GetAwaiter().GetResult()).ToArray();
        writeBatch.Execute();
        foreach (var t in writeTasks)
            t.GetAwaiter().GetResult();

        _documentCount += existed.Count(e => !e);
    }

    private HashEntry[] BuildEntries(VectorDocument<T> vectorDocument)
    {
        var metadata = vectorDocument.Document.Metadata ?? new Dictionary<string, object>();
        var entries = new List<HashEntry>
        {
            new(IdField, vectorDocument.Document.Id),
            new(ContentField, vectorDocument.Document.Content ?? string.Empty),
            new(MetadataField, JsonConvert.SerializeObject(metadata)),
            new(EmbeddingField, ToBytes(vectorDocument.Embedding)),
        };

        foreach (var field in _fields)
        {
            if (metadata.TryGetValue(field.Name, out var value))
                entries.Add(new HashEntry(field.Name, Stringify(value)));
        }

        return entries.ToArray();
    }

    /// <inheritdoc/>
    protected override IEnumerable<Document<T>> GetSimilarCore(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters)
    {
        var expr = RedisVectorQueryBuilder.BuildFilterExpression(metadataFilters, _fieldTypes, out var unpushedKeys);
        var knnK = unpushedKeys.Count > 0 ? topK * CandidateMultiplier : topK;

        var query = $"{expr}=>[KNN {knnK.ToString(CultureInfo.InvariantCulture)} @{EmbeddingField} $BLOB AS {ScoreField}]";
        var args = new List<object>
        {
            _indexName, query,
            "PARAMS", "2", "BLOB", ToBytes(queryVector),
            "SORTBY", ScoreField,
            "RETURN", "4", IdField, ContentField, MetadataField, ScoreField,
            "DIALECT", "2",
            "LIMIT", "0", knnK.ToString(CultureInfo.InvariantCulture),
        };

        var parsed = ParseSearch(Db.Execute("FT.SEARCH", args), includeScore: true);

        IEnumerable<(Document<T> Doc, double Distance)> candidates = parsed;
        if (unpushedKeys.Count > 0)
        {
            var postFilters = unpushedKeys
                .Where(metadataFilters.ContainsKey)
                .ToDictionary(k => k, k => metadataFilters[k]);
            candidates = candidates.Where(c => MatchesFilters(c.Doc, postFilters));
        }

        var results = new List<Document<T>>();
        foreach (var candidate in candidates.Take(topK))
        {
            candidate.Doc.RelevanceScore = NumOps.FromDouble(ToSimilarity(candidate.Distance));
            candidate.Doc.HasRelevanceScore = true;
            results.Add(candidate.Doc);
        }

        return results;
    }

    private double ToSimilarity(double distance)
    {
        return _metric == DistanceMetricType.Cosine ? 1.0 - distance : 1.0 / (1.0 + distance);
    }

    private List<(Document<T> Doc, double Distance)> ParseSearch(RedisResult result, bool includeScore)
    {
        var docs = new List<(Document<T>, double)>();
        var array = (RedisResult[]?)result;
        if (array == null || array.Length < 1)
            return docs;

        // array[0] is the total count; then repeating (key, fields[]) pairs.
        for (var i = 1; i + 1 < array.Length; i += 2)
        {
            var fields = (RedisResult[]?)array[i + 1];
            if (fields == null)
                continue;

            var map = new Dictionary<string, string>();
            for (var j = 0; j + 1 < fields.Length; j += 2)
                map[(string)fields[j]!] = (string)fields[j + 1]!;

            var doc = BuildDocument(map);
            if (doc == null)
                continue;

            var distance = 0.0;
            if (includeScore && map.TryGetValue(ScoreField, out var scoreText))
                double.TryParse(scoreText, NumberStyles.Float, CultureInfo.InvariantCulture, out distance);

            docs.Add((doc, distance));
        }

        return docs;
    }

    private static Document<T>? BuildDocument(IReadOnlyDictionary<string, string> map)
    {
        if (!map.TryGetValue(IdField, out var id) || string.IsNullOrEmpty(id))
            return null;

        map.TryGetValue(ContentField, out var content);
        map.TryGetValue(MetadataField, out var metadataJson);
        var metadata = string.IsNullOrEmpty(metadataJson)
            ? new Dictionary<string, object>()
            : JsonConvert.DeserializeObject<Dictionary<string, object>>(metadataJson!) ?? new Dictionary<string, object>();

        return new Document<T>(id, content ?? string.Empty, metadata);
    }

    /// <inheritdoc/>
    protected override Document<T>? GetByIdCore(string documentId)
    {
        var entries = Db.HashGetAll(_keyPrefix + documentId);
        if (entries.Length == 0)
            return null;

        var map = entries.ToDictionary(e => (string)e.Name!, e => (string)e.Value!);
        return BuildDocument(map);
    }

    /// <inheritdoc/>
    protected override bool RemoveCore(string documentId)
    {
        var removed = Db.KeyDelete(_keyPrefix + documentId);
        if (removed && _documentCount > 0)
            _documentCount--;
        return removed;
    }

    /// <inheritdoc/>
    protected override IEnumerable<Document<T>> GetAllCore()
    {
        var all = new List<Document<T>>();
        var offset = 0;
        const int batchSize = 1000;

        while (true)
        {
            var args = new List<object>
            {
                _indexName, "*",
                "RETURN", "3", IdField, ContentField, MetadataField,
                "DIALECT", "2",
                "LIMIT", offset.ToString(CultureInfo.InvariantCulture), batchSize.ToString(CultureInfo.InvariantCulture),
            };

            var parsed = ParseSearch(Db.Execute("FT.SEARCH", args), includeScore: false);
            if (parsed.Count == 0)
                break;

            all.AddRange(parsed.Select(p => p.Doc));
            offset += batchSize;
        }

        return all;
    }

    /// <inheritdoc/>
    public override void Clear()
    {
        try
        {
            Db.Execute("FT.DROPINDEX", _indexName, "DD");
        }
        catch (RedisServerException)
        {
            // Index did not exist; nothing to drop.
        }

        CreateIndex();
        _documentCount = 0;
    }

    /// <summary>Disposes the owned connection multiplexer, if this instance created it.</summary>
    public void Dispose()
    {
        if (_ownsConnection)
            _connection.Dispose();
    }
}
