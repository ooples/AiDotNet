using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Filtering;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores
{
    /// <summary>
    /// Milvus vector-database document store backed by the real Milvus v2 REST API (<c>/v2/vectordb/*</c>).
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    /// <remarks>
    /// <para>
    /// This store talks to a running Milvus instance over HTTP. It manages a single collection with a
    /// VarChar primary key and dynamic fields enabled, upserts entities (vector + fields), performs
    /// filtered vector search, deletes entities by id and queries the whole collection with paging.
    /// The original content is stored under <c>content</c>; the full metadata dictionary is stored as a
    /// JSON string under <c>metadata_json</c> (for lossless round-tripping) while each scalar metadata
    /// entry is also flattened to a <c>m_&lt;key&gt;</c> dynamic field so it can be used in filter
    /// expressions.
    /// </para>
    /// <para><b>For Beginners:</b> Milvus is an open-source vector database. This class is a real client
    /// for it - every method here makes an HTTP call to your Milvus server using its REST API. Milvus
    /// wraps every response in a <c>{ "code": 0, "data": ... }</c> envelope; a non-zero code means the
    /// operation failed.
    /// </para>
    /// </remarks>
    [ComponentType(ComponentType.DocumentStore)]
    [PipelineStage(PipelineStage.Indexing)]
    public class MilvusDocumentStore<T> : DocumentStoreBase<T>
    {
        private const string IdField = "id";
        private const string VectorField = "vector";
        private const string ContentField = "content";
        private const string MetadataField = "metadata_json";
        private const string MetaPrefix = "m_";

        private readonly HttpClient _httpClient;
        private readonly string _collectionName;
        private readonly string _metricType;
        private int _vectorDimension;
        private int _documentCount;
        private bool _collectionReady;

        /// <summary>
        /// Gets the number of documents (entities) currently stored in the collection.
        /// </summary>
        public override int DocumentCount => _documentCount;

        /// <summary>
        /// Gets the dimensionality of vectors stored in this collection.
        /// </summary>
        public override int VectorDimension => _vectorDimension;

        /// <summary>
        /// Gets the name of the Milvus collection this store is bound to.
        /// </summary>
        public string CollectionName { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="MilvusDocumentStore{T}"/> class.
        /// </summary>
        /// <param name="collectionName">The Milvus collection name.</param>
        /// <param name="url">The base URL of the Milvus server, e.g. <c>http://localhost:19530</c>.</param>
        /// <param name="token">Optional Milvus token (sent as an <c>Authorization: Bearer</c> header).</param>
        /// <param name="distanceMetric">The metric type used when creating the collection.</param>
        /// <param name="vectorDimension">
        /// The vector dimension used to create the collection if it does not already exist.
        /// When 0 the collection must already exist or the dimension is inferred from the first document.
        /// </param>
        /// <param name="httpClient">Optional pre-configured <see cref="HttpClient"/> (primarily for testing).</param>
        /// <param name="handler">Optional <see cref="HttpMessageHandler"/> used to build the client (for testing).</param>
        public MilvusDocumentStore(
            string collectionName,
            string url,
            string? token = null,
            DistanceMetricType distanceMetric = DistanceMetricType.Cosine,
            int vectorDimension = 0,
            HttpClient? httpClient = null,
            HttpMessageHandler? handler = null)
        {
            if (string.IsNullOrWhiteSpace(collectionName))
                throw new ArgumentException("Collection name cannot be empty", nameof(collectionName));
            if (vectorDimension < 0)
                throw new ArgumentOutOfRangeException(nameof(vectorDimension), "Vector dimension cannot be negative");

            _collectionName = collectionName;
            CollectionName = collectionName;
            _metricType = MapMetric(distanceMetric);
            _vectorDimension = vectorDimension;
            _documentCount = 0;

            _httpClient = httpClient ?? (handler != null ? new HttpClient(handler) : new HttpClient());
            if (_httpClient.BaseAddress == null)
            {
                if (string.IsNullOrWhiteSpace(url))
                    throw new ArgumentException("Url cannot be empty", nameof(url));
                _httpClient.BaseAddress = new Uri(url);
            }

            if (!string.IsNullOrWhiteSpace(token) && _httpClient.DefaultRequestHeaders.Authorization == null)
                _httpClient.DefaultRequestHeaders.Add("Authorization", "Bearer " + token);

            InitializeCollection(vectorDimension);
        }

        private static string MapMetric(DistanceMetricType metric)
        {
            switch (metric)
            {
                case DistanceMetricType.Cosine:
                    return "COSINE";
                case DistanceMetricType.Euclidean:
                    return "L2";
                default:
                    throw new NotSupportedException(
                        $"Metric '{metric}' is not supported by Milvus. Use Cosine (COSINE) or Euclidean (L2).");
            }
        }

        private void InitializeCollection(int requestedDimension)
        {
            HttpResponseInfo info;
            try
            {
                info = SendAsync("/v2/vectordb/collections/describe",
                    new { collectionName = _collectionName }).GetAwaiter().GetResult();
            }
            catch (HttpRequestException)
            {
                // Server not reachable at construction time; defer creation to first write.
                return;
            }

            if (IsSuccess(info.Status))
            {
                var root = JObject.Parse(info.Body);
                var code = root["code"]?.Value<int>() ?? 0;
                if (code == 0)
                {
                    _collectionReady = true;
                    var fields = root["data"]?["fields"] as JArray;
                    if (fields != null)
                    {
                        foreach (var field in fields)
                        {
                            var dim = field?["params"]?["dim"];
                            if (dim != null && dim.Type != JTokenType.Null)
                            {
                                var parsed = Convert.ToInt32(dim, CultureInfo.InvariantCulture);
                                if (parsed > 0)
                                    _vectorDimension = parsed;
                            }
                        }
                    }
                    return;
                }
            }

            if (requestedDimension > 0)
                CreateCollection(requestedDimension);
        }

        private void CreateCollection(int dimension)
        {
            var body = new
            {
                collectionName = _collectionName,
                dimension,
                metricType = _metricType,
                idType = "VarChar",
                primaryFieldName = IdField,
                vectorFieldName = VectorField,
                enableDynamicField = true
            };

            var info = SendAsync("/v2/vectordb/collections/create", body).GetAwaiter().GetResult();
            EnsureSuccess(info, "create collection");
            _vectorDimension = dimension;
            _collectionReady = true;
        }

        private void EnsureCollectionForDimension(int dimension)
        {
            if (_collectionReady)
                return;
            CreateCollection(dimension);
        }

        /// <inheritdoc/>
        protected override void AddCore(VectorDocument<T> vectorDocument)
            => AddCoreImplAsync(vectorDocument, CancellationToken.None).GetAwaiter().GetResult();

        /// <inheritdoc/>
        protected override Task AddCoreAsync(VectorDocument<T> vectorDocument, CancellationToken cancellationToken)
            => AddCoreImplAsync(vectorDocument, cancellationToken);

        private async Task AddCoreImplAsync(VectorDocument<T> vectorDocument, CancellationToken cancellationToken)
        {
            EnsureCollectionForDimension(vectorDocument.Embedding.Length);
            if (_vectorDimension == 0)
                _vectorDimension = vectorDocument.Embedding.Length;

            var body = new { collectionName = _collectionName, data = new[] { BuildEntity(vectorDocument) } };
            var info = await SendAsync("/v2/vectordb/entities/upsert", body, cancellationToken).ConfigureAwait(false);
            EnsureSuccess(info, "upsert entity");
            _documentCount++;
        }

        /// <inheritdoc/>
        protected override void AddBatchCore(IList<VectorDocument<T>> vectorDocuments)
            => AddBatchCoreImplAsync(vectorDocuments, CancellationToken.None).GetAwaiter().GetResult();

        /// <inheritdoc/>
        protected override Task AddBatchCoreAsync(IList<VectorDocument<T>> vectorDocuments, CancellationToken cancellationToken)
            => AddBatchCoreImplAsync(vectorDocuments, cancellationToken);

        private async Task AddBatchCoreImplAsync(IList<VectorDocument<T>> vectorDocuments, CancellationToken cancellationToken)
        {
            if (vectorDocuments.Count == 0)
                return;

            EnsureCollectionForDimension(vectorDocuments[0].Embedding.Length);
            if (_vectorDimension == 0)
                _vectorDimension = vectorDocuments[0].Embedding.Length;

            var data = vectorDocuments.Select(BuildEntity).ToList();
            var body = new { collectionName = _collectionName, data };
            var info = await SendAsync("/v2/vectordb/entities/upsert", body, cancellationToken).ConfigureAwait(false);
            EnsureSuccess(info, "batch upsert entities");
            _documentCount += vectorDocuments.Count;
        }

        private Dictionary<string, object?> BuildEntity(VectorDocument<T> vectorDocument)
        {
            var vector = vectorDocument.Embedding.ToArray().Select(v => Convert.ToDouble(v)).ToArray();
            var metadata = vectorDocument.Document.Metadata ?? new Dictionary<string, object>();

            var entity = new Dictionary<string, object?>
            {
                [IdField] = vectorDocument.Document.Id,
                [VectorField] = vector,
                [ContentField] = vectorDocument.Document.Content,
                [MetadataField] = JsonConvert.SerializeObject(metadata)
            };

            foreach (var kvp in metadata)
                entity[MetaPrefix + kvp.Key] = kvp.Value;

            return entity;
        }

        /// <inheritdoc/>
        protected override IEnumerable<Document<T>> GetSimilarCore(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters)
            => GetSimilarCoreImplAsync(queryVector, topK, metadataFilters, CancellationToken.None).GetAwaiter().GetResult();

        /// <inheritdoc/>
        protected override Task<IEnumerable<Document<T>>> GetSimilarCoreAsync(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters, CancellationToken cancellationToken)
            => GetSimilarCoreImplAsync(queryVector, topK, metadataFilters, cancellationToken);

        private Task<IEnumerable<Document<T>>> GetSimilarCoreImplAsync(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters, CancellationToken cancellationToken)
            => SearchImplAsync(queryVector, topK, BuildFilterExpression(metadataFilters), cancellationToken);

        /// <inheritdoc/>
        protected override IEnumerable<Document<T>> GetSimilarWithFilterCore(Vector<T> queryVector, MetadataFilter filter, int topK)
            => SearchImplAsync(queryVector, topK, TranslateFilter(filter), CancellationToken.None).GetAwaiter().GetResult();

        /// <inheritdoc/>
        protected override Task<IEnumerable<Document<T>>> GetSimilarWithFilterCoreAsync(Vector<T> queryVector, MetadataFilter filter, int topK, CancellationToken cancellationToken)
            => SearchImplAsync(queryVector, topK, TranslateFilter(filter), cancellationToken);

        private async Task<IEnumerable<Document<T>>> SearchImplAsync(Vector<T> queryVector, int topK, string? filter, CancellationToken cancellationToken)
        {
            var vector = queryVector.ToArray().Select(v => Convert.ToDouble(v)).ToArray();

            var body = new Dictionary<string, object>
            {
                ["collectionName"] = _collectionName,
                ["data"] = new[] { vector },
                ["annsField"] = VectorField,
                ["limit"] = topK,
                ["outputFields"] = new[] { IdField, ContentField, MetadataField }
            };

            if (filter != null)
                body["filter"] = filter;

            var info = await SendAsync("/v2/vectordb/entities/search", body, cancellationToken).ConfigureAwait(false);
            EnsureSuccess(info, "search");

            var results = new List<Document<T>>();
            var hits = JObject.Parse(info.Body)["data"] as JArray;
            if (hits == null)
                return results;

            foreach (var hit in hits)
            {
                var doc = ParseEntity(hit as JObject);
                if (doc == null)
                    continue;

                var distance = hit?["distance"];
                var score = distance != null && distance.Type != JTokenType.Null
                    ? Convert.ToDouble(distance, CultureInfo.InvariantCulture)
                    : 0.0;
                doc.RelevanceScore = NumOps.FromDouble(score);
                doc.HasRelevanceScore = true;
                results.Add(doc);
            }

            return results;
        }

        /// <summary>
        /// Translates the metadata filter dictionary into a Milvus boolean filter expression.
        /// </summary>
        /// <remarks>
        /// Equality (string/bool) uses <c>==</c>, numeric values use <c>&gt;=</c> (mirroring the
        /// base-class "field &gt;= value" semantics) and collection values become an <c>in [...]</c>
        /// clause (any-of). All conditions are combined with <c>and</c>. Filterable fields are stored
        /// flattened under the <c>m_</c> prefix.
        /// </remarks>
        private static string? BuildFilterExpression(Dictionary<string, object>? metadataFilters)
        {
            if (metadataFilters == null || metadataFilters.Count == 0)
                return null;

            var clauses = new List<string>();

            foreach (var kvp in metadataFilters)
            {
                var field = MetaPrefix + kvp.Key;
                var value = kvp.Value;

                if (value == null)
                {
                    continue;
                }
                else if (value is string s)
                {
                    clauses.Add($"{field} == {Quote(s)}");
                }
                else if (value is bool b)
                {
                    clauses.Add($"{field} == {(b ? "true" : "false")}");
                }
                else if (IsNumeric(value))
                {
                    clauses.Add($"{field} >= {Convert.ToDouble(value, CultureInfo.InvariantCulture).ToString("R", CultureInfo.InvariantCulture)}");
                }
                else if (value is System.Collections.IEnumerable enumerable)
                {
                    var items = new List<string>();
                    foreach (var item in enumerable)
                    {
                        if (item is string || item == null)
                            items.Add(Quote(item?.ToString() ?? string.Empty));
                        else if (IsNumeric(item))
                            items.Add(Convert.ToDouble(item, CultureInfo.InvariantCulture).ToString("R", CultureInfo.InvariantCulture));
                        else
                            items.Add(Quote(item.ToString() ?? string.Empty));
                    }
                    if (items.Count > 0)
                        clauses.Add($"{field} in [{string.Join(", ", items)}]");
                }
                else
                {
                    clauses.Add($"{field} == {Quote(value.ToString() ?? string.Empty)}");
                }
            }

            return clauses.Count > 0 ? string.Join(" and ", clauses) : null;
        }

        private static string Quote(string value) => "\"" + value.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";

        // ------------------------------------------------------------------
        // MetadataFilter AST -> Milvus boolean filter expression translation.
        // Filterable fields are stored flattened under the "m_" prefix.
        // ------------------------------------------------------------------

        /// <summary>Translates a <see cref="MetadataFilter"/> expression tree into a Milvus boolean expression.</summary>
        internal static string TranslateFilter(MetadataFilter filter)
        {
            if (filter == null)
                throw new ArgumentNullException(nameof(filter));
            return BuildExpression(filter);
        }

        private static string BuildExpression(MetadataFilter filter)
        {
            switch (filter)
            {
                case ComparisonFilter comparison:
                    return MetaPrefix + comparison.Key + " " + MilvusOperator(comparison.Operator) + " " + FormatValue(comparison.Value);
                case InFilter inFilter:
                    return MetaPrefix + inFilter.Key + " in [" + string.Join(", ", inFilter.Values.Select(FormatValue)) + "]";
                case ExistsFilter existsFilter:
                    return "exists " + MetaPrefix + existsFilter.Key;
                case NotFilter notFilter:
                    return "not (" + BuildExpression(notFilter.Operand) + ")";
                case LogicalFilter logical when logical.Operator == MetadataFilterOperator.And:
                    return "(" + string.Join(" and ", logical.Operands.Select(BuildExpression)) + ")";
                case LogicalFilter logical:
                    return "(" + string.Join(" or ", logical.Operands.Select(BuildExpression)) + ")";
                default:
                    throw new NotSupportedException($"Unsupported metadata filter node: {filter.GetType().Name}");
            }
        }

        private static string FormatValue(object value)
        {
            if (value == null)
                return "null";
            if (value is bool b)
                return b ? "true" : "false";
            if (MetadataFilter.IsNumeric(value))
                return Convert.ToDouble(value, CultureInfo.InvariantCulture).ToString("R", CultureInfo.InvariantCulture);
            return Quote(value.ToString() ?? string.Empty);
        }

        private static string MilvusOperator(MetadataFilterOperator op)
        {
            switch (op)
            {
                case MetadataFilterOperator.Eq: return "==";
                case MetadataFilterOperator.Ne: return "!=";
                case MetadataFilterOperator.Gt: return ">";
                case MetadataFilterOperator.Gte: return ">=";
                case MetadataFilterOperator.Lt: return "<";
                case MetadataFilterOperator.Lte: return "<=";
                default: throw new NotSupportedException($"Unsupported comparison operator: {op}");
            }
        }

        /// <inheritdoc/>
        protected override Document<T>? GetByIdCore(string documentId)
            => GetByIdCoreImplAsync(documentId, CancellationToken.None).GetAwaiter().GetResult();

        /// <inheritdoc/>
        protected override Task<Document<T>?> GetByIdCoreAsync(string documentId, CancellationToken cancellationToken)
            => GetByIdCoreImplAsync(documentId, cancellationToken);

        private async Task<Document<T>?> GetByIdCoreImplAsync(string documentId, CancellationToken cancellationToken)
        {
            var body = new
            {
                collectionName = _collectionName,
                filter = $"{IdField} == {Quote(documentId)}",
                outputFields = new[] { IdField, ContentField, MetadataField },
                limit = 1
            };

            var info = await SendAsync("/v2/vectordb/entities/query", body, cancellationToken).ConfigureAwait(false);
            EnsureSuccess(info, "query by id");

            var rows = JObject.Parse(info.Body)["data"] as JArray;
            if (rows == null || rows.Count == 0)
                return null;

            return ParseEntity(rows[0] as JObject);
        }

        /// <inheritdoc/>
        protected override bool RemoveCore(string documentId)
            => RemoveCoreImplAsync(documentId, CancellationToken.None).GetAwaiter().GetResult();

        /// <inheritdoc/>
        protected override Task<bool> RemoveCoreAsync(string documentId, CancellationToken cancellationToken)
            => RemoveCoreImplAsync(documentId, cancellationToken);

        private async Task<bool> RemoveCoreImplAsync(string documentId, CancellationToken cancellationToken)
        {
            var body = new
            {
                collectionName = _collectionName,
                filter = $"{IdField} == {Quote(documentId)}"
            };

            var info = await SendAsync("/v2/vectordb/entities/delete", body, cancellationToken).ConfigureAwait(false);
            if (!IsSuccess(info.Status))
                return false;

            var code = JObject.Parse(info.Body)["code"]?.Value<int>() ?? 0;
            if (code != 0)
                return false;

            if (_documentCount > 0)
                _documentCount--;
            return true;
        }

        /// <inheritdoc/>
        protected override IEnumerable<Document<T>> GetAllCore()
            => GetAllCoreImplAsync(CancellationToken.None).GetAwaiter().GetResult();

        /// <inheritdoc/>
        protected override Task<IEnumerable<Document<T>>> GetAllCoreAsync(CancellationToken cancellationToken)
            => GetAllCoreImplAsync(cancellationToken);

        private async Task<IEnumerable<Document<T>>> GetAllCoreImplAsync(CancellationToken cancellationToken)
        {
            var all = new List<Document<T>>();
            const int pageSize = 1000;
            var offset = 0;

            while (true)
            {
                var body = new Dictionary<string, object>
                {
                    ["collectionName"] = _collectionName,
                    ["filter"] = $"{IdField} != \"\"",
                    ["outputFields"] = new[] { IdField, ContentField, MetadataField },
                    ["limit"] = pageSize,
                    ["offset"] = offset
                };

                var info = await SendAsync("/v2/vectordb/entities/query", body, cancellationToken).ConfigureAwait(false);
                EnsureSuccess(info, "query all");

                var rows = JObject.Parse(info.Body)["data"] as JArray;
                if (rows == null || rows.Count == 0)
                    break;

                foreach (var row in rows)
                {
                    var doc = ParseEntity(row as JObject);
                    if (doc != null)
                        all.Add(doc);
                }

                if (rows.Count < pageSize)
                    break;
                offset += pageSize;
            }

            return all;
        }

        /// <inheritdoc/>
        public override void Clear()
            => ClearImplAsync(CancellationToken.None).GetAwaiter().GetResult();

        /// <inheritdoc/>
        public override Task ClearAsync(CancellationToken cancellationToken = default)
            => ClearImplAsync(cancellationToken);

        private async Task ClearImplAsync(CancellationToken cancellationToken)
        {
            var info = await SendAsync("/v2/vectordb/collections/drop",
                new { collectionName = _collectionName }, cancellationToken).ConfigureAwait(false);
            EnsureSuccess(info, "drop collection");

            _documentCount = 0;
            _collectionReady = false;

            if (_vectorDimension > 0)
                CreateCollection(_vectorDimension);
        }

        private Document<T>? ParseEntity(JObject? entity)
        {
            if (entity == null)
                return null;

            var id = entity[IdField]?.ToString();
            if (string.IsNullOrEmpty(id))
                return null;

            var content = entity[ContentField]?.ToString() ?? string.Empty;

            var metadata = new Dictionary<string, object>();
            var metaJson = entity[MetadataField]?.ToString();
            if (!string.IsNullOrEmpty(metaJson))
            {
                var parsed = JsonConvert.DeserializeObject<Dictionary<string, object>>(metaJson!);
                if (parsed != null)
                    metadata = parsed;
            }

            return new Document<T>(id!, content, metadata);
        }

        private static bool IsNumeric(object value)
        {
            return value is sbyte || value is byte || value is short || value is ushort
                || value is int || value is uint || value is long || value is ulong
                || value is float || value is double || value is decimal;
        }

        private static bool IsSuccess(HttpStatusCode status) => (int)status >= 200 && (int)status < 300;

        private void EnsureSuccess(HttpResponseInfo info, string operation)
        {
            if (!IsSuccess(info.Status))
                throw new HttpRequestException($"Milvus {operation} failed with status {(int)info.Status}: {info.Body}");

            var code = JObject.Parse(info.Body)["code"]?.Value<int>() ?? 0;
            if (code != 0)
                throw new HttpRequestException($"Milvus {operation} failed with code {code}: {info.Body}");
        }

        private async Task<HttpResponseInfo> SendAsync(string path, object body, CancellationToken cancellationToken = default)
        {
            using (var request = new HttpRequestMessage(HttpMethod.Post, path))
            {
                var json = JsonConvert.SerializeObject(body);
                request.Content = new StringContent(json, Encoding.UTF8, "application/json");

                using (var response = await _httpClient.SendAsync(request, cancellationToken).ConfigureAwait(false))
                {
                    var content = response.Content != null
                        ? await response.Content.ReadAsStringAsync().ConfigureAwait(false)
                        : string.Empty;
                    return new HttpResponseInfo(response.StatusCode, content);
                }
            }
        }

        private readonly struct HttpResponseInfo
        {
            public HttpResponseInfo(HttpStatusCode status, string body)
            {
                Status = status;
                Body = body;
            }

            public HttpStatusCode Status { get; }
            public string Body { get; }
        }
    }
}
