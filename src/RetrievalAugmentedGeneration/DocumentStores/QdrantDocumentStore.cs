using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Security.Cryptography;
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
    /// Qdrant vector-database document store backed by the real Qdrant REST API.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    /// <remarks>
    /// <para>
    /// This store talks to a running Qdrant instance over HTTP. It manages a single collection,
    /// upserts points (vector + payload), performs filtered vector search, deletes points and
    /// scrolls the whole collection. Document metadata is stored under a nested <c>metadata</c>
    /// payload object, while the original string document id and content are stored under the
    /// reserved payload keys <c>_doc_id</c> and <c>_content</c>.
    /// </para>
    /// <para><b>For Beginners:</b> Qdrant is an open-source vector database. This class is a real
    /// client for it - every method here makes an HTTP call to your Qdrant server. Qdrant requires
    /// point ids to be unsigned integers or UUIDs, so each document's string id is turned into a
    /// deterministic UUID; the original string id is kept in the payload so you always get it back.
    /// </para>
    /// </remarks>
    [ComponentType(ComponentType.DocumentStore)]
    [PipelineStage(PipelineStage.Indexing)]
    public class QdrantDocumentStore<T> : DocumentStoreBase<T>
    {
        private const string DocIdKey = "_doc_id";
        private const string ContentKey = "_content";

        private readonly HttpClient _httpClient;
        private readonly string _collectionName;
        private readonly string _distance;
        private int _vectorDimension;
        private int _documentCount;
        private bool _collectionReady;

        /// <summary>
        /// Gets the number of documents (points) currently stored in the collection.
        /// </summary>
        public override int DocumentCount => _documentCount;

        /// <summary>
        /// Gets the dimensionality of vectors stored in this collection.
        /// </summary>
        public override int VectorDimension => _vectorDimension;

        /// <summary>
        /// Gets the name of the Qdrant collection this store is bound to.
        /// </summary>
        public string CollectionName { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="QdrantDocumentStore{T}"/> class.
        /// </summary>
        /// <param name="collectionName">The Qdrant collection name.</param>
        /// <param name="url">The base URL of the Qdrant server, e.g. <c>http://localhost:6333</c>.</param>
        /// <param name="apiKey">Optional Qdrant API key (sent as the <c>api-key</c> header).</param>
        /// <param name="distanceMetric">The distance metric used when creating the collection.</param>
        /// <param name="vectorDimension">
        /// The vector dimension used to create the collection if it does not already exist.
        /// When 0 the collection must already exist (its dimension is read from the server) or the
        /// dimension is inferred from the first document added.
        /// </param>
        /// <param name="httpClient">
        /// Optional pre-configured <see cref="HttpClient"/>. Primarily for testing; when supplied its
        /// <see cref="HttpClient.BaseAddress"/> is used if already set.
        /// </param>
        /// <param name="handler">Optional <see cref="HttpMessageHandler"/> used to build the client (for testing).</param>
        public QdrantDocumentStore(
            string collectionName,
            string url,
            string? apiKey = null,
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
            _distance = MapDistance(distanceMetric);
            _vectorDimension = vectorDimension;
            _documentCount = 0;

            _httpClient = httpClient ?? (handler != null ? new HttpClient(handler) : new HttpClient());
            if (_httpClient.BaseAddress == null)
            {
                if (string.IsNullOrWhiteSpace(url))
                    throw new ArgumentException("Url cannot be empty", nameof(url));
                _httpClient.BaseAddress = new Uri(url);
            }

            if (!string.IsNullOrWhiteSpace(apiKey) && !_httpClient.DefaultRequestHeaders.Contains("api-key"))
                _httpClient.DefaultRequestHeaders.Add("api-key", apiKey);

            InitializeCollection(vectorDimension);
        }

        private static string MapDistance(DistanceMetricType metric)
        {
            switch (metric)
            {
                case DistanceMetricType.Cosine:
                    return "Cosine";
                case DistanceMetricType.Euclidean:
                    return "Euclid";
                case DistanceMetricType.Manhattan:
                    return "Manhattan";
                default:
                    throw new NotSupportedException(
                        $"Distance metric '{metric}' is not supported by Qdrant. Use Cosine, Euclidean or Manhattan.");
            }
        }

        private void InitializeCollection(int requestedDimension)
        {
            HttpResponseInfo info;
            try
            {
                info = SendAsync(HttpMethod.Get, $"/collections/{_collectionName}", null).GetAwaiter().GetResult();
            }
            catch (HttpRequestException)
            {
                // Server not reachable at construction time; defer creation to first write.
                return;
            }

            if (info.Status == HttpStatusCode.OK)
            {
                var root = JObject.Parse(info.Body)["result"];
                var size = root?["config"]?["params"]?["vectors"]?["size"]?.Value<int>();
                if (size.HasValue && size.Value > 0)
                    _vectorDimension = size.Value;

                var count = root?["points_count"]?.Value<int>();
                if (count.HasValue)
                    _documentCount = count.Value;

                _collectionReady = true;
                return;
            }

            if (info.Status == HttpStatusCode.NotFound && requestedDimension > 0)
                CreateCollection(requestedDimension);
        }

        private void CreateCollection(int dimension)
        {
            var body = new
            {
                vectors = new
                {
                    size = dimension,
                    distance = _distance
                }
            };

            var info = SendAsync(HttpMethod.Put, $"/collections/{_collectionName}", body).GetAwaiter().GetResult();
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

            var body = new { points = new[] { BuildPoint(vectorDocument) } };
            var info = await SendAsync(HttpMethod.Put, $"/collections/{_collectionName}/points?wait=true", body, cancellationToken).ConfigureAwait(false);
            EnsureSuccess(info, "upsert point");
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

            var points = vectorDocuments.Select(BuildPoint).ToList();
            var body = new { points };
            var info = await SendAsync(HttpMethod.Put, $"/collections/{_collectionName}/points?wait=true", body, cancellationToken).ConfigureAwait(false);
            EnsureSuccess(info, "batch upsert points");
            _documentCount += vectorDocuments.Count;
        }

        private object BuildPoint(VectorDocument<T> vectorDocument)
        {
            var vector = vectorDocument.Embedding.ToArray().Select(v => Convert.ToDouble(v)).ToArray();
            var payload = new Dictionary<string, object>
            {
                [DocIdKey] = vectorDocument.Document.Id,
                [ContentKey] = vectorDocument.Document.Content,
                ["metadata"] = vectorDocument.Document.Metadata ?? new Dictionary<string, object>()
            };

            return new
            {
                id = ToPointId(vectorDocument.Document.Id),
                vector,
                payload
            };
        }

        /// <inheritdoc/>
        protected override IEnumerable<Document<T>> GetSimilarCore(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters)
            => GetSimilarCoreImplAsync(queryVector, topK, metadataFilters, CancellationToken.None).GetAwaiter().GetResult();

        /// <inheritdoc/>
        protected override Task<IEnumerable<Document<T>>> GetSimilarCoreAsync(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters, CancellationToken cancellationToken)
            => GetSimilarCoreImplAsync(queryVector, topK, metadataFilters, cancellationToken);

        private Task<IEnumerable<Document<T>>> GetSimilarCoreImplAsync(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters, CancellationToken cancellationToken)
            => SearchImplAsync(queryVector, topK, BuildFilter(metadataFilters), cancellationToken);

        /// <inheritdoc/>
        protected override IEnumerable<Document<T>> GetSimilarWithFilterCore(Vector<T> queryVector, MetadataFilter filter, int topK)
            => SearchImplAsync(queryVector, topK, TranslateFilter(filter), CancellationToken.None).GetAwaiter().GetResult();

        /// <inheritdoc/>
        protected override Task<IEnumerable<Document<T>>> GetSimilarWithFilterCoreAsync(Vector<T> queryVector, MetadataFilter filter, int topK, CancellationToken cancellationToken)
            => SearchImplAsync(queryVector, topK, TranslateFilter(filter), cancellationToken);

        private async Task<IEnumerable<Document<T>>> SearchImplAsync(Vector<T> queryVector, int topK, object? filter, CancellationToken cancellationToken)
        {
            var vector = queryVector.ToArray().Select(v => Convert.ToDouble(v)).ToArray();

            var body = new Dictionary<string, object>
            {
                ["vector"] = vector,
                ["limit"] = topK,
                ["with_payload"] = true,
                ["with_vector"] = false
            };

            if (filter != null)
                body["filter"] = filter;

            var info = await SendAsync(HttpMethod.Post, $"/collections/{_collectionName}/points/search", body, cancellationToken).ConfigureAwait(false);
            EnsureSuccess(info, "search");

            var results = new List<Document<T>>();
            var hits = JObject.Parse(info.Body)["result"] as JArray;
            if (hits == null)
                return results;

            foreach (var hit in hits)
            {
                var doc = ParsePayload(hit?["payload"] as JObject);
                if (doc == null)
                    continue;

                var score = hit?["score"] != null ? Convert.ToDouble(hit!["score"], CultureInfo.InvariantCulture) : 0.0;
                doc.RelevanceScore = NumOps.FromDouble(score);
                doc.HasRelevanceScore = true;
                results.Add(doc);
            }

            return results;
        }

        // ------------------------------------------------------------------
        // MetadataFilter AST -> Qdrant filter translation.
        // Leaf conditions map to field conditions (match / range); logical
        // nodes map to nested bool filters (must / should / must_not).
        // ------------------------------------------------------------------

        /// <summary>
        /// Translates a <see cref="MetadataFilter"/> expression tree into a Qdrant filter object.
        /// </summary>
        internal static object TranslateFilter(MetadataFilter filter)
        {
            if (filter == null)
                throw new ArgumentNullException(nameof(filter));

            // A bare leaf condition must be wrapped in a "must" so the top-level value is a valid filter.
            if (filter is ComparisonFilter c && c.Operator == MetadataFilterOperator.Eq
                || filter is ComparisonFilter cg && (cg.Operator == MetadataFilterOperator.Gt || cg.Operator == MetadataFilterOperator.Gte
                    || cg.Operator == MetadataFilterOperator.Lt || cg.Operator == MetadataFilterOperator.Lte)
                || filter is InFilter)
            {
                return new Dictionary<string, object> { ["must"] = new[] { TranslateClause(filter) } };
            }

            return TranslateClause(filter);
        }

        private static object TranslateClause(MetadataFilter filter)
        {
            switch (filter)
            {
                case ComparisonFilter comparison:
                    return TranslateComparison(comparison);
                case InFilter inFilter:
                    return new Dictionary<string, object>
                    {
                        ["key"] = FieldKey(inFilter.Key),
                        ["match"] = new Dictionary<string, object> { ["any"] = inFilter.Values.ToArray() }
                    };
                case ExistsFilter existsFilter:
                    return new Dictionary<string, object>
                    {
                        ["must_not"] = new object[]
                        {
                            new Dictionary<string, object>
                            {
                                ["is_empty"] = new Dictionary<string, object> { ["key"] = FieldKey(existsFilter.Key) }
                            }
                        }
                    };
                case NotFilter notFilter:
                    return new Dictionary<string, object>
                    {
                        ["must_not"] = new[] { TranslateClause(notFilter.Operand) }
                    };
                case LogicalFilter logical when logical.Operator == MetadataFilterOperator.And:
                    return new Dictionary<string, object>
                    {
                        ["must"] = logical.Operands.Select(TranslateClause).ToArray()
                    };
                case LogicalFilter logical:
                    return new Dictionary<string, object>
                    {
                        ["should"] = logical.Operands.Select(TranslateClause).ToArray()
                    };
                default:
                    throw new NotSupportedException($"Unsupported metadata filter node: {filter.GetType().Name}");
            }
        }

        private static object TranslateComparison(ComparisonFilter comparison)
        {
            var key = FieldKey(comparison.Key);
            switch (comparison.Operator)
            {
                case MetadataFilterOperator.Eq:
                    return new Dictionary<string, object>
                    {
                        ["key"] = key,
                        ["match"] = new Dictionary<string, object> { ["value"] = comparison.Value }
                    };
                case MetadataFilterOperator.Ne:
                    return new Dictionary<string, object>
                    {
                        ["must_not"] = new object[]
                        {
                            new Dictionary<string, object>
                            {
                                ["key"] = key,
                                ["match"] = new Dictionary<string, object> { ["value"] = comparison.Value }
                            }
                        }
                    };
                case MetadataFilterOperator.Gt:
                    return RangeCondition(key, "gt", comparison.Value);
                case MetadataFilterOperator.Gte:
                    return RangeCondition(key, "gte", comparison.Value);
                case MetadataFilterOperator.Lt:
                    return RangeCondition(key, "lt", comparison.Value);
                case MetadataFilterOperator.Lte:
                    return RangeCondition(key, "lte", comparison.Value);
                default:
                    throw new NotSupportedException($"Unsupported comparison operator: {comparison.Operator}");
            }
        }

        private static object RangeCondition(string key, string op, object value)
        {
            object bound = MetadataFilter.IsNumeric(value)
                ? Convert.ToDouble(value, CultureInfo.InvariantCulture)
                : value;
            return new Dictionary<string, object>
            {
                ["key"] = key,
                ["range"] = new Dictionary<string, object> { [op] = bound }
            };
        }

        private static string FieldKey(string key) => "metadata." + key;

        /// <summary>
        /// Translates the metadata filter dictionary into a Qdrant filter object.
        /// </summary>
        /// <remarks>
        /// Equality (string/bool) becomes a <c>match</c> condition and numeric values become a
        /// <c>range</c> with <c>gte</c> (mirroring the base-class "field &gt;= value" semantics).
        /// All scalar conditions are combined under <c>must</c>. Collection values become <c>should</c> (any-of).
        /// </remarks>
        private static object? BuildFilter(Dictionary<string, object>? metadataFilters)
        {
            if (metadataFilters == null || metadataFilters.Count == 0)
                return null;

            var must = new List<object>();
            var should = new List<object>();

            foreach (var kvp in metadataFilters)
            {
                var key = "metadata." + kvp.Key;
                var value = kvp.Value;

                if (value == null || value is string || value is bool)
                {
                    must.Add(new { key, match = new { value } });
                }
                else if (IsNumeric(value))
                {
                    must.Add(new { key, range = new { gte = Convert.ToDouble(value, CultureInfo.InvariantCulture) } });
                }
                else if (value is System.Collections.IEnumerable enumerable)
                {
                    foreach (var item in enumerable)
                        should.Add(new { key, match = new { value = item } });
                }
                else
                {
                    must.Add(new { key, match = new { value = value.ToString() } });
                }
            }

            var filter = new Dictionary<string, object>();
            if (must.Count > 0)
                filter["must"] = must;
            if (should.Count > 0)
                filter["should"] = should;

            return filter.Count > 0 ? filter : null;
        }

        /// <inheritdoc/>
        protected override Document<T>? GetByIdCore(string documentId)
            => GetByIdCoreImplAsync(documentId, CancellationToken.None).GetAwaiter().GetResult();

        /// <inheritdoc/>
        protected override Task<Document<T>?> GetByIdCoreAsync(string documentId, CancellationToken cancellationToken)
            => GetByIdCoreImplAsync(documentId, cancellationToken);

        private async Task<Document<T>?> GetByIdCoreImplAsync(string documentId, CancellationToken cancellationToken)
        {
            var info = await SendAsync(HttpMethod.Get, $"/collections/{_collectionName}/points/{ToPointId(documentId)}", null, cancellationToken).ConfigureAwait(false);
            if (info.Status == HttpStatusCode.NotFound)
                return null;
            EnsureSuccess(info, "get point");

            var result = JObject.Parse(info.Body)["result"] as JObject;
            if (result == null)
                return null;

            return ParsePayload(result["payload"] as JObject);
        }

        /// <inheritdoc/>
        protected override bool RemoveCore(string documentId)
            => RemoveCoreImplAsync(documentId, CancellationToken.None).GetAwaiter().GetResult();

        /// <inheritdoc/>
        protected override Task<bool> RemoveCoreAsync(string documentId, CancellationToken cancellationToken)
            => RemoveCoreImplAsync(documentId, cancellationToken);

        private async Task<bool> RemoveCoreImplAsync(string documentId, CancellationToken cancellationToken)
        {
            var body = new { points = new[] { ToPointId(documentId) } };
            var info = await SendAsync(HttpMethod.Post, $"/collections/{_collectionName}/points/delete?wait=true", body, cancellationToken).ConfigureAwait(false);
            if (!IsSuccess(info.Status))
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
            object? offset = null;

            while (true)
            {
                var body = new Dictionary<string, object>
                {
                    ["limit"] = 256,
                    ["with_payload"] = true,
                    ["with_vector"] = false
                };
                if (offset != null)
                    body["offset"] = offset;

                var info = await SendAsync(HttpMethod.Post, $"/collections/{_collectionName}/points/scroll", body, cancellationToken).ConfigureAwait(false);
                EnsureSuccess(info, "scroll");

                var result = JObject.Parse(info.Body)["result"] as JObject;
                var points = result?["points"] as JArray;
                if (points == null || points.Count == 0)
                    break;

                foreach (var point in points)
                {
                    var doc = ParsePayload(point?["payload"] as JObject);
                    if (doc != null)
                        all.Add(doc);
                }

                var next = result?["next_page_offset"];
                if (next == null || next.Type == JTokenType.Null)
                    break;
                offset = next.ToObject<object>();
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
            var info = await SendAsync(HttpMethod.Delete, $"/collections/{_collectionName}", null, cancellationToken).ConfigureAwait(false);
            EnsureSuccess(info, "delete collection");

            _documentCount = 0;
            _collectionReady = false;

            if (_vectorDimension > 0)
                CreateCollection(_vectorDimension);
        }

        private Document<T>? ParsePayload(JObject? payload)
        {
            if (payload == null)
                return null;

            var id = payload[DocIdKey]?.ToString();
            if (string.IsNullOrEmpty(id))
                return null;

            var content = payload[ContentKey]?.ToString() ?? string.Empty;
            var metadata = (payload["metadata"] as JObject)?.ToObject<Dictionary<string, object>>()
                           ?? new Dictionary<string, object>();

            return new Document<T>(id!, content, metadata);
        }

        private static bool IsNumeric(object value)
        {
            return value is sbyte || value is byte || value is short || value is ushort
                || value is int || value is uint || value is long || value is ulong
                || value is float || value is double || value is decimal;
        }

        private static bool IsSuccess(HttpStatusCode status) => (int)status >= 200 && (int)status < 300;

        /// <summary>
        /// Produces a deterministic UUID string for an arbitrary document id, since Qdrant point
        /// ids must be unsigned integers or UUIDs. The mapping is stable so lookups/deletes work.
        /// </summary>
        private static string ToPointId(string documentId)
        {
            using (var md5 = MD5.Create())
            {
                var hash = md5.ComputeHash(Encoding.UTF8.GetBytes(documentId));
                return new Guid(hash).ToString();
            }
        }

        private void EnsureSuccess(HttpResponseInfo info, string operation)
        {
            if (!IsSuccess(info.Status))
                throw new HttpRequestException($"Qdrant {operation} failed with status {(int)info.Status}: {info.Body}");
        }

        private async Task<HttpResponseInfo> SendAsync(HttpMethod method, string path, object? body, CancellationToken cancellationToken = default)
        {
            using (var request = new HttpRequestMessage(method, path))
            {
                if (body != null)
                {
                    var json = JsonConvert.SerializeObject(body);
                    request.Content = new StringContent(json, Encoding.UTF8, "application/json");
                }

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
