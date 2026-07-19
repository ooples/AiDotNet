global using AiDotNet.RetrievalAugmentedGeneration.Models;

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
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores
{
    /// <summary>
    /// Weaviate vector-database document store backed by the real Weaviate REST/GraphQL API.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    /// <remarks>
    /// <para>
    /// This store talks to a running Weaviate instance over HTTP. It manages a single class,
    /// upserts objects (vector + properties), performs <c>nearVector</c> GraphQL search with a
    /// <c>where</c> filter, deletes objects and cursor-pages the whole class. The original document
    /// id and content are stored under the reserved properties <c>docId</c> and <c>content</c>; the
    /// full metadata dictionary is stored as a JSON string under <c>metadataJson</c> (for lossless
    /// round-tripping) while each scalar metadata entry is also flattened to a <c>m_&lt;key&gt;</c>
    /// property so it can be used in server-side <c>where</c> filters.
    /// </para>
    /// <para><b>For Beginners:</b> Weaviate is an open-source vector database with a GraphQL API.
    /// This class is a real client for it - every method here makes an HTTP call to your Weaviate
    /// server. Weaviate requires object ids to be UUIDs, so each document's string id is turned into
    /// a deterministic UUID; the original string id is kept in the <c>docId</c> property so you always
    /// get it back.
    /// </para>
    /// </remarks>
    [ComponentType(ComponentType.DocumentStore)]
    [PipelineStage(PipelineStage.Indexing)]
    public class WeaviateDocumentStore<T> : DocumentStoreBase<T>
    {
        private const string DocIdProp = "docId";
        private const string ContentProp = "content";
        private const string MetadataProp = "metadataJson";
        private const string MetaPrefix = "m_";

        private readonly HttpClient _httpClient;
        private readonly string _className;
        private readonly string _distance;
        private int _vectorDimension;
        private int _documentCount;
        private bool _classReady;

        /// <summary>
        /// Gets the number of documents (objects) currently stored in the class.
        /// </summary>
        public override int DocumentCount => _documentCount;

        /// <summary>
        /// Gets the dimensionality of vectors stored in this class.
        /// </summary>
        public override int VectorDimension => _vectorDimension;

        /// <summary>
        /// Gets the name of the Weaviate class this store is bound to.
        /// </summary>
        public string ClassName { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="WeaviateDocumentStore{T}"/> class.
        /// </summary>
        /// <param name="className">The Weaviate class name (must start with an uppercase letter).</param>
        /// <param name="url">The base URL of the Weaviate server, e.g. <c>http://localhost:8080</c>.</param>
        /// <param name="apiKey">Optional Weaviate API key (sent as an <c>Authorization: Bearer</c> header).</param>
        /// <param name="distanceMetric">The distance metric used when creating the class.</param>
        /// <param name="vectorDimension">Optional expected vector dimension (informational; inferred from the first document when 0).</param>
        /// <param name="httpClient">Optional pre-configured <see cref="HttpClient"/> (primarily for testing).</param>
        /// <param name="handler">Optional <see cref="HttpMessageHandler"/> used to build the client (for testing).</param>
        public WeaviateDocumentStore(
            string className,
            string url,
            string? apiKey = null,
            DistanceMetricType distanceMetric = DistanceMetricType.Cosine,
            int vectorDimension = 0,
            HttpClient? httpClient = null,
            HttpMessageHandler? handler = null)
        {
            if (string.IsNullOrWhiteSpace(className))
                throw new ArgumentException("Class name cannot be empty", nameof(className));
            if (vectorDimension < 0)
                throw new ArgumentOutOfRangeException(nameof(vectorDimension), "Vector dimension cannot be negative");

            // Weaviate GraphQL class names must be capitalized.
            _className = char.ToUpperInvariant(className[0]) + className.Substring(1);
            ClassName = _className;
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

            if (!string.IsNullOrWhiteSpace(apiKey) && _httpClient.DefaultRequestHeaders.Authorization == null)
                _httpClient.DefaultRequestHeaders.Add("Authorization", "Bearer " + apiKey);

            InitializeClass();
        }

        private static string MapDistance(DistanceMetricType metric)
        {
            switch (metric)
            {
                case DistanceMetricType.Cosine:
                    return "cosine";
                case DistanceMetricType.Euclidean:
                    return "l2-squared";
                case DistanceMetricType.Manhattan:
                    return "manhattan";
                case DistanceMetricType.Hamming:
                    return "hamming";
                default:
                    throw new NotSupportedException(
                        $"Distance metric '{metric}' is not supported by Weaviate. Use Cosine, Euclidean, Manhattan or Hamming.");
            }
        }

        private void InitializeClass()
        {
            HttpResponseInfo info;
            try
            {
                info = SendAsync(HttpMethod.Get, $"/v1/schema/{_className}", null).GetAwaiter().GetResult();
            }
            catch (HttpRequestException)
            {
                // Server not reachable at construction time; defer creation to first write.
                return;
            }

            if (info.Status == HttpStatusCode.OK)
            {
                _classReady = true;
                return;
            }

            if (info.Status == HttpStatusCode.NotFound)
                CreateClass();
        }

        private void CreateClass()
        {
            var body = new
            {
                @class = _className,
                vectorizer = "none",
                vectorIndexConfig = new { distance = _distance },
                properties = new object[]
                {
                    new { name = DocIdProp, dataType = new[] { "text" } },
                    new { name = ContentProp, dataType = new[] { "text" } },
                    new { name = MetadataProp, dataType = new[] { "text" } }
                }
            };

            var info = SendAsync(HttpMethod.Post, "/v1/schema", body).GetAwaiter().GetResult();
            EnsureSuccess(info, "create class");
            _classReady = true;
        }

        private void EnsureClass()
        {
            if (!_classReady)
                CreateClass();
        }

        /// <inheritdoc/>
        protected override void AddCore(VectorDocument<T> vectorDocument)
            => AddCoreImplAsync(vectorDocument, CancellationToken.None).GetAwaiter().GetResult();

        /// <inheritdoc/>
        protected override Task AddCoreAsync(VectorDocument<T> vectorDocument, CancellationToken cancellationToken)
            => AddCoreImplAsync(vectorDocument, cancellationToken);

        private async Task AddCoreImplAsync(VectorDocument<T> vectorDocument, CancellationToken cancellationToken)
        {
            EnsureClass();
            if (_vectorDimension == 0)
                _vectorDimension = vectorDocument.Embedding.Length;

            var body = BuildObject(vectorDocument);
            var info = await SendAsync(HttpMethod.Post, "/v1/objects", body, cancellationToken).ConfigureAwait(false);
            EnsureSuccess(info, "create object");
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

            EnsureClass();
            if (_vectorDimension == 0)
                _vectorDimension = vectorDocuments[0].Embedding.Length;

            var objects = vectorDocuments.Select(BuildObject).ToList();
            var body = new { objects };
            var info = await SendAsync(HttpMethod.Post, "/v1/batch/objects", body, cancellationToken).ConfigureAwait(false);
            EnsureSuccess(info, "batch create objects");
            _documentCount += vectorDocuments.Count;
        }

        private object BuildObject(VectorDocument<T> vectorDocument)
        {
            var vector = vectorDocument.Embedding.ToArray().Select(v => Convert.ToDouble(v)).ToArray();
            var metadata = vectorDocument.Document.Metadata ?? new Dictionary<string, object>();

            var properties = new Dictionary<string, object?>
            {
                [DocIdProp] = vectorDocument.Document.Id,
                [ContentProp] = vectorDocument.Document.Content,
                [MetadataProp] = JsonConvert.SerializeObject(metadata)
            };

            foreach (var kvp in metadata)
                properties[MetaPrefix + kvp.Key] = kvp.Value;

            return new
            {
                @class = _className,
                id = ToObjectId(vectorDocument.Document.Id),
                properties,
                vector
            };
        }

        /// <inheritdoc/>
        protected override IEnumerable<Document<T>> GetSimilarCore(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters)
            => GetSimilarCoreImplAsync(queryVector, topK, metadataFilters, CancellationToken.None).GetAwaiter().GetResult();

        /// <inheritdoc/>
        protected override Task<IEnumerable<Document<T>>> GetSimilarCoreAsync(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters, CancellationToken cancellationToken)
            => GetSimilarCoreImplAsync(queryVector, topK, metadataFilters, cancellationToken);

        private async Task<IEnumerable<Document<T>>> GetSimilarCoreImplAsync(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters, CancellationToken cancellationToken)
        {
            var vector = queryVector.ToArray().Select(v => Convert.ToDouble(v)).ToArray();
            var vectorJson = "[" + string.Join(",", vector.Select(v => v.ToString("R", CultureInfo.InvariantCulture))) + "]";

            var whereClause = BuildWhere(metadataFilters);
            var whereArg = whereClause != null ? ", where: " + whereClause : string.Empty;

            var query =
                "{ Get { " + _className + "(limit: " + topK.ToString(CultureInfo.InvariantCulture) +
                ", nearVector: {vector: " + vectorJson + "}" + whereArg + ") " +
                "{ " + DocIdProp + " " + ContentProp + " " + MetadataProp +
                " _additional { certainty distance } } } }";

            var info = await SendAsync(HttpMethod.Post, "/v1/graphql", new { query }, cancellationToken).ConfigureAwait(false);
            EnsureSuccess(info, "search");

            var results = new List<Document<T>>();
            var hits = JObject.Parse(info.Body)["data"]?["Get"]?[_className] as JArray;
            if (hits == null)
                return results;

            foreach (var hit in hits)
            {
                var doc = ParseProperties(hit as JObject);
                if (doc == null)
                    continue;

                var additional = hit?["_additional"];
                double score = 0.0;
                var certainty = additional?["certainty"];
                if (certainty != null && certainty.Type != JTokenType.Null)
                    score = Convert.ToDouble(certainty, CultureInfo.InvariantCulture);
                else
                {
                    var distance = additional?["distance"];
                    if (distance != null && distance.Type != JTokenType.Null)
                        score = 1.0 - Convert.ToDouble(distance, CultureInfo.InvariantCulture);
                }

                doc.RelevanceScore = NumOps.FromDouble(score);
                doc.HasRelevanceScore = true;
                results.Add(doc);
            }

            return results;
        }

        /// <summary>
        /// Translates the metadata filter dictionary into a Weaviate GraphQL <c>where</c> clause.
        /// </summary>
        /// <remarks>
        /// Equality (string/bool) uses <c>Equal</c>, numeric values use <c>GreaterThanEqual</c>
        /// (mirroring the base-class "field &gt;= value" semantics) and collection values become an
        /// <c>Or</c> group of <c>Equal</c> operands (any-of). All top-level conditions are combined
        /// under <c>And</c>. Filterable properties are stored flattened under the <c>m_</c> prefix.
        /// </remarks>
        private static string? BuildWhere(Dictionary<string, object>? metadataFilters)
        {
            if (metadataFilters == null || metadataFilters.Count == 0)
                return null;

            var operands = new List<string>();

            foreach (var kvp in metadataFilters)
            {
                var path = "[\"" + MetaPrefix + kvp.Key + "\"]";
                var value = kvp.Value;

                if (value == null)
                {
                    operands.Add("{path: " + path + ", operator: IsNull, valueBoolean: true}");
                }
                else if (value is string s)
                {
                    operands.Add("{path: " + path + ", operator: Equal, valueText: " + JsonConvert.ToString(s) + "}");
                }
                else if (value is bool b)
                {
                    operands.Add("{path: " + path + ", operator: Equal, valueBoolean: " + (b ? "true" : "false") + "}");
                }
                else if (IsNumeric(value))
                {
                    var num = Convert.ToDouble(value, CultureInfo.InvariantCulture).ToString("R", CultureInfo.InvariantCulture);
                    operands.Add("{path: " + path + ", operator: GreaterThanEqual, valueNumber: " + num + "}");
                }
                else if (value is System.Collections.IEnumerable enumerable)
                {
                    var anyOf = new List<string>();
                    foreach (var item in enumerable)
                    {
                        anyOf.Add("{path: " + path + ", operator: Equal, valueText: " +
                                  JsonConvert.ToString(item?.ToString() ?? string.Empty) + "}");
                    }
                    if (anyOf.Count > 0)
                        operands.Add("{operator: Or, operands: [" + string.Join(", ", anyOf) + "]}");
                }
                else
                {
                    operands.Add("{path: " + path + ", operator: Equal, valueText: " +
                                 JsonConvert.ToString(value.ToString() ?? string.Empty) + "}");
                }
            }

            if (operands.Count == 0)
                return null;
            if (operands.Count == 1)
                return operands[0];

            return "{operator: And, operands: [" + string.Join(", ", operands) + "]}";
        }

        /// <inheritdoc/>
        protected override Document<T>? GetByIdCore(string documentId)
            => GetByIdCoreImplAsync(documentId, CancellationToken.None).GetAwaiter().GetResult();

        /// <inheritdoc/>
        protected override Task<Document<T>?> GetByIdCoreAsync(string documentId, CancellationToken cancellationToken)
            => GetByIdCoreImplAsync(documentId, cancellationToken);

        private async Task<Document<T>?> GetByIdCoreImplAsync(string documentId, CancellationToken cancellationToken)
        {
            var info = await SendAsync(HttpMethod.Get, $"/v1/objects/{_className}/{ToObjectId(documentId)}", null, cancellationToken).ConfigureAwait(false);
            if (info.Status == HttpStatusCode.NotFound)
                return null;
            EnsureSuccess(info, "get object");

            return ParseObject(JObject.Parse(info.Body));
        }

        /// <inheritdoc/>
        protected override bool RemoveCore(string documentId)
            => RemoveCoreImplAsync(documentId, CancellationToken.None).GetAwaiter().GetResult();

        /// <inheritdoc/>
        protected override Task<bool> RemoveCoreAsync(string documentId, CancellationToken cancellationToken)
            => RemoveCoreImplAsync(documentId, cancellationToken);

        private async Task<bool> RemoveCoreImplAsync(string documentId, CancellationToken cancellationToken)
        {
            var info = await SendAsync(HttpMethod.Delete, $"/v1/objects/{_className}/{ToObjectId(documentId)}", null, cancellationToken).ConfigureAwait(false);
            if (info.Status == HttpStatusCode.NotFound)
                return false;
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
            string? after = null;
            const int pageSize = 100;

            while (true)
            {
                var path = $"/v1/objects?class={_className}&limit={pageSize}&include=vector";
                if (after != null)
                    path += "&after=" + after;

                var info = await SendAsync(HttpMethod.Get, path, null, cancellationToken).ConfigureAwait(false);
                EnsureSuccess(info, "list objects");

                var objects = JObject.Parse(info.Body)["objects"] as JArray;
                if (objects == null || objects.Count == 0)
                    break;

                foreach (var obj in objects)
                {
                    var doc = ParseObject(obj as JObject);
                    if (doc != null)
                        all.Add(doc);
                }

                after = objects[objects.Count - 1]?["id"]?.ToString();
                if (string.IsNullOrEmpty(after) || objects.Count < pageSize)
                    break;
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
            var info = await SendAsync(HttpMethod.Delete, $"/v1/schema/{_className}", null, cancellationToken).ConfigureAwait(false);
            if (!IsSuccess(info.Status) && info.Status != HttpStatusCode.NotFound)
                EnsureSuccess(info, "delete class");

            _documentCount = 0;
            _classReady = false;
            CreateClass();
        }

        // REST responses (GetById / list) nest fields under "properties".
        private Document<T>? ParseObject(JObject? obj)
        {
            return ParseProperties(obj?["properties"] as JObject);
        }

        // GraphQL search hits expose the fields flat on the hit object.
        private Document<T>? ParseProperties(JObject? properties)
        {
            if (properties == null)
                return null;

            var id = properties[DocIdProp]?.ToString();
            if (string.IsNullOrEmpty(id))
                return null;

            var content = properties[ContentProp]?.ToString() ?? string.Empty;

            var metadata = new Dictionary<string, object>();
            var metaJson = properties[MetadataProp]?.ToString();
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

        /// <summary>
        /// Produces a deterministic UUID string for an arbitrary document id, since Weaviate object
        /// ids must be UUIDs. The mapping is stable so lookups/deletes work.
        /// </summary>
        private static string ToObjectId(string documentId)
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
                throw new HttpRequestException($"Weaviate {operation} failed with status {(int)info.Status}: {info.Body}");
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
