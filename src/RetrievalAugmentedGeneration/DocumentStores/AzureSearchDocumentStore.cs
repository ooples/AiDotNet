using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores
{
    /// <summary>
    /// Azure AI Search document store backed by the real Azure AI Search REST API.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    /// <remarks>
    /// <para>
    /// This store talks to an Azure AI Search service over HTTP. It manages a single index with a
    /// vector field and an HNSW vector-search profile, upserts documents via the <c>mergeOrUpload</c>
    /// action, performs vector search with an OData <c>$filter</c>, deletes documents via the
    /// <c>delete</c> action and pages the whole index. The original id and content are stored under
    /// <c>id</c> and <c>content</c>; the full metadata dictionary is stored as a JSON string under
    /// <c>metadata_json</c> (for lossless round-tripping) while each scalar metadata entry is also
    /// flattened to a top-level field so it can be used in <c>$filter</c> expressions.
    /// </para>
    /// <para><b>For Beginners:</b> Azure AI Search is Microsoft's managed search-and-vector service.
    /// This class is a real client for it - every method here makes an HTTP call to your search
    /// service. Authentication uses the <c>api-key</c> header and every request carries an
    /// <c>api-version</c> query-string parameter.
    /// </para>
    /// <para><b>API deviation:</b> Azure indexes have a fixed schema, so metadata keys used in filters
    /// must exist as filterable fields in the index. This client creates the index with the base fields
    /// (<c>id</c>, <c>content</c>, <c>metadata_json</c>, <c>embedding</c>); flattened metadata fields are
    /// still sent on upload and referenced by <c>$filter</c>, but for production filtering the caller
    /// should ensure those fields are declared in the index (or pre-create the index).
    /// </para>
    /// </remarks>
    [ComponentType(ComponentType.DocumentStore)]
    [PipelineStage(PipelineStage.Indexing)]
    public class AzureSearchDocumentStore<T> : DocumentStoreBase<T>
    {
        private const string IdField = "id";
        private const string ContentField = "content";
        private const string MetadataField = "metadata_json";
        private const string EmbeddingField = "embedding";
        private const string VectorProfile = "vprofile";
        private const string VectorAlgorithm = "hnsw-algo";

        private readonly HttpClient _httpClient;
        private readonly string _indexName;
        private readonly string _apiVersion;
        private readonly string _vectorMetric;
        private int _vectorDimension;
        private int _documentCount;
        private bool _indexReady;

        /// <summary>
        /// Gets the number of documents currently stored in the index.
        /// </summary>
        public override int DocumentCount => _documentCount;

        /// <summary>
        /// Gets the dimensionality of vectors stored in this index.
        /// </summary>
        public override int VectorDimension => _vectorDimension;

        /// <summary>
        /// Gets the name of the Azure AI Search index this store is bound to.
        /// </summary>
        public string IndexName { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="AzureSearchDocumentStore{T}"/> class.
        /// </summary>
        /// <param name="indexName">The Azure AI Search index name.</param>
        /// <param name="endpoint">The service endpoint, e.g. <c>https://myservice.search.windows.net</c>.</param>
        /// <param name="apiKey">The Azure AI Search admin API key (sent as the <c>api-key</c> header).</param>
        /// <param name="distanceMetric">The vector distance metric used when creating the index.</param>
        /// <param name="vectorDimension">
        /// The vector dimension used to create the index if it does not already exist.
        /// When 0 the index must already exist or the dimension is inferred from the first document.
        /// </param>
        /// <param name="apiVersion">The REST API version query-string value.</param>
        /// <param name="httpClient">Optional pre-configured <see cref="HttpClient"/> (primarily for testing).</param>
        /// <param name="handler">Optional <see cref="HttpMessageHandler"/> used to build the client (for testing).</param>
        public AzureSearchDocumentStore(
            string indexName,
            string endpoint,
            string apiKey,
            DistanceMetricType distanceMetric = DistanceMetricType.Cosine,
            int vectorDimension = 0,
            string apiVersion = "2023-11-01",
            HttpClient? httpClient = null,
            HttpMessageHandler? handler = null)
        {
            if (string.IsNullOrWhiteSpace(indexName))
                throw new ArgumentException("Index name cannot be empty", nameof(indexName));
            if (vectorDimension < 0)
                throw new ArgumentOutOfRangeException(nameof(vectorDimension), "Vector dimension cannot be negative");

            _indexName = indexName;
            IndexName = indexName;
            _apiVersion = string.IsNullOrWhiteSpace(apiVersion) ? "2023-11-01" : apiVersion;
            _vectorMetric = MapMetric(distanceMetric);
            _vectorDimension = vectorDimension;
            _documentCount = 0;

            _httpClient = httpClient ?? (handler != null ? new HttpClient(handler) : new HttpClient());
            if (_httpClient.BaseAddress == null)
            {
                if (string.IsNullOrWhiteSpace(endpoint))
                    throw new ArgumentException("Endpoint cannot be empty", nameof(endpoint));
                _httpClient.BaseAddress = new Uri(endpoint);
            }

            if (!string.IsNullOrWhiteSpace(apiKey) && !_httpClient.DefaultRequestHeaders.Contains("api-key"))
                _httpClient.DefaultRequestHeaders.Add("api-key", apiKey);

            InitializeIndex(vectorDimension);
        }

        private static string MapMetric(DistanceMetricType metric)
        {
            switch (metric)
            {
                case DistanceMetricType.Cosine:
                    return "cosine";
                case DistanceMetricType.Euclidean:
                    return "euclidean";
                default:
                    throw new NotSupportedException(
                        $"Metric '{metric}' is not supported by Azure AI Search. Use Cosine or Euclidean (or dotProduct via a pre-created index).");
            }
        }

        private string WithApiVersion(string path)
        {
            var separator = path.Contains("?") ? "&" : "?";
            return path + separator + "api-version=" + _apiVersion;
        }

        private void InitializeIndex(int requestedDimension)
        {
            HttpResponseInfo info;
            try
            {
                info = SendAsync(HttpMethod.Get, WithApiVersion($"/indexes/{_indexName}"), null).GetAwaiter().GetResult();
            }
            catch (HttpRequestException)
            {
                // Service not reachable at construction time; defer creation to first write.
                return;
            }

            if (info.Status == HttpStatusCode.OK)
            {
                _indexReady = true;
                var fields = JObject.Parse(info.Body)["fields"] as JArray;
                if (fields != null)
                {
                    foreach (var field in fields)
                    {
                        if ((string?)field?["name"] == EmbeddingField)
                        {
                            var dim = field?["dimensions"];
                            if (dim != null && dim.Type != JTokenType.Null)
                            {
                                var parsed = Convert.ToInt32(dim, CultureInfo.InvariantCulture);
                                if (parsed > 0)
                                    _vectorDimension = parsed;
                            }
                        }
                    }
                }
                return;
            }

            if (info.Status == HttpStatusCode.NotFound && requestedDimension > 0)
                CreateIndex(requestedDimension);
        }

        private void CreateIndex(int dimension)
        {
            var body = new
            {
                name = _indexName,
                fields = new object[]
                {
                    new { name = IdField, type = "Edm.String", key = true, filterable = true, retrievable = true },
                    new { name = ContentField, type = "Edm.String", searchable = true, retrievable = true },
                    new { name = MetadataField, type = "Edm.String", retrievable = true },
                    new
                    {
                        name = EmbeddingField,
                        type = "Collection(Edm.Single)",
                        searchable = true,
                        retrievable = false,
                        dimensions = dimension,
                        vectorSearchProfile = VectorProfile
                    }
                },
                vectorSearch = new
                {
                    algorithms = new object[]
                    {
                        new
                        {
                            name = VectorAlgorithm,
                            kind = "hnsw",
                            hnswParameters = new { metric = _vectorMetric, m = 4, efConstruction = 400, efSearch = 500 }
                        }
                    },
                    profiles = new object[]
                    {
                        new { name = VectorProfile, algorithm = VectorAlgorithm }
                    }
                }
            };

            var info = SendAsync(HttpMethod.Put, WithApiVersion($"/indexes/{_indexName}"), body).GetAwaiter().GetResult();
            EnsureSuccess(info, "create index");
            _vectorDimension = dimension;
            _indexReady = true;
        }

        private void EnsureIndexForDimension(int dimension)
        {
            if (_indexReady)
                return;
            CreateIndex(dimension);
        }

        /// <inheritdoc/>
        protected override void AddCore(VectorDocument<T> vectorDocument)
        {
            EnsureIndexForDimension(vectorDocument.Embedding.Length);
            if (_vectorDimension == 0)
                _vectorDimension = vectorDocument.Embedding.Length;

            UploadActions(new[] { BuildAction(vectorDocument, "mergeOrUpload") }, "upload document");
            _documentCount++;
        }

        /// <inheritdoc/>
        protected override void AddBatchCore(IList<VectorDocument<T>> vectorDocuments)
        {
            if (vectorDocuments.Count == 0)
                return;

            EnsureIndexForDimension(vectorDocuments[0].Embedding.Length);
            if (_vectorDimension == 0)
                _vectorDimension = vectorDocuments[0].Embedding.Length;

            var actions = vectorDocuments.Select(d => BuildAction(d, "mergeOrUpload")).ToList();
            UploadActions(actions, "batch upload documents");
            _documentCount += vectorDocuments.Count;
        }

        private void UploadActions(IEnumerable<Dictionary<string, object?>> actions, string operation)
        {
            var body = new { value = actions };
            var info = SendAsync(HttpMethod.Post, WithApiVersion($"/indexes/{_indexName}/docs/index"), body).GetAwaiter().GetResult();
            EnsureSuccess(info, operation);
        }

        private Dictionary<string, object?> BuildAction(VectorDocument<T> vectorDocument, string action)
        {
            var vector = vectorDocument.Embedding.ToArray().Select(v => Convert.ToDouble(v)).ToArray();
            var metadata = vectorDocument.Document.Metadata ?? new Dictionary<string, object>();

            var doc = new Dictionary<string, object?>
            {
                ["@search.action"] = action,
                [IdField] = vectorDocument.Document.Id,
                [ContentField] = vectorDocument.Document.Content,
                [MetadataField] = JsonConvert.SerializeObject(metadata),
                [EmbeddingField] = vector
            };

            foreach (var kvp in metadata)
                doc[kvp.Key] = kvp.Value;

            return doc;
        }

        /// <inheritdoc/>
        protected override IEnumerable<Document<T>> GetSimilarCore(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters)
        {
            var vector = queryVector.ToArray().Select(v => Convert.ToDouble(v)).ToArray();

            var body = new Dictionary<string, object>
            {
                ["vectorQueries"] = new object[]
                {
                    new { kind = "vector", vector, fields = EmbeddingField, k = topK }
                },
                ["select"] = $"{IdField},{ContentField},{MetadataField}",
                ["top"] = topK
            };

            var filter = BuildODataFilter(metadataFilters);
            if (filter != null)
                body["filter"] = filter;

            var info = SendAsync(HttpMethod.Post, WithApiVersion($"/indexes/{_indexName}/docs/search"), body).GetAwaiter().GetResult();
            EnsureSuccess(info, "search");

            var results = new List<Document<T>>();
            var hits = JObject.Parse(info.Body)["value"] as JArray;
            if (hits == null)
                return results;

            foreach (var hit in hits)
            {
                var doc = ParseDocument(hit as JObject);
                if (doc == null)
                    continue;

                var scoreToken = hit?["@search.score"];
                var score = scoreToken != null && scoreToken.Type != JTokenType.Null
                    ? Convert.ToDouble(scoreToken, CultureInfo.InvariantCulture)
                    : 0.0;
                doc.RelevanceScore = NumOps.FromDouble(score);
                doc.HasRelevanceScore = true;
                results.Add(doc);
            }

            return results;
        }

        /// <summary>
        /// Translates the metadata filter dictionary into an Azure AI Search OData <c>$filter</c> string.
        /// </summary>
        /// <remarks>
        /// Equality (string/bool) uses <c>eq</c>, numeric values use <c>ge</c> (mirroring the base-class
        /// "field &gt;= value" semantics) and collection values become an <c>or</c> group of <c>eq</c>
        /// comparisons (any-of). All conditions are combined with <c>and</c>.
        /// </remarks>
        private static string? BuildODataFilter(Dictionary<string, object>? metadataFilters)
        {
            if (metadataFilters == null || metadataFilters.Count == 0)
                return null;

            var clauses = new List<string>();

            foreach (var kvp in metadataFilters)
            {
                var field = kvp.Key;
                var value = kvp.Value;

                if (value == null)
                {
                    clauses.Add($"{field} eq null");
                }
                else if (value is string s)
                {
                    clauses.Add($"{field} eq {ODataString(s)}");
                }
                else if (value is bool b)
                {
                    clauses.Add($"{field} eq {(b ? "true" : "false")}");
                }
                else if (IsNumeric(value))
                {
                    clauses.Add($"{field} ge {Convert.ToDouble(value, CultureInfo.InvariantCulture).ToString("R", CultureInfo.InvariantCulture)}");
                }
                else if (value is System.Collections.IEnumerable enumerable)
                {
                    var anyOf = new List<string>();
                    foreach (var item in enumerable)
                    {
                        if (item is string || item == null)
                            anyOf.Add($"{field} eq {ODataString(item?.ToString() ?? string.Empty)}");
                        else if (item is bool ib)
                            anyOf.Add($"{field} eq {(ib ? "true" : "false")}");
                        else if (IsNumeric(item))
                            anyOf.Add($"{field} eq {Convert.ToDouble(item, CultureInfo.InvariantCulture).ToString("R", CultureInfo.InvariantCulture)}");
                        else
                            anyOf.Add($"{field} eq {ODataString(item.ToString() ?? string.Empty)}");
                    }
                    if (anyOf.Count > 0)
                        clauses.Add("(" + string.Join(" or ", anyOf) + ")");
                }
                else
                {
                    clauses.Add($"{field} eq {ODataString(value.ToString() ?? string.Empty)}");
                }
            }

            return clauses.Count > 0 ? string.Join(" and ", clauses) : null;
        }

        // OData string literals are single-quoted; embedded single quotes are doubled.
        private static string ODataString(string value) => "'" + value.Replace("'", "''") + "'";

        /// <inheritdoc/>
        protected override Document<T>? GetByIdCore(string documentId)
        {
            var encoded = Uri.EscapeDataString(documentId);
            var info = SendAsync(HttpMethod.Get, WithApiVersion($"/indexes/{_indexName}/docs/{encoded}"), null).GetAwaiter().GetResult();
            if (info.Status == HttpStatusCode.NotFound)
                return null;
            EnsureSuccess(info, "get document");

            return ParseDocument(JObject.Parse(info.Body));
        }

        /// <inheritdoc/>
        protected override bool RemoveCore(string documentId)
        {
            var action = new Dictionary<string, object?>
            {
                ["@search.action"] = "delete",
                [IdField] = documentId
            };

            var body = new { value = new[] { action } };
            var info = SendAsync(HttpMethod.Post, WithApiVersion($"/indexes/{_indexName}/docs/index"), body).GetAwaiter().GetResult();
            if (!IsSuccess(info.Status))
                return false;

            if (_documentCount > 0)
                _documentCount--;
            return true;
        }

        /// <inheritdoc/>
        protected override IEnumerable<Document<T>> GetAllCore()
        {
            var all = new List<Document<T>>();
            const int pageSize = 1000;
            var skip = 0;

            while (true)
            {
                var body = new Dictionary<string, object>
                {
                    ["search"] = "*",
                    ["select"] = $"{IdField},{ContentField},{MetadataField}",
                    ["top"] = pageSize,
                    ["skip"] = skip
                };

                var info = SendAsync(HttpMethod.Post, WithApiVersion($"/indexes/{_indexName}/docs/search"), body).GetAwaiter().GetResult();
                EnsureSuccess(info, "list documents");

                var rows = JObject.Parse(info.Body)["value"] as JArray;
                if (rows == null || rows.Count == 0)
                    break;

                foreach (var row in rows)
                {
                    var doc = ParseDocument(row as JObject);
                    if (doc != null)
                        all.Add(doc);
                }

                if (rows.Count < pageSize)
                    break;
                skip += pageSize;
            }

            return all;
        }

        /// <inheritdoc/>
        public override void Clear()
        {
            var info = SendAsync(HttpMethod.Delete, WithApiVersion($"/indexes/{_indexName}"), null).GetAwaiter().GetResult();
            if (!IsSuccess(info.Status) && info.Status != HttpStatusCode.NotFound)
                EnsureSuccess(info, "delete index");

            _documentCount = 0;
            _indexReady = false;

            if (_vectorDimension > 0)
                CreateIndex(_vectorDimension);
        }

        private Document<T>? ParseDocument(JObject? row)
        {
            if (row == null)
                return null;

            var id = row[IdField]?.ToString();
            if (string.IsNullOrEmpty(id))
                return null;

            var content = row[ContentField]?.ToString() ?? string.Empty;

            var metadata = new Dictionary<string, object>();
            var metaJson = row[MetadataField]?.ToString();
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
                throw new HttpRequestException($"Azure AI Search {operation} failed with status {(int)info.Status}: {info.Body}");
        }

        private async Task<HttpResponseInfo> SendAsync(HttpMethod method, string path, object? body)
        {
            using (var request = new HttpRequestMessage(method, path))
            {
                if (body != null)
                {
                    var json = JsonConvert.SerializeObject(body);
                    request.Content = new StringContent(json, Encoding.UTF8, "application/json");
                }

                using (var response = await _httpClient.SendAsync(request).ConfigureAwait(false))
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
