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
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores
{
    /// <summary>
    /// Pinecone vector-database document store backed by the real Pinecone REST API.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    /// <remarks>
    /// <para>
    /// This store talks to a Pinecone index over HTTP using the index host as the base URL and the
    /// <c>Api-Key</c> header for authentication. It upserts vectors with metadata, queries with a
    /// top-k + filter + <c>includeMetadata</c>, deletes vectors and reads index statistics. Document
    /// content is stored inside the vector metadata under the reserved key <c>_content</c>. An optional
    /// namespace scopes every operation.
    /// </para>
    /// <para><b>For Beginners:</b> Pinecone is a fully managed cloud vector database. This class is a
    /// real client for it - each method makes an HTTP call to your Pinecone index. The index (and its
    /// vector dimension) is created and managed in the Pinecone console; this client reads the
    /// dimension from <c>describe_index_stats</c> when it starts up.
    /// </para>
    /// </remarks>
    [ComponentType(ComponentType.DocumentStore)]
    [PipelineStage(PipelineStage.Indexing)]
    public class PineconeDocumentStore<T> : DocumentStoreBase<T>
    {
        private const string ContentKey = "_content";

        private readonly HttpClient _httpClient;
        private readonly string _indexName;
        private readonly string _namespace;
        private int _vectorDimension;
        private int _documentCount;

        /// <summary>
        /// Gets the number of vectors currently stored in the index (within the configured namespace).
        /// </summary>
        public override int DocumentCount => _documentCount;

        /// <summary>
        /// Gets the dimensionality of vectors stored in this index.
        /// </summary>
        public override int VectorDimension => _vectorDimension;

        /// <summary>
        /// Gets the name of the Pinecone index this store is bound to.
        /// </summary>
        public string IndexName => _indexName;

        /// <summary>
        /// Gets the namespace all operations are scoped to (empty string means the default namespace).
        /// </summary>
        public string Namespace => _namespace;

        /// <summary>
        /// Initializes a new instance of the <see cref="PineconeDocumentStore{T}"/> class.
        /// </summary>
        /// <param name="indexName">A logical name for the index (used for diagnostics).</param>
        /// <param name="url">The index host base URL, e.g. <c>https://my-index-abc123.svc.us-east1-gcp.pinecone.io</c>.</param>
        /// <param name="apiKey">The Pinecone API key (sent as the <c>Api-Key</c> header).</param>
        /// <param name="namespace">Optional namespace that scopes all operations.</param>
        /// <param name="httpClient">
        /// Optional pre-configured <see cref="HttpClient"/>. Primarily for testing; when supplied its
        /// <see cref="HttpClient.BaseAddress"/> is used if already set.
        /// </param>
        /// <param name="handler">Optional <see cref="HttpMessageHandler"/> used to build the client (for testing).</param>
        public PineconeDocumentStore(
            string indexName,
            string url,
            string apiKey,
            string? @namespace = null,
            HttpClient? httpClient = null,
            HttpMessageHandler? handler = null)
        {
            if (string.IsNullOrWhiteSpace(indexName))
                throw new ArgumentException("Index name cannot be empty", nameof(indexName));

            _indexName = indexName;
            _namespace = @namespace ?? string.Empty;
            _documentCount = 0;
            _vectorDimension = 0;

            _httpClient = httpClient ?? (handler != null ? new HttpClient(handler) : new HttpClient());
            if (_httpClient.BaseAddress == null)
            {
                if (string.IsNullOrWhiteSpace(url))
                    throw new ArgumentException("Url cannot be empty", nameof(url));
                _httpClient.BaseAddress = new Uri(url);
            }

            if (!string.IsNullOrWhiteSpace(apiKey) && !_httpClient.DefaultRequestHeaders.Contains("Api-Key"))
                _httpClient.DefaultRequestHeaders.Add("Api-Key", apiKey);

            InitializeStats();
        }

        private void InitializeStats()
        {
            try
            {
                var info = SendAsync(HttpMethod.Post, "/describe_index_stats", new { }, CancellationToken.None).GetAwaiter().GetResult();
                if (!IsSuccess(info.Status))
                    return;

                var root = JObject.Parse(info.Body);
                var dimension = root["dimension"]?.Value<int>();
                if (dimension.HasValue && dimension.Value > 0)
                    _vectorDimension = dimension.Value;

                _documentCount = ReadNamespaceCount(root);
            }
            catch (HttpRequestException)
            {
                // Index host not reachable at construction; stats are refreshed on later operations.
            }
        }

        private int ReadNamespaceCount(JObject stats)
        {
            var namespaces = stats["namespaces"] as JObject;
            if (namespaces != null)
            {
                var key = _namespace.Length == 0 ? string.Empty : _namespace;
                var ns = namespaces[key] as JObject;
                if (ns != null)
                    return ns["vectorCount"]?.Value<int>() ?? 0;

                if (_namespace.Length == 0)
                    return stats["totalVectorCount"]?.Value<int>() ?? 0;

                return 0;
            }

            return stats["totalVectorCount"]?.Value<int>() ?? 0;
        }

        /// <inheritdoc/>
        protected override void AddCore(VectorDocument<T> vectorDocument)
            => AddCoreImplAsync(vectorDocument, CancellationToken.None).GetAwaiter().GetResult();

        /// <inheritdoc/>
        protected override Task AddCoreAsync(VectorDocument<T> vectorDocument, CancellationToken cancellationToken)
            => AddCoreImplAsync(vectorDocument, cancellationToken);

        private async Task AddCoreImplAsync(VectorDocument<T> vectorDocument, CancellationToken cancellationToken)
        {
            if (_vectorDimension == 0)
                _vectorDimension = vectorDocument.Embedding.Length;
            else if (vectorDocument.Embedding.Length != _vectorDimension)
                throw new ArgumentException(
                    $"Vector dimension mismatch. Expected {_vectorDimension}, got {vectorDocument.Embedding.Length}",
                    nameof(vectorDocument));

            await UpsertAsync(new[] { vectorDocument }, cancellationToken).ConfigureAwait(false);
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

            if (_vectorDimension == 0)
                _vectorDimension = vectorDocuments[0].Embedding.Length;

            foreach (var vd in vectorDocuments)
            {
                if (vd.Embedding.Length != _vectorDimension)
                    throw new ArgumentException(
                        $"Vector dimension mismatch in batch. Expected {_vectorDimension}, got {vd.Embedding.Length} for document {vd.Document.Id}",
                        nameof(vectorDocuments));
            }

            await UpsertAsync(vectorDocuments, cancellationToken).ConfigureAwait(false);
            _documentCount += vectorDocuments.Count;
        }

        private async Task UpsertAsync(IEnumerable<VectorDocument<T>> vectorDocuments, CancellationToken cancellationToken)
        {
            var vectors = vectorDocuments.Select(vd =>
            {
                var values = vd.Embedding.ToArray().Select(v => Convert.ToDouble(v)).ToArray();
                var metadata = BuildMetadata(vd.Document);
                return (object)new
                {
                    id = vd.Document.Id,
                    values,
                    metadata
                };
            }).ToList();

            var body = new Dictionary<string, object> { ["vectors"] = vectors };
            if (_namespace.Length > 0)
                body["namespace"] = _namespace;

            var info = await SendAsync(HttpMethod.Post, "/vectors/upsert", body, cancellationToken).ConfigureAwait(false);
            EnsureSuccess(info, "upsert");
        }

        private static Dictionary<string, object> BuildMetadata(Document<T> document)
        {
            var metadata = new Dictionary<string, object>();
            if (document.Metadata != null)
            {
                foreach (var kvp in document.Metadata)
                    metadata[kvp.Key] = kvp.Value;
            }

            metadata[ContentKey] = document.Content;
            return metadata;
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

            var body = new Dictionary<string, object>
            {
                ["vector"] = vector,
                ["topK"] = topK,
                ["includeMetadata"] = true,
                ["includeValues"] = false
            };

            var filter = BuildFilter(metadataFilters);
            if (filter != null)
                body["filter"] = filter;

            if (_namespace.Length > 0)
                body["namespace"] = _namespace;

            var info = await SendAsync(HttpMethod.Post, "/query", body, cancellationToken).ConfigureAwait(false);
            EnsureSuccess(info, "query");

            var results = new List<Document<T>>();
            var matches = JObject.Parse(info.Body)["matches"] as JArray;
            if (matches == null)
                return results;

            foreach (var match in matches)
            {
                var id = match?["id"]?.ToString();
                if (string.IsNullOrEmpty(id))
                    continue;

                var doc = ParseMetadata(id!, match?["metadata"] as JObject);
                var score = match?["score"] != null ? Convert.ToDouble(match!["score"], CultureInfo.InvariantCulture) : 0.0;
                doc.RelevanceScore = NumOps.FromDouble(score);
                doc.HasRelevanceScore = true;
                results.Add(doc);
            }

            return results;
        }

        /// <summary>
        /// Translates the metadata filter dictionary into a Pinecone metadata filter object.
        /// </summary>
        /// <remarks>
        /// Equality (string/bool) becomes <c>$eq</c>, numeric values become <c>$gte</c> (mirroring the
        /// base-class "field &gt;= value" semantics) and collection values become <c>$in</c>. Multiple
        /// filters are ANDed together, matching Pinecone's default combination of top-level keys.
        /// </remarks>
        private static object? BuildFilter(Dictionary<string, object>? metadataFilters)
        {
            if (metadataFilters == null || metadataFilters.Count == 0)
                return null;

            var filter = new Dictionary<string, object>();

            foreach (var kvp in metadataFilters)
            {
                var value = kvp.Value;

                if (value == null || value is string || value is bool)
                {
                    filter[kvp.Key] = new Dictionary<string, object> { ["$eq"] = value! };
                }
                else if (IsNumeric(value))
                {
                    filter[kvp.Key] = new Dictionary<string, object>
                    {
                        ["$gte"] = Convert.ToDouble(value, CultureInfo.InvariantCulture)
                    };
                }
                else if (value is System.Collections.IEnumerable enumerable)
                {
                    var items = enumerable.Cast<object>().ToList();
                    filter[kvp.Key] = new Dictionary<string, object> { ["$in"] = items };
                }
                else
                {
                    filter[kvp.Key] = new Dictionary<string, object> { ["$eq"] = value.ToString() ?? string.Empty };
                }
            }

            return filter;
        }

        /// <inheritdoc/>
        protected override Document<T>? GetByIdCore(string documentId)
            => GetByIdCoreImplAsync(documentId, CancellationToken.None).GetAwaiter().GetResult();

        /// <inheritdoc/>
        protected override Task<Document<T>?> GetByIdCoreAsync(string documentId, CancellationToken cancellationToken)
            => GetByIdCoreImplAsync(documentId, cancellationToken);

        private async Task<Document<T>?> GetByIdCoreImplAsync(string documentId, CancellationToken cancellationToken)
        {
            var path = $"/vectors/fetch?ids={Uri.EscapeDataString(documentId)}";
            if (_namespace.Length > 0)
                path += $"&namespace={Uri.EscapeDataString(_namespace)}";

            var info = await SendAsync(HttpMethod.Get, path, null, cancellationToken).ConfigureAwait(false);
            if (info.Status == HttpStatusCode.NotFound)
                return null;
            EnsureSuccess(info, "fetch");

            var vectors = JObject.Parse(info.Body)["vectors"] as JObject;
            var vector = vectors?[documentId] as JObject;
            if (vector == null)
                return null;

            return ParseMetadata(documentId, vector["metadata"] as JObject);
        }

        /// <inheritdoc/>
        protected override bool RemoveCore(string documentId)
            => RemoveCoreImplAsync(documentId, CancellationToken.None).GetAwaiter().GetResult();

        /// <inheritdoc/>
        protected override Task<bool> RemoveCoreAsync(string documentId, CancellationToken cancellationToken)
            => RemoveCoreImplAsync(documentId, cancellationToken);

        private async Task<bool> RemoveCoreImplAsync(string documentId, CancellationToken cancellationToken)
        {
            var body = new Dictionary<string, object> { ["ids"] = new[] { documentId } };
            if (_namespace.Length > 0)
                body["namespace"] = _namespace;

            var info = await SendAsync(HttpMethod.Post, "/vectors/delete", body, cancellationToken).ConfigureAwait(false);
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
            string? paginationToken = null;

            while (true)
            {
                var path = "/vectors/list?limit=100";
                if (_namespace.Length > 0)
                    path += $"&namespace={Uri.EscapeDataString(_namespace)}";
                if (!string.IsNullOrEmpty(paginationToken))
                    path += $"&paginationToken={Uri.EscapeDataString(paginationToken!)}";

                var info = await SendAsync(HttpMethod.Get, path, null, cancellationToken).ConfigureAwait(false);
                EnsureSuccess(info, "list");

                var root = JObject.Parse(info.Body);
                var ids = (root["vectors"] as JArray)?
                    .Select(v => v?["id"]?.ToString())
                    .Where(id => !string.IsNullOrEmpty(id))
                    .Select(id => id!)
                    .ToList() ?? new List<string>();

                foreach (var id in ids)
                {
                    var doc = await GetByIdCoreImplAsync(id, cancellationToken).ConfigureAwait(false);
                    if (doc != null)
                        all.Add(doc);
                }

                paginationToken = root["pagination"]?["next"]?.ToString();
                if (string.IsNullOrEmpty(paginationToken))
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
            var body = new Dictionary<string, object> { ["deleteAll"] = true };
            if (_namespace.Length > 0)
                body["namespace"] = _namespace;

            var info = await SendAsync(HttpMethod.Post, "/vectors/delete", body, cancellationToken).ConfigureAwait(false);
            EnsureSuccess(info, "delete all");

            _documentCount = 0;
        }

        private Document<T> ParseMetadata(string id, JObject? metadata)
        {
            var content = string.Empty;
            var docMetadata = new Dictionary<string, object>();

            if (metadata != null)
            {
                foreach (var property in metadata.Properties())
                {
                    if (property.Name == ContentKey)
                        content = property.Value?.ToString() ?? string.Empty;
                    else
                        docMetadata[property.Name] = property.Value?.ToObject<object>() ?? string.Empty;
                }
            }

            return new Document<T>(id, content, docMetadata);
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
                throw new HttpRequestException($"Pinecone {operation} failed with status {(int)info.Status}: {info.Body}");
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
