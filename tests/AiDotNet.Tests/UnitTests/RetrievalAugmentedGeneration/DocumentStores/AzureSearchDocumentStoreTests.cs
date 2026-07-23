using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.DocumentStores
{
    /// <summary>
    /// Unit tests for the real Azure AI Search REST client. All HTTP traffic is intercepted by a mock
    /// <see cref="HttpMessageHandler"/> so these run in CI with no network access.
    /// </summary>
    public class AzureSearchDocumentStoreTests
    {
        private const string Endpoint = "https://svc.search.windows.net";
        private const string Index = "test";

        private const string ExistingIndexJson =
            "{\"name\":\"test\",\"fields\":[{\"name\":\"embedding\",\"type\":\"Collection(Edm.Single)\",\"dimensions\":3}]}";

        #region infrastructure

        private sealed class RecordedRequest
        {
            public RecordedRequest(string method, string path, string body)
            {
                Method = method;
                Path = path;
                Body = body;
            }

            public string Method { get; }
            public string Path { get; }
            public string Body { get; }
        }

        private sealed class MockHandler : HttpMessageHandler
        {
            private readonly List<(string Method, string PathContains, Func<string, HttpResponseMessage> Resp)> _routes = new();
            public List<RecordedRequest> Requests { get; } = new();
            public string DefaultBody { get; set; } = "{}";

            public MockHandler On(string method, string pathContains, Func<HttpResponseMessage> resp)
            {
                _routes.Add((method, pathContains, _ => resp()));
                return this;
            }

            public MockHandler On(string method, string pathContains, Func<string, HttpResponseMessage> resp)
            {
                _routes.Add((method, pathContains, resp));
                return this;
            }

            protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            {
                var body = request.Content != null
                    ? await request.Content.ReadAsStringAsync().ConfigureAwait(false)
                    : string.Empty;
                var path = request.RequestUri!.AbsolutePath;
                Requests.Add(new RecordedRequest(request.Method.Method, request.RequestUri.PathAndQuery, body));

                foreach (var route in _routes)
                {
                    if (route.Method == request.Method.Method && path.Contains(route.PathContains))
                        return route.Resp(body);
                }

                return Json(DefaultBody);
            }
        }

        private static HttpResponseMessage Json(string body, HttpStatusCode code = HttpStatusCode.OK)
            => new HttpResponseMessage(code) { Content = new StringContent(body, Encoding.UTF8, "application/json") };

        private static AzureSearchDocumentStore<float> StoreWith(MockHandler handler,
            DistanceMetricType metric = DistanceMetricType.Cosine, int vectorDimension = 0)
        {
            var client = new HttpClient(handler) { BaseAddress = new Uri(Endpoint) };
            return new AzureSearchDocumentStore<float>(Index, Endpoint, apiKey: "k",
                distanceMetric: metric, vectorDimension: vectorDimension, httpClient: client);
        }

        private static VectorDocument<float> Doc(string id, string content, float[] vector, Dictionary<string, object>? metadata = null)
        {
            var d = new Document<float>(id, content, metadata ?? new Dictionary<string, object>());
            return new VectorDocument<float> { Document = d, Embedding = new Vector<float>(vector) };
        }

        #endregion

        [Fact]
        public void Constructor_ReadsExistingIndex_SetsDimension()
        {
            var handler = new MockHandler()
                .On("GET", "/indexes/test", () => Json(ExistingIndexJson));

            var store = StoreWith(handler);

            Assert.Equal(3, store.VectorDimension);
            Assert.Equal(Index, store.IndexName);
            Assert.Contains(handler.Requests, r => r.Method == "GET" && r.Path.Contains("api-version=2023-11-01"));
        }

        [Fact]
        public void Constructor_IndexMissing_CreatesWithHnswProfileAndMetric()
        {
            var handler = new MockHandler()
                .On("PUT", "/indexes/test", () => Json("{}"))
                .On("GET", "/indexes/test", () => Json("{}", HttpStatusCode.NotFound));

            var store = StoreWith(handler, metric: DistanceMetricType.Euclidean, vectorDimension: 4);

            Assert.Equal(4, store.VectorDimension);
            var create = handler.Requests.Single(r => r.Method == "PUT" && r.Path.StartsWith("/indexes/test"));
            var body = JObject.Parse(create.Body);
            var embedding = ((JArray)body["fields"]!).Single(f => (string?)f["name"] == "embedding");
            Assert.Equal(4, (int)embedding["dimensions"]!);
            Assert.Equal("vprofile", (string?)embedding["vectorSearchProfile"]);
            var algo = ((JArray)body["vectorSearch"]!["algorithms"]!)[0]!;
            Assert.Equal("hnsw", (string?)algo["kind"]);
            Assert.Equal("euclidean", (string?)algo["hnswParameters"]!["metric"]);
        }

        [Fact]
        public void Add_UploadsMergeOrUpload_WithVectorAndFlattenedMetadata()
        {
            var handler = new MockHandler()
                .On("GET", "/indexes/test", () => Json(ExistingIndexJson))
                .On("POST", "/indexes/test/docs/index", () => Json("{\"value\":[{\"key\":\"doc1\",\"status\":true}]}"));

            var store = StoreWith(handler);
            store.Add(Doc("doc1", "Hello world", new float[] { 1, 0, 0 },
                new Dictionary<string, object> { ["category"] = "science" }));

            Assert.Equal(1, store.DocumentCount);
            var upload = handler.Requests.Single(r => r.Method == "POST" && r.Path.Contains("/docs/index"));
            var action = JObject.Parse(upload.Body)["value"]![0]!;
            Assert.Equal("mergeOrUpload", (string?)action["@search.action"]);
            Assert.Equal("doc1", (string?)action["id"]);
            Assert.Equal("Hello world", (string?)action["content"]);
            Assert.Equal("science", (string?)action["category"]);
            Assert.Equal(3, ((JArray)action["embedding"]!).Count);
        }

        [Fact]
        public void GetSimilar_ParsesRankedHits_InOrder()
        {
            var searchResponse =
                "{\"value\":[" +
                "{\"@search.score\":0.98,\"id\":\"doc1\",\"content\":\"first\",\"metadata_json\":\"{\\\"category\\\":\\\"science\\\"}\"}," +
                "{\"@search.score\":0.55,\"id\":\"doc2\",\"content\":\"second\",\"metadata_json\":\"{}\"}" +
                "]}";

            var handler = new MockHandler()
                .On("POST", "/indexes/test/docs/search", () => Json(searchResponse))
                .On("GET", "/indexes/test", () => Json(ExistingIndexJson));

            var store = StoreWith(handler);
            var results = store.GetSimilar(new Vector<float>(new float[] { 1, 0, 0 }), topK: 2).ToList();

            Assert.Equal(2, results.Count);
            Assert.Equal("doc1", results[0].Id);
            Assert.Equal("first", results[0].Content);
            Assert.Equal("science", results[0].Metadata["category"].ToString());
            Assert.True(results[0].HasRelevanceScore);
            Assert.True(Convert.ToDouble(results[0].RelevanceScore) > Convert.ToDouble(results[1].RelevanceScore));

            var search = handler.Requests.Single(r => r.Path.Contains("/docs/search"));
            var body = JObject.Parse(search.Body);
            var vq = ((JArray)body["vectorQueries"]!)[0]!;
            Assert.Equal("vector", (string?)vq["kind"]);
            Assert.Equal("embedding", (string?)vq["fields"]);
            Assert.Equal(2, (int)vq["k"]!);
            Assert.Equal(3, ((JArray)vq["vector"]!).Count);
        }

        [Fact]
        public void GetSimilarWithFilters_TranslatesToOData()
        {
            var handler = new MockHandler()
                .On("POST", "/indexes/test/docs/search", () => Json("{\"value\":[]}"))
                .On("GET", "/indexes/test", () => Json(ExistingIndexJson));

            var store = StoreWith(handler);
            var filters = new Dictionary<string, object>
            {
                ["category"] = "science",
                ["year"] = 2020,
                ["tag"] = new List<string> { "a", "b" }
            };
            store.GetSimilarWithFilters(new Vector<float>(new float[] { 1, 0, 0 }), 10, filters).ToList();

            var filter = (string)JObject.Parse(handler.Requests.Single(r => r.Path.Contains("/docs/search")).Body)["filter"]!;
            Assert.Contains("category eq 'science'", filter);
            Assert.Contains("year ge 2020", filter);
            Assert.Contains("(tag eq 'a' or tag eq 'b')", filter);
            Assert.Contains(" and ", filter);
        }

        [Fact]
        public void GetById_Found_ReturnsDocument()
        {
            var handler = new MockHandler()
                .On("GET", "/indexes/test/docs/", () => Json(
                    "{\"id\":\"doc1\",\"content\":\"hi\",\"metadata_json\":\"{\\\"k\\\":\\\"v\\\"}\"}"))
                .On("GET", "/indexes/test", () => Json(ExistingIndexJson));

            var store = StoreWith(handler);
            var doc = store.GetById("doc1");

            Assert.NotNull(doc);
            Assert.Equal("doc1", doc!.Id);
            Assert.Equal("hi", doc.Content);
            Assert.Equal("v", doc.Metadata["k"].ToString());
        }

        [Fact]
        public void GetById_NotFound_ReturnsNull()
        {
            var handler = new MockHandler()
                .On("GET", "/indexes/test/docs/", () => Json("{}", HttpStatusCode.NotFound))
                .On("GET", "/indexes/test", () => Json(ExistingIndexJson));

            var store = StoreWith(handler);
            Assert.Null(store.GetById("missing"));
        }

        [Fact]
        public void Remove_SendsDeleteAction_ReturnsTrue()
        {
            var handler = new MockHandler()
                .On("POST", "/indexes/test/docs/index", () => Json("{\"value\":[{\"key\":\"doc1\",\"status\":true}]}"))
                .On("GET", "/indexes/test", () => Json(ExistingIndexJson));

            var store = StoreWith(handler);
            Assert.True(store.Remove("doc1"));

            var del = handler.Requests.Single(r => r.Method == "POST" && r.Path.Contains("/docs/index"));
            var action = JObject.Parse(del.Body)["value"]![0]!;
            Assert.Equal("delete", (string?)action["@search.action"]);
            Assert.Equal("doc1", (string?)action["id"]);
        }

        [Fact]
        public void GetAll_PagesUntilExhausted()
        {
            var call = 0;
            var handler = new MockHandler()
                .On("POST", "/indexes/test/docs/search", _ =>
                {
                    call++;
                    if (call == 1)
                    {
                        var items = string.Join(",", Enumerable.Range(0, 1000).Select(i =>
                            "{\"id\":\"d" + i + "\",\"content\":\"c\",\"metadata_json\":\"{}\"}"));
                        return Json("{\"value\":[" + items + "]}");
                    }
                    return Json("{\"value\":[{\"id\":\"dZ\",\"content\":\"c\",\"metadata_json\":\"{}\"}]}");
                })
                .On("GET", "/indexes/test", () => Json(ExistingIndexJson));

            var store = StoreWith(handler);
            var all = store.GetAll().ToList();

            Assert.Equal(1001, all.Count);
            Assert.Contains(all, d => d.Id == "dZ");
            Assert.Equal(2, handler.Requests.Count(r => r.Path.Contains("/docs/search")));
        }

        [Fact]
        public void Clear_DeletesIndex_AndRecreates()
        {
            var handler = new MockHandler()
                .On("DELETE", "/indexes/test", () => Json("{}", HttpStatusCode.NoContent))
                .On("PUT", "/indexes/test", () => Json("{}"))
                .On("GET", "/indexes/test", () => Json(ExistingIndexJson));

            // Existing index reports dim 3, so Clear recreates it.
            var store = StoreWith(handler);
            store.Clear();

            Assert.Equal(0, store.DocumentCount);
            Assert.Contains(handler.Requests, r => r.Method == "DELETE" && r.Path.StartsWith("/indexes/test"));
            Assert.Contains(handler.Requests, r => r.Method == "PUT" && r.Path.StartsWith("/indexes/test"));
        }

        [Fact]
        public void Constructor_UnsupportedMetric_Throws()
        {
            Assert.Throws<NotSupportedException>(() =>
                new AzureSearchDocumentStore<float>(Index, Endpoint, "k", distanceMetric: DistanceMetricType.Jaccard,
                    httpClient: new HttpClient(new MockHandler()) { BaseAddress = new Uri(Endpoint) }));
        }

        #region Integration (gated - require a live Azure AI Search service)

        [Trait("Category", "Integration")]
        [SkippableFact]
        public void Integration_UpsertSearchDelete_RoundTrips()
        {
            var endpoint = Environment.GetEnvironmentVariable("AZURE_SEARCH_ENDPOINT");
            var apiKey = Environment.GetEnvironmentVariable("AZURE_SEARCH_API_KEY");
            Skip.If(string.IsNullOrWhiteSpace(endpoint) || string.IsNullOrWhiteSpace(apiKey),
                "Set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY to run Azure AI Search integration tests.");

            var index = "aidotnet-it-" + Guid.NewGuid().ToString("N");
            var store = new AzureSearchDocumentStore<float>(index, endpoint!, apiKey!,
                distanceMetric: DistanceMetricType.Cosine, vectorDimension: 4);

            try
            {
                var id = "it-" + Guid.NewGuid().ToString("N");
                store.Add(Doc(id, "integration content", new float[] { 1, 0, 0, 0 },
                    new Dictionary<string, object> { ["category"] = "test" }));

                var results = store.GetSimilar(new Vector<float>(new float[] { 1, 0, 0, 0 }), topK: 5).ToList();
                Assert.Contains(results, r => r.Id == id);

                Assert.True(store.Remove(id));
            }
            finally
            {
                store.Clear();
            }
        }

        #endregion
    }
}
