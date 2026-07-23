using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.DocumentStores
{
    /// <summary>
    /// Unit tests for the real Pinecone REST client. All HTTP traffic is intercepted by a mock
    /// <see cref="HttpMessageHandler"/> so these run in CI with no network access.
    /// </summary>
    public class PineconeDocumentStoreTests
    {
        private const string BaseUrl = "https://test-index.svc.pinecone.io";
        private const string Index = "test-index";

        // describe_index_stats for a 3-d index with 0 vectors in the default namespace.
        private const string StatsJson =
            "{\"namespaces\":{\"\":{\"vectorCount\":0}},\"dimension\":3,\"totalVectorCount\":0}";

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

        private static PineconeDocumentStore<float> StoreWith(MockHandler handler, string? ns = null)
        {
            var client = new HttpClient(handler) { BaseAddress = new Uri(BaseUrl) };
            return new PineconeDocumentStore<float>(Index, BaseUrl, apiKey: "k", @namespace: ns, httpClient: client);
        }

        private static VectorDocument<float> Doc(string id, string content, float[] vector, Dictionary<string, object>? metadata = null)
        {
            var d = new Document<float>(id, content, metadata ?? new Dictionary<string, object>());
            return new VectorDocument<float> { Document = d, Embedding = new Vector<float>(vector) };
        }

        #endregion

        [Fact]
        public void Constructor_ReadsStats_SetsDimensionAndCount()
        {
            var handler = new MockHandler()
                .On("POST", "/describe_index_stats", () => Json(
                    "{\"namespaces\":{\"\":{\"vectorCount\":7}},\"dimension\":8,\"totalVectorCount\":7}"));

            var store = StoreWith(handler);

            Assert.Equal(8, store.VectorDimension);
            Assert.Equal(7, store.DocumentCount);
            Assert.Equal(Index, store.IndexName);
        }

        [Fact]
        public void Add_Upserts_SendsCorrectRequest()
        {
            var handler = new MockHandler()
                .On("POST", "/vectors/upsert", () => Json("{\"upsertedCount\":1}"))
                .On("POST", "/describe_index_stats", () => Json(StatsJson));

            var store = StoreWith(handler);
            store.Add(Doc("doc1", "Hello world", new float[] { 1, 0, 0 },
                new Dictionary<string, object> { ["category"] = "science" }));

            Assert.Equal(1, store.DocumentCount);
            var upsert = handler.Requests.Single(r => r.Path.Contains("/vectors/upsert"));
            var vector = JObject.Parse(upsert.Body)["vectors"]![0]!;
            Assert.Equal("doc1", (string?)vector["id"]);
            Assert.Equal(3, ((JArray)vector["values"]!).Count);
            Assert.Equal("Hello world", (string?)vector["metadata"]!["_content"]);
            Assert.Equal("science", (string?)vector["metadata"]!["category"]);
        }

        [Fact]
        public void AddBatch_Upserts_AllVectors()
        {
            var handler = new MockHandler()
                .On("POST", "/vectors/upsert", () => Json("{\"upsertedCount\":3}"))
                .On("POST", "/describe_index_stats", () => Json(StatsJson));

            var store = StoreWith(handler);
            store.AddBatch(new List<VectorDocument<float>>
            {
                Doc("d1", "c1", new float[] { 1, 0, 0 }),
                Doc("d2", "c2", new float[] { 0, 1, 0 }),
                Doc("d3", "c3", new float[] { 0, 0, 1 })
            });

            Assert.Equal(3, store.DocumentCount);
            var upsert = handler.Requests.Single(r => r.Path.Contains("/vectors/upsert"));
            Assert.Equal(3, ((JArray)JObject.Parse(upsert.Body)["vectors"]!).Count);
        }

        [Fact]
        public void GetSimilar_ParsesMatches_InOrder()
        {
            var queryResponse =
                "{\"matches\":[" +
                "{\"id\":\"doc1\",\"score\":0.97,\"metadata\":{\"_content\":\"first\",\"category\":\"science\"}}," +
                "{\"id\":\"doc2\",\"score\":0.42,\"metadata\":{\"_content\":\"second\"}}" +
                "],\"namespace\":\"\"}";

            var handler = new MockHandler()
                .On("POST", "/query", () => Json(queryResponse))
                .On("POST", "/describe_index_stats", () => Json(StatsJson));

            var store = StoreWith(handler);
            var results = store.GetSimilar(new Vector<float>(new float[] { 1, 0, 0 }), topK: 2).ToList();

            Assert.Equal(2, results.Count);
            Assert.Equal("doc1", results[0].Id);
            Assert.Equal("first", results[0].Content);
            Assert.Equal("science", results[0].Metadata["category"].ToString());
            Assert.True(results[0].HasRelevanceScore);
            Assert.True(Convert.ToDouble(results[0].RelevanceScore) > Convert.ToDouble(results[1].RelevanceScore));

            var query = handler.Requests.Single(r => r.Path.Contains("/query"));
            var body = JObject.Parse(query.Body);
            Assert.Equal(2, (int)body["topK"]!);
            Assert.True((bool)body["includeMetadata"]!);
            Assert.Equal(3, ((JArray)body["vector"]!).Count);
        }

        [Fact]
        public void GetSimilarWithFilters_TranslatesEqInGte()
        {
            var handler = new MockHandler()
                .On("POST", "/query", () => Json("{\"matches\":[]}"))
                .On("POST", "/describe_index_stats", () => Json(StatsJson));

            var store = StoreWith(handler);
            var filters = new Dictionary<string, object>
            {
                ["category"] = "science",
                ["year"] = 2020,
                ["tags"] = new[] { "a", "b" }
            };
            store.GetSimilarWithFilters(new Vector<float>(new float[] { 1, 0, 0 }), 10, filters).ToList();

            var query = handler.Requests.Single(r => r.Path.Contains("/query"));
            var filter = JObject.Parse(query.Body)["filter"]!;

            Assert.Equal("science", (string?)filter["category"]!["$eq"]);
            Assert.Equal(2020.0, (double)filter["year"]!["$gte"]!);
            var inArray = (JArray)filter["tags"]!["$in"]!;
            Assert.Equal(new[] { "a", "b" }, inArray.Select(t => (string?)t).ToArray());
        }

        [Fact]
        public void GetById_Found_ReturnsDocument()
        {
            var handler = new MockHandler()
                .On("GET", "/vectors/fetch", () => Json(
                    "{\"vectors\":{\"doc1\":{\"id\":\"doc1\",\"metadata\":{\"_content\":\"hi\",\"k\":\"v\"}}}}"))
                .On("POST", "/describe_index_stats", () => Json(StatsJson));

            var store = StoreWith(handler);
            var doc = store.GetById("doc1");

            Assert.NotNull(doc);
            Assert.Equal("doc1", doc!.Id);
            Assert.Equal("hi", doc.Content);
            Assert.Equal("v", doc.Metadata["k"].ToString());
        }

        [Fact]
        public void GetById_Missing_ReturnsNull()
        {
            var handler = new MockHandler()
                .On("GET", "/vectors/fetch", () => Json("{\"vectors\":{}}"))
                .On("POST", "/describe_index_stats", () => Json(StatsJson));

            var store = StoreWith(handler);
            Assert.Null(store.GetById("missing"));
        }

        [Fact]
        public void Remove_SendsDeleteIds_ReturnsTrue()
        {
            var handler = new MockHandler()
                .On("POST", "/vectors/delete", () => Json("{}"))
                .On("POST", "/describe_index_stats", () => Json(StatsJson));

            var store = StoreWith(handler);
            var removed = store.Remove("doc1");

            Assert.True(removed);
            var delete = handler.Requests.Single(r => r.Path.Contains("/vectors/delete"));
            var ids = (JArray)JObject.Parse(delete.Body)["ids"]!;
            Assert.Equal("doc1", (string?)ids[0]);
        }

        [Fact]
        public void Clear_SendsDeleteAll()
        {
            var handler = new MockHandler()
                .On("POST", "/vectors/delete", () => Json("{}"))
                .On("POST", "/describe_index_stats", () => Json(StatsJson));

            var store = StoreWith(handler);
            store.Clear();

            Assert.Equal(0, store.DocumentCount);
            var delete = handler.Requests.Single(r => r.Path.Contains("/vectors/delete"));
            Assert.True((bool)JObject.Parse(delete.Body)["deleteAll"]!);
        }

        [Fact]
        public void Namespace_IsIncludedInRequests()
        {
            var handler = new MockHandler()
                .On("POST", "/vectors/upsert", () => Json("{\"upsertedCount\":1}"))
                .On("POST", "/describe_index_stats", () => Json(StatsJson));

            var store = StoreWith(handler, ns: "prod");
            store.Add(Doc("doc1", "c", new float[] { 1, 0, 0 }));

            var upsert = handler.Requests.Single(r => r.Path.Contains("/vectors/upsert"));
            Assert.Equal("prod", (string?)JObject.Parse(upsert.Body)["namespace"]);
            Assert.Equal("prod", store.Namespace);
        }

        [Fact]
        public void GetAll_ListsThenFetches()
        {
            var handler = new MockHandler()
                .On("GET", "/vectors/list", () => Json(
                    "{\"vectors\":[{\"id\":\"d1\"},{\"id\":\"d2\"}],\"pagination\":{}}"))
                .On("GET", "/vectors/fetch", body => Json(
                    "{\"vectors\":{\"d1\":{\"id\":\"d1\",\"metadata\":{\"_content\":\"c1\"}}," +
                    "\"d2\":{\"id\":\"d2\",\"metadata\":{\"_content\":\"c2\"}}}}"))
                .On("POST", "/describe_index_stats", () => Json(StatsJson));

            var store = StoreWith(handler);
            var all = store.GetAll().ToList();

            Assert.Equal(2, all.Count);
            Assert.Contains(all, d => d.Id == "d1");
            Assert.Contains(all, d => d.Id == "d2");
        }

        [Fact]
        public void Constructor_WithEmptyIndexName_Throws()
        {
            var client = new HttpClient(new MockHandler()) { BaseAddress = new Uri(BaseUrl) };
            Assert.Throws<ArgumentException>(() =>
                new PineconeDocumentStore<float>("", BaseUrl, "k", httpClient: client));
        }

        #region Integration (gated - require a live Pinecone index)

        [Trait("Category", "Integration")]
        [SkippableFact]
        public void Integration_UpsertQueryDelete_RoundTrips()
        {
            var url = Environment.GetEnvironmentVariable("PINECONE_INDEX_URL");
            var apiKey = Environment.GetEnvironmentVariable("PINECONE_API_KEY");
            Skip.If(string.IsNullOrWhiteSpace(url) || string.IsNullOrWhiteSpace(apiKey),
                "Set PINECONE_INDEX_URL and PINECONE_API_KEY to run Pinecone integration tests.");

            var store = new PineconeDocumentStore<float>("integration", url!, apiKey!,
                @namespace: "aidotnet-tests");

            var dim = store.VectorDimension > 0 ? store.VectorDimension : 8;
            var vec = Enumerable.Range(0, dim).Select(i => (float)(i == 0 ? 1.0 : 0.0)).ToArray();
            var id = "it-" + Guid.NewGuid().ToString("N");

            store.Add(Doc(id, "integration content", vec,
                new Dictionary<string, object> { ["category"] = "test" }));

            var results = store.GetSimilar(new Vector<float>(vec), topK: 5).ToList();
            Assert.Contains(results, r => r.Id == id);

            Assert.True(store.Remove(id));
        }

        #endregion
    }
}
