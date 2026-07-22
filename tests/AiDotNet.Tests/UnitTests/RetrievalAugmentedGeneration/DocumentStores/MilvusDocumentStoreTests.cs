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
    /// Unit tests for the real Milvus v2 REST client. All HTTP traffic is intercepted by a mock
    /// <see cref="HttpMessageHandler"/> so these run in CI with no network access.
    /// </summary>
    public class MilvusDocumentStoreTests
    {
        private const string BaseUrl = "http://localhost:19530";
        private const string Collection = "test";

        // describe response for an existing 3-d collection.
        private const string ExistingDescribeJson =
            "{\"code\":0,\"data\":{\"collectionName\":\"test\",\"fields\":[{\"name\":\"vector\",\"params\":{\"dim\":3}}]}}";

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
            public string DefaultBody { get; set; } = "{\"code\":0,\"data\":{}}";

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

        private static MilvusDocumentStore<float> StoreWith(MockHandler handler,
            DistanceMetricType metric = DistanceMetricType.Cosine, int vectorDimension = 0)
        {
            var client = new HttpClient(handler) { BaseAddress = new Uri(BaseUrl) };
            return new MilvusDocumentStore<float>(Collection, BaseUrl, token: "t",
                distanceMetric: metric, vectorDimension: vectorDimension, httpClient: client);
        }

        private static VectorDocument<float> Doc(string id, string content, float[] vector, Dictionary<string, object>? metadata = null)
        {
            var d = new Document<float>(id, content, metadata ?? new Dictionary<string, object>());
            return new VectorDocument<float> { Document = d, Embedding = new Vector<float>(vector) };
        }

        #endregion

        [Fact]
        public void Constructor_ReadsExistingCollection_SetsDimension()
        {
            var handler = new MockHandler()
                .On("POST", "/collections/describe", () => Json(ExistingDescribeJson));

            var store = StoreWith(handler);

            Assert.Equal(3, store.VectorDimension);
            Assert.Equal(Collection, store.CollectionName);
        }

        [Fact]
        public void Constructor_CollectionMissing_CreatesWithMappedMetric()
        {
            var handler = new MockHandler()
                .On("POST", "/collections/describe", () => Json("{\"code\":100,\"message\":\"collection not found\"}"))
                .On("POST", "/collections/create", () => Json("{\"code\":0,\"data\":{}}"));

            var store = StoreWith(handler, metric: DistanceMetricType.Euclidean, vectorDimension: 4);

            Assert.Equal(4, store.VectorDimension);
            var create = handler.Requests.Single(r => r.Path.Contains("/collections/create"));
            var body = JObject.Parse(create.Body);
            Assert.Equal(4, (int)body["dimension"]!);
            Assert.Equal("L2", (string?)body["metricType"]);
            Assert.Equal("VarChar", (string?)body["idType"]);
            Assert.True((bool)body["enableDynamicField"]!);
        }

        [Fact]
        public void Add_UpsertsEntity_SendsCorrectRequest()
        {
            var handler = new MockHandler()
                .On("POST", "/collections/describe", () => Json(ExistingDescribeJson))
                .On("POST", "/entities/upsert", () => Json("{\"code\":0,\"data\":{\"upsertCount\":1}}"));

            var store = StoreWith(handler);
            store.Add(Doc("doc1", "Hello world", new float[] { 1, 0, 0 },
                new Dictionary<string, object> { ["category"] = "science" }));

            Assert.Equal(1, store.DocumentCount);
            var upsert = handler.Requests.Single(r => r.Path.Contains("/entities/upsert"));
            var body = JObject.Parse(upsert.Body);
            Assert.Equal("test", (string?)body["collectionName"]);
            var row = body["data"]![0]!;
            Assert.Equal("doc1", (string?)row["id"]);
            Assert.Equal("Hello world", (string?)row["content"]);
            Assert.Equal("science", (string?)row["m_category"]);
            Assert.Equal(3, ((JArray)row["vector"]!).Count);
        }

        [Fact]
        public void GetSimilar_ParsesRankedHits_InOrder()
        {
            var searchResponse =
                "{\"code\":0,\"data\":[" +
                "{\"id\":\"doc1\",\"distance\":0.98,\"content\":\"first\",\"metadata_json\":\"{\\\"category\\\":\\\"science\\\"}\"}," +
                "{\"id\":\"doc2\",\"distance\":0.55,\"content\":\"second\",\"metadata_json\":\"{}\"}" +
                "]}";

            var handler = new MockHandler()
                .On("POST", "/entities/search", () => Json(searchResponse))
                .On("POST", "/collections/describe", () => Json(ExistingDescribeJson));

            var store = StoreWith(handler);
            var results = store.GetSimilar(new Vector<float>(new float[] { 1, 0, 0 }), topK: 2).ToList();

            Assert.Equal(2, results.Count);
            Assert.Equal("doc1", results[0].Id);
            Assert.Equal("first", results[0].Content);
            Assert.True(results[0].HasRelevanceScore);
            Assert.True(Convert.ToDouble(results[0].RelevanceScore) > Convert.ToDouble(results[1].RelevanceScore));

            var search = handler.Requests.Single(r => r.Path.Contains("/entities/search"));
            var body = JObject.Parse(search.Body);
            Assert.Equal(2, (int)body["limit"]!);
            Assert.Equal("vector", (string?)body["annsField"]);
            Assert.Equal(3, ((JArray)body["data"]![0]!).Count);
        }

        [Fact]
        public void GetSimilarWithFilters_TranslatesEqualityRangeAndAnyOf()
        {
            var handler = new MockHandler()
                .On("POST", "/entities/search", () => Json("{\"code\":0,\"data\":[]}"))
                .On("POST", "/collections/describe", () => Json(ExistingDescribeJson));

            var store = StoreWith(handler);
            var filters = new Dictionary<string, object>
            {
                ["category"] = "science",
                ["year"] = 2020,
                ["tag"] = new List<string> { "a", "b" }
            };
            store.GetSimilarWithFilters(new Vector<float>(new float[] { 1, 0, 0 }), 10, filters).ToList();

            var expr = (string)JObject.Parse(handler.Requests.Single(r => r.Path.Contains("/entities/search")).Body)["filter"]!;
            Assert.Contains("m_category == \"science\"", expr);
            Assert.Contains("m_year >= 2020", expr);
            Assert.Contains("m_tag in [\"a\", \"b\"]", expr);
            Assert.Contains(" and ", expr);
        }

        [Fact]
        public void GetById_Found_ReturnsDocument()
        {
            var handler = new MockHandler()
                .On("POST", "/entities/query", () => Json(
                    "{\"code\":0,\"data\":[{\"id\":\"doc1\",\"content\":\"hi\",\"metadata_json\":\"{\\\"k\\\":\\\"v\\\"}\"}]}"))
                .On("POST", "/collections/describe", () => Json(ExistingDescribeJson));

            var store = StoreWith(handler);
            var doc = store.GetById("doc1");

            Assert.NotNull(doc);
            Assert.Equal("doc1", doc!.Id);
            Assert.Equal("hi", doc.Content);
            Assert.Equal("v", doc.Metadata["k"].ToString());

            var query = handler.Requests.Single(r => r.Path.Contains("/entities/query"));
            Assert.Contains("id == \"doc1\"", (string)JObject.Parse(query.Body)["filter"]!);
        }

        [Fact]
        public void GetById_NotFound_ReturnsNull()
        {
            var handler = new MockHandler()
                .On("POST", "/entities/query", () => Json("{\"code\":0,\"data\":[]}"))
                .On("POST", "/collections/describe", () => Json(ExistingDescribeJson));

            var store = StoreWith(handler);
            Assert.Null(store.GetById("missing"));
        }

        [Fact]
        public void Remove_SendsDelete_ReturnsTrue()
        {
            var handler = new MockHandler()
                .On("POST", "/entities/delete", () => Json("{\"code\":0,\"data\":{}}"))
                .On("POST", "/collections/describe", () => Json(ExistingDescribeJson));

            var store = StoreWith(handler);
            Assert.True(store.Remove("doc1"));

            var delete = handler.Requests.Single(r => r.Path.Contains("/entities/delete"));
            Assert.Contains("id == \"doc1\"", (string)JObject.Parse(delete.Body)["filter"]!);
        }

        [Fact]
        public void GetAll_PagesUntilExhausted()
        {
            var call = 0;
            var handler = new MockHandler()
                .On("POST", "/entities/query", _ =>
                {
                    call++;
                    if (call == 1)
                    {
                        var items = string.Join(",", Enumerable.Range(0, 1000).Select(i =>
                            "{\"id\":\"d" + i + "\",\"content\":\"c\",\"metadata_json\":\"{}\"}"));
                        return Json("{\"code\":0,\"data\":[" + items + "]}");
                    }
                    return Json("{\"code\":0,\"data\":[{\"id\":\"dZ\",\"content\":\"c\",\"metadata_json\":\"{}\"}]}");
                })
                .On("POST", "/collections/describe", () => Json(ExistingDescribeJson));

            var store = StoreWith(handler);
            var all = store.GetAll().ToList();

            Assert.Equal(1001, all.Count);
            Assert.Contains(all, d => d.Id == "dZ");
            Assert.Equal(2, handler.Requests.Count(r => r.Path.Contains("/entities/query")));
        }

        [Fact]
        public void Clear_DropsCollection_AndRecreates()
        {
            var handler = new MockHandler()
                .On("POST", "/collections/drop", () => Json("{\"code\":0,\"data\":{}}"))
                .On("POST", "/collections/create", () => Json("{\"code\":0,\"data\":{}}"))
                .On("POST", "/collections/describe", () => Json(ExistingDescribeJson));

            // Existing collection reports dim 3, so Clear recreates it.
            var store = StoreWith(handler);
            store.Clear();

            Assert.Equal(0, store.DocumentCount);
            Assert.Contains(handler.Requests, r => r.Path.Contains("/collections/drop"));
            Assert.Contains(handler.Requests, r => r.Path.Contains("/collections/create"));
        }

        [Fact]
        public void Constructor_UnsupportedMetric_Throws()
        {
            Assert.Throws<NotSupportedException>(() =>
                new MilvusDocumentStore<float>(Collection, BaseUrl, distanceMetric: DistanceMetricType.Jaccard,
                    httpClient: new HttpClient(new MockHandler()) { BaseAddress = new Uri(BaseUrl) }));
        }

        #region Integration (gated - require a live Milvus instance)

        [Trait("Category", "Integration")]
        [SkippableFact]
        public void Integration_UpsertSearchDelete_RoundTrips()
        {
            var url = Environment.GetEnvironmentVariable("MILVUS_URL");
            Skip.If(string.IsNullOrWhiteSpace(url),
                "Set MILVUS_URL (and optionally MILVUS_TOKEN) to run Milvus integration tests.");

            var token = Environment.GetEnvironmentVariable("MILVUS_TOKEN");
            var collection = "aidotnet_it_" + Guid.NewGuid().ToString("N");
            var store = new MilvusDocumentStore<float>(collection, url!, token,
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
