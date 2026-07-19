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
    /// Unit tests for the real Weaviate REST/GraphQL client. All HTTP traffic is intercepted by a mock
    /// <see cref="HttpMessageHandler"/> so these run in CI with no network access.
    /// </summary>
    public class WeaviateDocumentStoreTests
    {
        private const string BaseUrl = "http://localhost:8080";
        private const string Class = "Article";

        private const string ExistingSchemaJson =
            "{\"class\":\"Article\",\"vectorIndexConfig\":{\"distance\":\"cosine\"}}";

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

        private static WeaviateDocumentStore<float> StoreWith(MockHandler handler,
            DistanceMetricType metric = DistanceMetricType.Cosine)
        {
            var client = new HttpClient(handler) { BaseAddress = new Uri(BaseUrl) };
            return new WeaviateDocumentStore<float>(Class, BaseUrl, apiKey: "k", distanceMetric: metric, httpClient: client);
        }

        private static VectorDocument<float> Doc(string id, string content, float[] vector, Dictionary<string, object>? metadata = null)
        {
            var d = new Document<float>(id, content, metadata ?? new Dictionary<string, object>());
            return new VectorDocument<float> { Document = d, Embedding = new Vector<float>(vector) };
        }

        #endregion

        [Fact]
        public void Constructor_CreatesClass_WhenMissing_WithMappedDistance()
        {
            var handler = new MockHandler()
                .On("GET", "/v1/schema/Article", () => Json("{}", HttpStatusCode.NotFound))
                .On("POST", "/v1/schema", () => Json("{}"));

            var store = StoreWith(handler, metric: DistanceMetricType.Euclidean);

            Assert.Equal(Class, store.ClassName);
            var create = handler.Requests.Single(r => r.Method == "POST" && r.Path == "/v1/schema");
            var body = JObject.Parse(create.Body);
            Assert.Equal("Article", (string?)body["class"]);
            Assert.Equal("none", (string?)body["vectorizer"]);
            Assert.Equal("l2-squared", (string?)body["vectorIndexConfig"]!["distance"]);
        }

        [Fact]
        public void Add_CreatesObject_WithPropertiesAndVector()
        {
            var handler = new MockHandler()
                .On("GET", "/v1/schema/Article", () => Json(ExistingSchemaJson))
                .On("POST", "/v1/objects", () => Json("{}"));

            var store = StoreWith(handler);
            store.Add(Doc("doc1", "Hello world", new float[] { 1, 0, 0 },
                new Dictionary<string, object> { ["category"] = "science" }));

            Assert.Equal(1, store.DocumentCount);
            var upsert = handler.Requests.Single(r => r.Method == "POST" && r.Path == "/v1/objects");
            var body = JObject.Parse(upsert.Body);
            Assert.Equal("Article", (string?)body["class"]);
            Assert.Equal("doc1", (string?)body["properties"]!["docId"]);
            Assert.Equal("Hello world", (string?)body["properties"]!["content"]);
            Assert.Equal("science", (string?)body["properties"]!["m_category"]);
            Assert.Equal(3, ((JArray)body["vector"]!).Count);
        }

        [Fact]
        public void AddBatch_SendsAllObjectsInOneBatchRequest()
        {
            var handler = new MockHandler()
                .On("GET", "/v1/schema/Article", () => Json(ExistingSchemaJson))
                .On("POST", "/v1/batch/objects", () => Json("[]"));

            var store = StoreWith(handler);
            store.AddBatch(new List<VectorDocument<float>>
            {
                Doc("d1", "c1", new float[] { 1, 0, 0 }),
                Doc("d2", "c2", new float[] { 0, 1, 0 })
            });

            Assert.Equal(2, store.DocumentCount);
            var batch = handler.Requests.Single(r => r.Method == "POST" && r.Path.Contains("/v1/batch/objects"));
            var objs = (JArray)JObject.Parse(batch.Body)["objects"]!;
            Assert.Equal(2, objs.Count);
        }

        [Fact]
        public void GetSimilar_ParsesGraphQlHits_InOrder()
        {
            var searchResponse =
                "{\"data\":{\"Get\":{\"Article\":[" +
                "{\"docId\":\"doc1\",\"content\":\"first\",\"metadataJson\":\"{\\\"category\\\":\\\"science\\\"}\",\"_additional\":{\"certainty\":0.98,\"distance\":0.04}}," +
                "{\"docId\":\"doc2\",\"content\":\"second\",\"metadataJson\":\"{}\",\"_additional\":{\"certainty\":0.55,\"distance\":0.9}}" +
                "]}}}";

            var handler = new MockHandler()
                .On("POST", "/v1/graphql", () => Json(searchResponse))
                .On("GET", "/v1/schema/Article", () => Json(ExistingSchemaJson));

            var store = StoreWith(handler);
            var results = store.GetSimilar(new Vector<float>(new float[] { 1, 0, 0 }), topK: 2).ToList();

            Assert.Equal(2, results.Count);
            Assert.Equal("doc1", results[0].Id);
            Assert.Equal("first", results[0].Content);
            Assert.Equal("science", results[0].Metadata["category"].ToString());
            Assert.True(results[0].HasRelevanceScore);
            Assert.True(Convert.ToDouble(results[0].RelevanceScore) > Convert.ToDouble(results[1].RelevanceScore));

            var search = handler.Requests.Single(r => r.Path.Contains("/v1/graphql"));
            var query = (string)JObject.Parse(search.Body)["query"]!;
            Assert.Contains("nearVector", query);
            Assert.Contains("Get { Article", query);
            Assert.Contains("limit: 2", query);
        }

        [Fact]
        public void GetSimilarWithFilters_TranslatesEqualityAndRange()
        {
            var handler = new MockHandler()
                .On("POST", "/v1/graphql", () => Json("{\"data\":{\"Get\":{\"Article\":[]}}}"))
                .On("GET", "/v1/schema/Article", () => Json(ExistingSchemaJson));

            var store = StoreWith(handler);
            var filters = new Dictionary<string, object> { ["category"] = "science", ["year"] = 2020 };
            store.GetSimilarWithFilters(new Vector<float>(new float[] { 1, 0, 0 }), 10, filters).ToList();

            var search = handler.Requests.Single(r => r.Path.Contains("/v1/graphql"));
            var query = (string)JObject.Parse(search.Body)["query"]!;

            Assert.Contains("operator: And", query);
            Assert.Contains("[\"m_category\"]", query);
            Assert.Contains("operator: Equal", query);
            Assert.Contains("valueText: \"science\"", query);
            Assert.Contains("[\"m_year\"]", query);
            Assert.Contains("operator: GreaterThanEqual", query);
            Assert.Contains("valueNumber: 2020", query);
        }

        [Fact]
        public void GetSimilarWithFilters_TranslatesAnyOfCollection()
        {
            var handler = new MockHandler()
                .On("POST", "/v1/graphql", () => Json("{\"data\":{\"Get\":{\"Article\":[]}}}"))
                .On("GET", "/v1/schema/Article", () => Json(ExistingSchemaJson));

            var store = StoreWith(handler);
            var filters = new Dictionary<string, object> { ["tag"] = new List<string> { "a", "b" } };
            store.GetSimilarWithFilters(new Vector<float>(new float[] { 1, 0, 0 }), 5, filters).ToList();

            var query = (string)JObject.Parse(handler.Requests.Single(r => r.Path.Contains("/v1/graphql")).Body)["query"]!;
            Assert.Contains("operator: Or", query);
            Assert.Contains("valueText: \"a\"", query);
            Assert.Contains("valueText: \"b\"", query);
        }

        [Fact]
        public void GetById_Found_ReturnsDocument()
        {
            var handler = new MockHandler()
                .On("GET", "/v1/objects/Article/", () => Json(
                    "{\"id\":\"uuid\",\"properties\":{\"docId\":\"doc1\",\"content\":\"hi\",\"metadataJson\":\"{\\\"k\\\":\\\"v\\\"}\"}}"))
                .On("GET", "/v1/schema/Article", () => Json(ExistingSchemaJson));

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
                .On("GET", "/v1/objects/Article/", () => Json("{}", HttpStatusCode.NotFound))
                .On("GET", "/v1/schema/Article", () => Json(ExistingSchemaJson));

            var store = StoreWith(handler);
            Assert.Null(store.GetById("missing"));
        }

        [Fact]
        public void Remove_SendsDelete_ReturnsTrue()
        {
            var handler = new MockHandler()
                .On("DELETE", "/v1/objects/Article/", () => Json("{}", HttpStatusCode.NoContent))
                .On("GET", "/v1/schema/Article", () => Json(ExistingSchemaJson));

            var store = StoreWith(handler);
            Assert.True(store.Remove("doc1"));
            Assert.Contains(handler.Requests, r => r.Method == "DELETE" && r.Path.Contains("/v1/objects/Article/"));
        }

        [Fact]
        public void GetAll_CursorPagesUntilExhausted()
        {
            var call = 0;
            var handler = new MockHandler()
                .On("GET", "/v1/objects", _ =>
                {
                    call++;
                    // First page returns 100 objects (full page) to force a second request.
                    if (call == 1)
                    {
                        var items = string.Join(",", Enumerable.Range(0, 100).Select(i =>
                            "{\"id\":\"u" + i + "\",\"properties\":{\"docId\":\"d" + i + "\",\"content\":\"c\",\"metadataJson\":\"{}\"}}"));
                        return Json("{\"objects\":[" + items + "]}");
                    }
                    return Json("{\"objects\":[{\"id\":\"uZ\",\"properties\":{\"docId\":\"dZ\",\"content\":\"c\",\"metadataJson\":\"{}\"}}]}");
                })
                .On("GET", "/v1/schema/Article", () => Json(ExistingSchemaJson));

            var store = StoreWith(handler);
            var all = store.GetAll().ToList();

            Assert.Equal(101, all.Count);
            Assert.Contains(all, d => d.Id == "dZ");
            Assert.Equal(2, handler.Requests.Count(r => r.Method == "GET" && r.Path.StartsWith("/v1/objects")));
        }

        [Fact]
        public void Clear_DeletesSchema_AndRecreates()
        {
            var handler = new MockHandler()
                .On("DELETE", "/v1/schema/Article", () => Json("{}"))
                .On("POST", "/v1/schema", () => Json("{}"))
                .On("GET", "/v1/schema/Article", () => Json(ExistingSchemaJson));

            var store = StoreWith(handler);
            store.Clear();

            Assert.Equal(0, store.DocumentCount);
            Assert.Contains(handler.Requests, r => r.Method == "DELETE" && r.Path.Contains("/v1/schema/Article"));
            Assert.Contains(handler.Requests, r => r.Method == "POST" && r.Path == "/v1/schema");
        }

        [Fact]
        public void Constructor_UnsupportedMetric_Throws()
        {
            Assert.Throws<NotSupportedException>(() =>
                new WeaviateDocumentStore<float>(Class, BaseUrl, distanceMetric: DistanceMetricType.Mahalanobis,
                    httpClient: new HttpClient(new MockHandler()) { BaseAddress = new Uri(BaseUrl) }));
        }

        #region Integration (gated - require a live Weaviate instance)

        [Trait("Category", "Integration")]
        [SkippableFact]
        public void Integration_UpsertSearchDelete_RoundTrips()
        {
            var url = Environment.GetEnvironmentVariable("WEAVIATE_URL");
            Skip.If(string.IsNullOrWhiteSpace(url),
                "Set WEAVIATE_URL (and optionally WEAVIATE_API_KEY) to run Weaviate integration tests.");

            var apiKey = Environment.GetEnvironmentVariable("WEAVIATE_API_KEY");
            var className = "Aidotnet_it_" + Guid.NewGuid().ToString("N");
            var store = new WeaviateDocumentStore<float>(className, url!, apiKey,
                distanceMetric: DistanceMetricType.Cosine);

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
