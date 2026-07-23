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
    /// Unit tests for the real Qdrant REST client. All HTTP traffic is intercepted by a mock
    /// <see cref="HttpMessageHandler"/> so these run in CI with no network access.
    /// </summary>
    public class QdrantDocumentStoreTests
    {
        private const string BaseUrl = "http://localhost:6333";
        private const string Collection = "test";

        // A collection that already exists with 3-d Cosine vectors and 0 points.
        private const string ExistingCollectionJson =
            "{\"result\":{\"status\":\"green\",\"points_count\":0," +
            "\"config\":{\"params\":{\"vectors\":{\"size\":3,\"distance\":\"Cosine\"}}}},\"status\":\"ok\"}";

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
            public string DefaultBody { get; set; } = "{\"result\":true,\"status\":\"ok\"}";

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

        private static QdrantDocumentStore<float> StoreWith(MockHandler handler,
            DistanceMetricType metric = DistanceMetricType.Cosine, int vectorDimension = 0)
        {
            var client = new HttpClient(handler) { BaseAddress = new Uri(BaseUrl) };
            return new QdrantDocumentStore<float>(Collection, BaseUrl, apiKey: "k",
                distanceMetric: metric, vectorDimension: vectorDimension, httpClient: client);
        }

        private static VectorDocument<float> Doc(string id, string content, float[] vector, Dictionary<string, object>? metadata = null)
        {
            var d = new Document<float>(id, content, metadata ?? new Dictionary<string, object>());
            return new VectorDocument<float> { Document = d, Embedding = new Vector<float>(vector) };
        }

        #endregion

        [Fact]
        public void Constructor_ReadsExistingCollection_SetsDimensionAndCount()
        {
            var handler = new MockHandler()
                .On("GET", "/collections/test", () => Json(
                    "{\"result\":{\"points_count\":5,\"config\":{\"params\":{\"vectors\":{\"size\":8,\"distance\":\"Cosine\"}}}}}"));

            var store = StoreWith(handler);

            Assert.Equal(8, store.VectorDimension);
            Assert.Equal(5, store.DocumentCount);
            Assert.Equal(Collection, store.CollectionName);
        }

        [Fact]
        public void Constructor_CollectionMissing_CreatesCollectionWithMappedDistance()
        {
            var handler = new MockHandler()
                .On("PUT", "/collections/test", () => Json("{\"result\":true,\"status\":\"ok\"}"))
                .On("GET", "/collections/test", () => Json("{}", HttpStatusCode.NotFound));

            var store = StoreWith(handler, metric: DistanceMetricType.Euclidean, vectorDimension: 4);

            Assert.Equal(4, store.VectorDimension);
            var create = handler.Requests.Single(r => r.Method == "PUT" && r.Path == "/collections/test");
            var body = JObject.Parse(create.Body);
            Assert.Equal(4, (int)body["vectors"]!["size"]!);
            Assert.Equal("Euclid", (string?)body["vectors"]!["distance"]);
        }

        [Fact]
        public void Add_UpsertsPoint_SendsCorrectRequest()
        {
            var handler = new MockHandler()
                .On("GET", "/collections/test", () => Json(ExistingCollectionJson));

            var store = StoreWith(handler);
            store.Add(Doc("doc1", "Hello world", new float[] { 1, 0, 0 },
                new Dictionary<string, object> { ["category"] = "science" }));

            Assert.Equal(1, store.DocumentCount);
            var upsert = handler.Requests.Single(r => r.Method == "PUT" && r.Path.StartsWith("/collections/test/points"));
            var body = JObject.Parse(upsert.Body);
            var point = body["points"]![0]!;
            Assert.Equal("doc1", (string?)point["payload"]!["_doc_id"]);
            Assert.Equal("Hello world", (string?)point["payload"]!["_content"]);
            Assert.Equal("science", (string?)point["payload"]!["metadata"]!["category"]);
            Assert.Equal(3, ((JArray)point["vector"]!).Count);
        }

        [Fact]
        public void AddBatch_SendsAllPointsInOneRequest()
        {
            var handler = new MockHandler()
                .On("GET", "/collections/test", () => Json(ExistingCollectionJson));

            var store = StoreWith(handler);
            store.AddBatch(new List<VectorDocument<float>>
            {
                Doc("d1", "c1", new float[] { 1, 0, 0 }),
                Doc("d2", "c2", new float[] { 0, 1, 0 }),
                Doc("d3", "c3", new float[] { 0, 0, 1 })
            });

            Assert.Equal(3, store.DocumentCount);
            var upsert = handler.Requests.Single(r => r.Method == "PUT" && r.Path.StartsWith("/collections/test/points"));
            var points = (JArray)JObject.Parse(upsert.Body)["points"]!;
            Assert.Equal(3, points.Count);
        }

        [Fact]
        public void GetSimilar_ParsesRankedHits_InOrder()
        {
            var searchResponse =
                "{\"result\":[" +
                "{\"id\":\"g1\",\"score\":0.98,\"payload\":{\"_doc_id\":\"doc1\",\"_content\":\"first\",\"metadata\":{\"category\":\"science\"}}}," +
                "{\"id\":\"g2\",\"score\":0.55,\"payload\":{\"_doc_id\":\"doc2\",\"_content\":\"second\",\"metadata\":{}}}" +
                "],\"status\":\"ok\"}";

            var handler = new MockHandler()
                .On("POST", "/points/search", () => Json(searchResponse))
                .On("GET", "/collections/test", () => Json(ExistingCollectionJson));

            var store = StoreWith(handler);
            var results = store.GetSimilar(new Vector<float>(new float[] { 1, 0, 0 }), topK: 2).ToList();

            Assert.Equal(2, results.Count);
            Assert.Equal("doc1", results[0].Id);
            Assert.Equal("first", results[0].Content);
            Assert.True(results[0].HasRelevanceScore);
            Assert.True(Convert.ToDouble(results[0].RelevanceScore) > Convert.ToDouble(results[1].RelevanceScore));

            var search = handler.Requests.Single(r => r.Path.Contains("/points/search"));
            var body = JObject.Parse(search.Body);
            Assert.Equal(2, (int)body["limit"]!);
            Assert.True((bool)body["with_payload"]!);
            Assert.Equal(3, ((JArray)body["vector"]!).Count);
        }

        [Fact]
        public void GetSimilarWithFilters_TranslatesEqualityAndRange()
        {
            var handler = new MockHandler()
                .On("POST", "/points/search", () => Json("{\"result\":[],\"status\":\"ok\"}"))
                .On("GET", "/collections/test", () => Json(ExistingCollectionJson));

            var store = StoreWith(handler);
            var filters = new Dictionary<string, object> { ["category"] = "science", ["year"] = 2020 };
            store.GetSimilarWithFilters(new Vector<float>(new float[] { 1, 0, 0 }), 10, filters).ToList();

            var search = handler.Requests.Single(r => r.Path.Contains("/points/search"));
            var must = (JArray)JObject.Parse(search.Body)["filter"]!["must"]!;

            var categoryCond = must.Single(c => (string?)c["key"] == "metadata.category");
            Assert.Equal("science", (string?)categoryCond["match"]!["value"]);

            var yearCond = must.Single(c => (string?)c["key"] == "metadata.year");
            Assert.Equal(2020.0, (double)yearCond["range"]!["gte"]!);
        }

        [Fact]
        public void GetById_Found_ReturnsDocument()
        {
            var handler = new MockHandler()
                .On("GET", "/collections/test/points/", () => Json(
                    "{\"result\":{\"id\":\"g1\",\"payload\":{\"_doc_id\":\"doc1\",\"_content\":\"hi\",\"metadata\":{\"k\":\"v\"}}},\"status\":\"ok\"}"))
                .On("GET", "/collections/test", () => Json(ExistingCollectionJson));

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
                .On("GET", "/collections/test/points/", () => Json("{}", HttpStatusCode.NotFound))
                .On("GET", "/collections/test", () => Json(ExistingCollectionJson));

            var store = StoreWith(handler);
            Assert.Null(store.GetById("missing"));
        }

        [Fact]
        public void Remove_SendsDelete_ReturnsTrue()
        {
            var handler = new MockHandler()
                .On("POST", "/points/delete", () => Json("{\"result\":{\"status\":\"completed\"},\"status\":\"ok\"}"))
                .On("GET", "/collections/test", () => Json(ExistingCollectionJson));

            var store = StoreWith(handler);
            var removed = store.Remove("doc1");

            Assert.True(removed);
            var delete = handler.Requests.Single(r => r.Path.Contains("/points/delete"));
            var points = (JArray)JObject.Parse(delete.Body)["points"]!;
            Assert.Single(points);
        }

        [Fact]
        public void GetAll_ScrollsAllPages()
        {
            var page = 0;
            var handler = new MockHandler()
                .On("POST", "/points/scroll", _ =>
                {
                    page++;
                    return page == 1
                        ? Json("{\"result\":{\"points\":[" +
                               "{\"id\":\"g1\",\"payload\":{\"_doc_id\":\"d1\",\"_content\":\"c1\",\"metadata\":{}}}," +
                               "{\"id\":\"g2\",\"payload\":{\"_doc_id\":\"d2\",\"_content\":\"c2\",\"metadata\":{}}}]," +
                               "\"next_page_offset\":\"g2\"},\"status\":\"ok\"}")
                        : Json("{\"result\":{\"points\":[" +
                               "{\"id\":\"g3\",\"payload\":{\"_doc_id\":\"d3\",\"_content\":\"c3\",\"metadata\":{}}}]," +
                               "\"next_page_offset\":null},\"status\":\"ok\"}");
                })
                .On("GET", "/collections/test", () => Json(ExistingCollectionJson));

            var store = StoreWith(handler);
            var all = store.GetAll().ToList();

            Assert.Equal(3, all.Count);
            Assert.Contains(all, d => d.Id == "d3");
            Assert.Equal(2, handler.Requests.Count(r => r.Path.Contains("/points/scroll")));
        }

        [Fact]
        public void Clear_DeletesCollection_AndRecreates()
        {
            var handler = new MockHandler()
                .On("DELETE", "/collections/test", () => Json("{\"result\":true,\"status\":\"ok\"}"))
                .On("PUT", "/collections/test", () => Json("{\"result\":true,\"status\":\"ok\"}"))
                .On("GET", "/collections/test", () => Json(ExistingCollectionJson));

            var store = StoreWith(handler);
            store.Clear();

            Assert.Equal(0, store.DocumentCount);
            Assert.Contains(handler.Requests, r => r.Method == "DELETE" && r.Path == "/collections/test");
            // Dimension known (3) so the collection is recreated.
            Assert.Contains(handler.Requests, r => r.Method == "PUT" && r.Path == "/collections/test");
        }

        [Fact]
        public void Constructor_UnsupportedMetric_Throws()
        {
            Assert.Throws<NotSupportedException>(() =>
                new QdrantDocumentStore<float>(Collection, BaseUrl, distanceMetric: DistanceMetricType.Jaccard,
                    httpClient: new HttpClient(new MockHandler()) { BaseAddress = new Uri(BaseUrl) }));
        }

        #region Integration (gated - require a live Qdrant instance)

        [Trait("Category", "Integration")]
        [SkippableFact]
        public void Integration_UpsertSearchDelete_RoundTrips()
        {
            var url = Environment.GetEnvironmentVariable("QDRANT_URL");
            Skip.If(string.IsNullOrWhiteSpace(url),
                "Set QDRANT_URL (and optionally QDRANT_API_KEY) to run Qdrant integration tests.");

            var apiKey = Environment.GetEnvironmentVariable("QDRANT_API_KEY");
            var collection = "aidotnet_it_" + Guid.NewGuid().ToString("N");
            var store = new QdrantDocumentStore<float>(collection, url!, apiKey,
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
