#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Linq;

using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Models;

using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.DocumentStores
{
    /// <summary>
    /// Tests for the real RedisVL (Redis + RediSearch) document store. The
    /// <see cref="RedisVectorQueryBuilder"/> and <see cref="RedisVectorField"/> logic is unit-tested
    /// without a server; the end-to-end store test is a SkippableFact that runs only when
    /// <c>REDIS_CONN</c> points at a live Redis Stack (RediSearch) server. The Storage.Redis project is
    /// net10-only (StackExchange.Redis TFMs), hence the net5+ gate.
    /// </summary>
    public class RedisVLDocumentStoreTests
    {
        private const string ConnEnv = "REDIS_CONN";

        private static readonly Dictionary<string, RedisVectorFieldType> Declared = new()
        {
            ["category"] = RedisVectorFieldType.Tag,
            ["year"] = RedisVectorFieldType.Numeric,
        };

        [Fact]
        public void QueryBuilder_NoFilters_ReturnsWildcard()
        {
            var expr = RedisVectorQueryBuilder.BuildFilterExpression(
                new Dictionary<string, object>(), Declared, out var unpushed);
            Assert.Equal("*", expr);
            Assert.Empty(unpushed);
        }

        [Fact]
        public void QueryBuilder_TagField_EmitsTagMatch()
        {
            var expr = RedisVectorQueryBuilder.BuildFilterExpression(
                new Dictionary<string, object> { ["category"] = "science" }, Declared, out var unpushed);
            Assert.Equal("(@category:{science})", expr);
            Assert.Empty(unpushed);
        }

        [Fact]
        public void QueryBuilder_NumericField_EmitsGteRange()
        {
            var expr = RedisVectorQueryBuilder.BuildFilterExpression(
                new Dictionary<string, object> { ["year"] = 2020 }, Declared, out _);
            Assert.Equal("(@year:[2020 +inf])", expr);
        }

        [Fact]
        public void QueryBuilder_CollectionTag_EmitsOrList()
        {
            var expr = RedisVectorQueryBuilder.BuildFilterExpression(
                new Dictionary<string, object> { ["category"] = new[] { "science", "history" } }, Declared, out _);
            Assert.Equal("(@category:{science|history})", expr);
        }

        [Fact]
        public void QueryBuilder_UndeclaredKey_ReportedForPostFiltering()
        {
            var expr = RedisVectorQueryBuilder.BuildFilterExpression(
                new Dictionary<string, object> { ["author"] = "smith" }, Declared, out var unpushed);
            Assert.Equal("*", expr);
            Assert.Equal(new[] { "author" }, unpushed);
        }

        [Fact]
        public void QueryBuilder_MixedDeclaredAndUndeclared()
        {
            var expr = RedisVectorQueryBuilder.BuildFilterExpression(
                new Dictionary<string, object> { ["category"] = "science", ["author"] = "smith" }, Declared, out var unpushed);
            Assert.Equal("(@category:{science})", expr);
            Assert.Equal(new[] { "author" }, unpushed);
        }

        [Fact]
        public void EscapeTag_EscapesSeparators()
        {
            Assert.Equal("hello\\ world", RedisVectorQueryBuilder.EscapeTag("hello world"));
            Assert.Equal("a\\-b", RedisVectorQueryBuilder.EscapeTag("a-b"));
        }

        [Fact]
        public void RedisVectorField_InvalidName_Throws()
        {
            Assert.Throws<ArgumentException>(() => new RedisVectorField("bad name!", RedisVectorFieldType.Tag));
        }

        [SkippableFact]
        [Trait("Category", "Integration")]
        public void EndToEnd_RoundTripsDocumentsAndFilters()
        {
            var conn = Environment.GetEnvironmentVariable(ConnEnv);
            Skip.If(string.IsNullOrWhiteSpace(conn), $"Set {ConnEnv} to run the RedisVL integration test.");

            var index = "v_idx_" + Guid.NewGuid().ToString("N").Substring(0, 8);
            using var store = new RedisVLDocumentStore<float>(
                conn!, index, vectorDimension: 3, DistanceMetricType.Cosine,
                filterableFields: new[]
                {
                    new RedisVectorField("category", RedisVectorFieldType.Tag),
                    new RedisVectorField("year", RedisVectorFieldType.Numeric),
                });
            try
            {
                store.Add(Doc("a", "alpha", new Vector<float>(new float[] { 1, 0, 0 }),
                    new Dictionary<string, object> { ["category"] = "science", ["year"] = 2021, ["author"] = "smith" }));
                store.AddBatch(new[]
                {
                    Doc("b", "beta", new Vector<float>(new float[] { 0, 1, 0 }),
                        new Dictionary<string, object> { ["category"] = "history", ["year"] = 2019, ["author"] = "jones" }),
                    Doc("c", "gamma", new Vector<float>(new float[] { 0.9f, 0.1f, 0 }),
                        new Dictionary<string, object> { ["category"] = "science", ["year"] = 2023, ["author"] = "smith" }),
                });

                Assert.Equal(3, store.DocumentCount);

                var top = store.GetSimilar(new Vector<float>(new float[] { 1, 0, 0 }), 2).ToList();
                Assert.Equal("a", top[0].Id);
                Assert.True(top[0].HasRelevanceScore);

                // Declared tag filter pushed server-side.
                var science = store.GetSimilarWithFilters(
                    new Vector<float>(new float[] { 1, 0, 0 }), 5,
                    new Dictionary<string, object> { ["category"] = "science" }).ToList();
                Assert.All(science, d => Assert.Equal("science", d.Metadata["category"].ToString()));

                // Undeclared key honoured via in-memory post-filtering.
                var byAuthor = store.GetSimilarWithFilters(
                    new Vector<float>(new float[] { 1, 0, 0 }), 5,
                    new Dictionary<string, object> { ["author"] = "jones" }).ToList();
                Assert.All(byAuthor, d => Assert.Equal("jones", d.Metadata["author"].ToString()));

                Assert.Equal("beta", store.GetById("b")!.Content);
                Assert.True(store.Remove("b"));
                Assert.Equal(2, store.DocumentCount);
                Assert.Equal(2, store.GetAll().Count());
            }
            finally
            {
                store.Clear();
            }
        }

        private static VectorDocument<float> Doc(string id, string content, Vector<float> embedding, Dictionary<string, object> metadata)
        {
            return new VectorDocument<float>(new Document<float>(id, content, metadata), embedding);
        }
    }
}
#endif
