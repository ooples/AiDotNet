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
    /// Tests for the real pgvector document store. The <see cref="PostgresVectorFilterBuilder"/> and
    /// <see cref="PgVectorMetric"/> logic is unit-tested without a server; the end-to-end store test is a
    /// SkippableFact that runs only when <c>POSTGRES_VECTOR_CONN</c> points at a live pgvector database.
    /// The Storage.Postgres project is net10-only (Npgsql TFMs), hence the net5+ gate.
    /// </summary>
    public class PostgresVectorDocumentStoreTests
    {
        private const string ConnEnv = "POSTGRES_VECTOR_CONN";

        [Fact]
        public void FilterBuilder_NoFilters_ReturnsEmpty()
        {
            var parameters = new Dictionary<string, object>();
            var sql = PostgresVectorFilterBuilder.Build(new Dictionary<string, object>(), parameters);
            Assert.Equal(string.Empty, sql);
            Assert.Empty(parameters);
        }

        [Fact]
        public void FilterBuilder_StringFilter_EmitsEquality()
        {
            var parameters = new Dictionary<string, object>();
            var sql = PostgresVectorFilterBuilder.Build(
                new Dictionary<string, object> { ["category"] = "science" }, parameters);

            Assert.Equal(" WHERE (metadata ->> @k0) = @v0", sql);
            Assert.Equal("category", parameters["k0"]);
            Assert.Equal("science", parameters["v0"]);
        }

        [Fact]
        public void FilterBuilder_BoolFilter_EmitsTextTrueFalse()
        {
            var parameters = new Dictionary<string, object>();
            var sql = PostgresVectorFilterBuilder.Build(
                new Dictionary<string, object> { ["published"] = true }, parameters);

            Assert.Equal(" WHERE (metadata ->> @k0) = @v0", sql);
            Assert.Equal("true", parameters["v0"]);
        }

        [Fact]
        public void FilterBuilder_NumericFilter_EmitsGteRange()
        {
            var parameters = new Dictionary<string, object>();
            var sql = PostgresVectorFilterBuilder.Build(
                new Dictionary<string, object> { ["year"] = 2020 }, parameters);

            Assert.Equal(" WHERE (metadata ->> @k0)::numeric >= @v0", sql);
            Assert.Equal(2020.0, (double)parameters["v0"]);
        }

        [Fact]
        public void FilterBuilder_CollectionFilter_EmitsAny()
        {
            var parameters = new Dictionary<string, object>();
            var sql = PostgresVectorFilterBuilder.Build(
                new Dictionary<string, object> { ["tag"] = new[] { "a", "b" } }, parameters);

            Assert.Equal(" WHERE (metadata ->> @k0) = ANY(@v0)", sql);
            Assert.Equal(new[] { "a", "b" }, (string[])parameters["v0"]);
        }

        [Fact]
        public void FilterBuilder_MultipleFilters_JoinedWithAnd()
        {
            var parameters = new Dictionary<string, object>();
            var sql = PostgresVectorFilterBuilder.Build(
                new Dictionary<string, object> { ["category"] = "science", ["year"] = 2020 }, parameters);

            Assert.Contains(" AND ", sql);
            Assert.Equal(4, parameters.Count);
        }

        [Theory]
        [InlineData(DistanceMetricType.Cosine, "<=>")]
        [InlineData(DistanceMetricType.Euclidean, "<->")]
        [InlineData(DistanceMetricType.Manhattan, "<+>")]
        public void Metric_Operator_MapsCorrectly(DistanceMetricType metric, string expected)
        {
            Assert.Equal(expected, PgVectorMetric.Operator(metric));
        }

        [Fact]
        public void Metric_Operator_UnsupportedThrows()
        {
            Assert.Throws<NotSupportedException>(() => PgVectorMetric.Operator(DistanceMetricType.Jaccard));
        }

        [Fact]
        public void Metric_ToSimilarity_Cosine_IsOneMinusDistance()
        {
            Assert.Equal(0.8, PgVectorMetric.ToSimilarity(DistanceMetricType.Cosine, 0.2), 6);
        }

        [Fact]
        public void Metric_ToSimilarity_Euclidean_IsInverseOfDistance()
        {
            Assert.Equal(0.5, PgVectorMetric.ToSimilarity(DistanceMetricType.Euclidean, 1.0), 6);
        }

        [SkippableFact]
        [Trait("Category", "Integration")]
        public void EndToEnd_RoundTripsDocumentsAndFilters()
        {
            var conn = Environment.GetEnvironmentVariable(ConnEnv);
            Skip.If(string.IsNullOrWhiteSpace(conn), $"Set {ConnEnv} to run the pgvector integration test.");

            var table = "vec_test_" + Guid.NewGuid().ToString("N");
            var store = new PostgresVectorDocumentStore<float>(conn!, table, vectorDimension: 3);
            try
            {
                store.Add(Doc("a", "alpha", new Vector<float>(new float[] { 1, 0, 0 }),
                    new Dictionary<string, object> { ["category"] = "science", ["year"] = 2021 }));
                store.AddBatch(new[]
                {
                    Doc("b", "beta", new Vector<float>(new float[] { 0, 1, 0 }),
                        new Dictionary<string, object> { ["category"] = "history", ["year"] = 2019 }),
                    Doc("c", "gamma", new Vector<float>(new float[] { 0.9f, 0.1f, 0 }),
                        new Dictionary<string, object> { ["category"] = "science", ["year"] = 2023 }),
                });

                Assert.Equal(3, store.DocumentCount);

                var top = store.GetSimilar(new Vector<float>(new float[] { 1, 0, 0 }), 2).ToList();
                Assert.Equal("a", top[0].Id);
                Assert.True(top[0].HasRelevanceScore);

                var filtered = store.GetSimilarWithFilters(
                    new Vector<float>(new float[] { 1, 0, 0 }), 5,
                    new Dictionary<string, object> { ["category"] = "science" }).ToList();
                Assert.All(filtered, d => Assert.Equal("science", d.Metadata["category"].ToString()));

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
