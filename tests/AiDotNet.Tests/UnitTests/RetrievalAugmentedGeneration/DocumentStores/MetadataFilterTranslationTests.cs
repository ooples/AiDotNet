using System.Collections.Generic;

using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Filtering;

using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.DocumentStores
{
    /// <summary>
    /// Unit tests that assert each store translates a <see cref="MetadataFilter"/> expression tree into
    /// the correct provider-native filter (JSON / GraphQL / boolean-expr / OData / SQL). These exercise
    /// the translation logic directly with no live services.
    /// </summary>
    public class MetadataFilterTranslationTests
    {
        private static readonly MetadataFilter Sample =
            MetadataFilter.Eq("category", "science").And(MetadataFilter.Gte("year", 2020));

        // ------------------------------------------------------------------
        // Qdrant: must / should / must_not + match / range (nested filters).
        // ------------------------------------------------------------------

        [Fact]
        public void Qdrant_Eq_WrapsInMustMatch()
        {
            var json = JObject.Parse(JsonConvert.SerializeObject(
                QdrantDocumentStore<float>.TranslateFilter(MetadataFilter.Eq("category", "science"))));
            var cond = (JArray)json["must"]!;
            Assert.Equal("metadata.category", (string?)cond[0]!["key"]);
            Assert.Equal("science", (string?)cond[0]!["match"]!["value"]);
        }

        [Fact]
        public void Qdrant_AndOfEqAndRange()
        {
            var json = JObject.Parse(JsonConvert.SerializeObject(QdrantDocumentStore<float>.TranslateFilter(Sample)));
            var must = (JArray)json["must"]!;
            Assert.Equal("metadata.category", (string?)must[0]!["key"]);
            Assert.Equal(2020.0, (double)must[1]!["range"]!["gte"]!);
        }

        [Fact]
        public void Qdrant_Not_MapsToMustNot()
        {
            var json = JObject.Parse(JsonConvert.SerializeObject(
                QdrantDocumentStore<float>.TranslateFilter(MetadataFilter.Eq("archived", true).Not())));
            Assert.NotNull(json["must_not"]);
        }

        [Fact]
        public void Qdrant_Or_MapsToShould_And_In_UsesMatchAny_And_Exists_UsesIsEmpty()
        {
            var or = JObject.Parse(JsonConvert.SerializeObject(QdrantDocumentStore<float>.TranslateFilter(
                MetadataFilter.Eq("a", "x").Or(MetadataFilter.Eq("a", "y")))));
            Assert.NotNull(or["should"]);

            var inJson = JObject.Parse(JsonConvert.SerializeObject(QdrantDocumentStore<float>.TranslateFilter(
                MetadataFilter.In("author", new object[] { "A", "B" }))));
            var anyArr = (JArray)inJson["must"]![0]!["match"]!["any"]!;
            Assert.Equal(2, anyArr.Count);

            var exists = JObject.Parse(JsonConvert.SerializeObject(QdrantDocumentStore<float>.TranslateFilter(
                MetadataFilter.Exists("author"))));
            Assert.Equal("metadata.author", (string?)((JArray)exists["must_not"]!)[0]!["is_empty"]!["key"]);
        }

        // ------------------------------------------------------------------
        // Pinecone: $and/$or/$eq/$ne/$in/$gte/$lte; $not is pushed down.
        // ------------------------------------------------------------------

        [Fact]
        public void Pinecone_AndOfEqAndGte()
        {
            var json = JObject.Parse(JsonConvert.SerializeObject(PineconeDocumentStore<float>.TranslateFilter(Sample)));
            var and = (JArray)json["$and"]!;
            Assert.Equal("science", (string?)and[0]!["category"]!["$eq"]);
            Assert.Equal(2020.0, (double)and[1]!["year"]!["$gte"]!);
        }

        [Fact]
        public void Pinecone_NotEq_PushesToNe()
        {
            var json = JObject.Parse(JsonConvert.SerializeObject(
                PineconeDocumentStore<float>.TranslateFilter(MetadataFilter.Eq("category", "science").Not())));
            Assert.Equal("science", (string?)json["category"]!["$ne"]);
        }

        [Fact]
        public void Pinecone_NotIn_PushesToNin_And_NotAnd_UsesOr()
        {
            var nin = JObject.Parse(JsonConvert.SerializeObject(
                PineconeDocumentStore<float>.TranslateFilter(MetadataFilter.In("author", new object[] { "A", "B" }).Not())));
            Assert.Equal(2, ((JArray)nin["author"]!["$nin"]!).Count);

            var demorgan = JObject.Parse(JsonConvert.SerializeObject(
                PineconeDocumentStore<float>.TranslateFilter(Sample.Not())));
            Assert.NotNull(demorgan["$or"]);
        }

        [Fact]
        public void Pinecone_In_And_Exists()
        {
            var inJson = JObject.Parse(JsonConvert.SerializeObject(
                PineconeDocumentStore<float>.TranslateFilter(MetadataFilter.In("author", new object[] { "A", "B" }))));
            Assert.Equal(2, ((JArray)inJson["author"]!["$in"]!).Count);

            var exists = JObject.Parse(JsonConvert.SerializeObject(
                PineconeDocumentStore<float>.TranslateFilter(MetadataFilter.Exists("author"))));
            Assert.True((bool)exists["author"]!["$exists"]!);
        }

        // ------------------------------------------------------------------
        // Weaviate GraphQL where (negation pushed down; ContainsAny for In).
        // ------------------------------------------------------------------

        [Fact]
        public void Weaviate_And_Equal_And_GreaterThanEqual()
        {
            var where = WeaviateDocumentStore<float>.TranslateWhere(Sample);
            Assert.Contains("operator: And", where);
            Assert.Contains("path: [\"m_category\"]", where);
            Assert.Contains("operator: Equal", where);
            Assert.Contains("valueText: \"science\"", where);
            Assert.Contains("operator: GreaterThanEqual", where);
            Assert.Contains("valueNumber: 2020", where);
        }

        [Fact]
        public void Weaviate_NotEq_UsesNotEqual_And_In_UsesContainsAny_And_Exists_UsesIsNull()
        {
            Assert.Contains("operator: NotEqual",
                WeaviateDocumentStore<float>.TranslateWhere(MetadataFilter.Eq("category", "science").Not()));

            var inWhere = WeaviateDocumentStore<float>.TranslateWhere(MetadataFilter.In("author", new object[] { "A", "B" }));
            Assert.Contains("operator: ContainsAny", inWhere);
            Assert.Contains("valueText: [\"A\", \"B\"]", inWhere);

            var exists = WeaviateDocumentStore<float>.TranslateWhere(MetadataFilter.Exists("author"));
            Assert.Contains("operator: IsNull", exists);
            Assert.Contains("valueBoolean: false", exists);
        }

        // ------------------------------------------------------------------
        // Milvus boolean expression.
        // ------------------------------------------------------------------

        [Fact]
        public void Milvus_And_Comparisons()
        {
            var expr = MilvusDocumentStore<float>.TranslateFilter(Sample);
            Assert.Equal("(m_category == \"science\" and m_year >= 2020)", expr);
        }

        [Fact]
        public void Milvus_Not_In_Exists_Or()
        {
            Assert.Equal("not (m_archived == true)",
                MilvusDocumentStore<float>.TranslateFilter(MetadataFilter.Eq("archived", true).Not()));
            Assert.Equal("m_year in [2020, 2021]",
                MilvusDocumentStore<float>.TranslateFilter(MetadataFilter.In("year", new object[] { 2020, 2021 })));
            Assert.Equal("exists m_author",
                MilvusDocumentStore<float>.TranslateFilter(MetadataFilter.Exists("author")));
            Assert.Equal("(m_a == \"x\" or m_a == \"y\")",
                MilvusDocumentStore<float>.TranslateFilter(MetadataFilter.Eq("a", "x").Or(MetadataFilter.Eq("a", "y"))));
        }

        // ------------------------------------------------------------------
        // Azure AI Search OData $filter.
        // ------------------------------------------------------------------

        [Fact]
        public void Azure_And_EqAndGe()
        {
            Assert.Equal("(category eq 'science' and year ge 2020)",
                AzureSearchDocumentStore<float>.TranslateFilter(Sample));
        }

        [Fact]
        public void Azure_Ne_Not_In_Exists()
        {
            Assert.Equal("category ne 'science'",
                AzureSearchDocumentStore<float>.TranslateFilter(MetadataFilter.Ne("category", "science")));
            Assert.Equal("not (category eq 'science')",
                AzureSearchDocumentStore<float>.TranslateFilter(MetadataFilter.Eq("category", "science").Not()));
            Assert.Equal("search.in(author, 'A|B', '|')",
                AzureSearchDocumentStore<float>.TranslateFilter(MetadataFilter.In("author", new object[] { "A", "B" })));
            Assert.Equal("author ne null",
                AzureSearchDocumentStore<float>.TranslateFilter(MetadataFilter.Exists("author")));
        }

#if NET5_0_OR_GREATER
        // ------------------------------------------------------------------
        // pgvector: parameterised jsonb WHERE (net10-only Storage.Postgres).
        // ------------------------------------------------------------------

        [Fact]
        public void Postgres_And_EqAndGte_Parameterised()
        {
            var parameters = new Dictionary<string, object>();
            var sql = PostgresVectorFilterBuilder.Build(Sample, parameters);

            Assert.Equal(" WHERE ((metadata ->> @k0) = @v0 AND (metadata ->> @k1)::numeric >= @v1)", sql);
            Assert.Equal("category", parameters["k0"]);
            Assert.Equal("science", parameters["v0"]);
            Assert.Equal("year", parameters["k1"]);
            Assert.Equal(2020.0, parameters["v1"]);
        }

        [Fact]
        public void Postgres_Not_In_Exists()
        {
            var p1 = new Dictionary<string, object>();
            Assert.Equal(" WHERE (NOT ((metadata ->> @k0) = @v0))",
                PostgresVectorFilterBuilder.Build(MetadataFilter.Eq("c", "x").Not(), p1));

            var p2 = new Dictionary<string, object>();
            Assert.Equal(" WHERE (metadata ->> @k0) = ANY(@v0)",
                PostgresVectorFilterBuilder.Build(MetadataFilter.In("a", new object[] { "x", "y" }), p2));

            var p3 = new Dictionary<string, object>();
            Assert.Equal(" WHERE jsonb_exists(metadata, @k0)",
                PostgresVectorFilterBuilder.Build(MetadataFilter.Exists("a"), p3));
        }

        [Fact]
        public void Postgres_NullFilter_ReturnsEmpty()
        {
            Assert.Equal(string.Empty, PostgresVectorFilterBuilder.Build((MetadataFilter?)null, new Dictionary<string, object>()));
        }

        // ------------------------------------------------------------------
        // RediSearch expression (net10-only Storage.Redis).
        // ------------------------------------------------------------------

        [Fact]
        public void Redis_And_Tag_And_NumericRange_FullyPushed()
        {
            var fields = new Dictionary<string, RedisVectorFieldType>
            {
                ["category"] = RedisVectorFieldType.Tag,
                ["year"] = RedisVectorFieldType.Numeric,
            };
            var expr = RedisVectorQueryBuilder.BuildFilterExpression(Sample, fields, out var fullyPushed);
            Assert.True(fullyPushed);
            Assert.Equal("(@category:{science} @year:[2020 +inf])", expr);
        }

        [Fact]
        public void Redis_Or_Not_In()
        {
            var fields = new Dictionary<string, RedisVectorFieldType> { ["c"] = RedisVectorFieldType.Tag };

            var or = RedisVectorQueryBuilder.BuildFilterExpression(
                MetadataFilter.Eq("c", "x").Or(MetadataFilter.Eq("c", "y")), fields, out _);
            Assert.Equal("(@c:{x} | @c:{y})", or);

            var not = RedisVectorQueryBuilder.BuildFilterExpression(
                MetadataFilter.Eq("c", "x").Not(), fields, out _);
            Assert.Equal("-(@c:{x})", not);

            var inExpr = RedisVectorQueryBuilder.BuildFilterExpression(
                MetadataFilter.In("c", new object[] { "x", "y" }), fields, out _);
            Assert.Equal("@c:{x|y}", inExpr);
        }

        [Fact]
        public void Redis_UndeclaredField_Or_Exists_NotFullyPushed()
        {
            var fields = new Dictionary<string, RedisVectorFieldType> { ["c"] = RedisVectorFieldType.Tag };

            RedisVectorQueryBuilder.BuildFilterExpression(MetadataFilter.Eq("unknown", "x"), fields, out var pushed1);
            Assert.False(pushed1);

            RedisVectorQueryBuilder.BuildFilterExpression(MetadataFilter.Exists("c"), fields, out var pushed2);
            Assert.False(pushed2);
        }

        // ------------------------------------------------------------------
        // Elasticsearch bool query (net10 test reference to Storage.Elasticsearch).
        // ------------------------------------------------------------------

        [Fact]
        public void Elasticsearch_And_Term_And_Range()
        {
            var json = JObject.Parse(JsonConvert.SerializeObject(ElasticsearchVectorFilterBuilder.Build(Sample)));
            var must = (JArray)json["bool"]!["must"]!;
            Assert.Equal("science", (string?)must[0]!["term"]!["metadata.category"]);
            Assert.Equal(2020, (int)must[1]!["range"]!["metadata.year"]!["gte"]!);
        }

        [Fact]
        public void Elasticsearch_Not_Or_In_Exists()
        {
            var not = JObject.Parse(JsonConvert.SerializeObject(
                ElasticsearchVectorFilterBuilder.Build(MetadataFilter.Eq("c", "x").Not())));
            Assert.NotNull(not["bool"]!["must_not"]);

            var or = JObject.Parse(JsonConvert.SerializeObject(
                ElasticsearchVectorFilterBuilder.Build(MetadataFilter.Eq("c", "x").Or(MetadataFilter.Eq("c", "y")))));
            Assert.Equal(1, (int)or["bool"]!["minimum_should_match"]!);

            var inJson = JObject.Parse(JsonConvert.SerializeObject(
                ElasticsearchVectorFilterBuilder.Build(MetadataFilter.In("a", new object[] { "x", "y" }))));
            Assert.Equal(2, ((JArray)inJson["terms"]!["metadata.a"]!).Count);

            var exists = JObject.Parse(JsonConvert.SerializeObject(
                ElasticsearchVectorFilterBuilder.Build(MetadataFilter.Exists("a"))));
            Assert.Equal("metadata.a", (string?)exists["exists"]!["field"]);
        }
#endif
    }
}
