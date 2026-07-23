using System;
using System.Collections.Generic;

using AiDotNet.RetrievalAugmentedGeneration.Filtering;

using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration.Filtering
{
    /// <summary>
    /// Exhaustive unit tests for the in-memory <see cref="MetadataFilter"/> evaluator covering equality,
    /// inequality, ranges, set membership, existence and arbitrary AND/OR/NOT nesting.
    /// </summary>
    public class MetadataFilterTests
    {
        private static Dictionary<string, object> Meta(params (string Key, object Value)[] pairs)
        {
            var d = new Dictionary<string, object>();
            foreach (var (key, value) in pairs)
                d[key] = value;
            return d;
        }

        // ---------------- Eq ----------------

        [Fact]
        public void Eq_MatchesEqualString()
        {
            var f = MetadataFilter.Eq("category", "science");
            Assert.True(f.Matches(Meta(("category", "science"))));
            Assert.False(f.Matches(Meta(("category", "food"))));
        }

        [Fact]
        public void Eq_MissingKey_DoesNotMatch()
        {
            var f = MetadataFilter.Eq("category", "science");
            Assert.False(f.Matches(Meta(("other", "x"))));
        }

        [Fact]
        public void Eq_NumericCoercion_IntLongStringAllEqual()
        {
            var f = MetadataFilter.Eq("year", 2020);
            Assert.True(f.Matches(Meta(("year", 2020))));
            Assert.True(f.Matches(Meta(("year", 2020L))));
            Assert.True(f.Matches(Meta(("year", 2020.0))));
            Assert.True(f.Matches(Meta(("year", "2020"))));
            Assert.False(f.Matches(Meta(("year", 2021))));
        }

        [Fact]
        public void Eq_Bool_Matches()
        {
            var f = MetadataFilter.Eq("published", true);
            Assert.True(f.Matches(Meta(("published", true))));
            Assert.False(f.Matches(Meta(("published", false))));
        }

        // ---------------- Ne ----------------

        [Fact]
        public void Ne_MatchesWhenDifferent()
        {
            var f = MetadataFilter.Ne("category", "science");
            Assert.True(f.Matches(Meta(("category", "food"))));
            Assert.False(f.Matches(Meta(("category", "science"))));
        }

        [Fact]
        public void Ne_MissingKey_Matches()
        {
            // Missing field is "not equal" to the value.
            var f = MetadataFilter.Ne("category", "science");
            Assert.True(f.Matches(Meta(("other", "x"))));
        }

        [Fact]
        public void Ne_IsLogicalNegationOfEq()
        {
            var meta = Meta(("category", "food"));
            var eq = MetadataFilter.Eq("category", "science");
            var ne = MetadataFilter.Ne("category", "science");
            Assert.Equal(!eq.Matches(meta), ne.Matches(meta));
        }

        // ---------------- Ranges ----------------

        [Theory]
        [InlineData(2019, false)]
        [InlineData(2020, false)]
        [InlineData(2021, true)]
        public void Gt_Numeric(int year, bool expected)
        {
            var f = MetadataFilter.Gt("year", 2020);
            Assert.Equal(expected, f.Matches(Meta(("year", year))));
        }

        [Theory]
        [InlineData(2019, false)]
        [InlineData(2020, true)]
        [InlineData(2021, true)]
        public void Gte_Numeric(int year, bool expected)
        {
            var f = MetadataFilter.Gte("year", 2020);
            Assert.Equal(expected, f.Matches(Meta(("year", year))));
        }

        [Theory]
        [InlineData(2019, true)]
        [InlineData(2020, false)]
        public void Lt_Numeric(int year, bool expected)
        {
            var f = MetadataFilter.Lt("year", 2020);
            Assert.Equal(expected, f.Matches(Meta(("year", year))));
        }

        [Theory]
        [InlineData(2020, true)]
        [InlineData(2021, false)]
        public void Lte_Numeric(int year, bool expected)
        {
            var f = MetadataFilter.Lte("year", 2020);
            Assert.Equal(expected, f.Matches(Meta(("year", year))));
        }

        [Theory]
        [InlineData(2019, false)]
        [InlineData(2020, true)]
        [InlineData(2022, true)]
        [InlineData(2024, true)]
        [InlineData(2025, false)]
        public void Range_InclusiveBounds(int year, bool expected)
        {
            var f = MetadataFilter.Range("year", 2020, 2024);
            Assert.Equal(expected, f.Matches(Meta(("year", year))));
        }

        [Fact]
        public void Comparison_NonComparableValue_DoesNotMatch()
        {
            // Comparing a string field with a numeric threshold is not orderable numerically.
            var f = MetadataFilter.Gt("name", 5);
            Assert.False(f.Matches(Meta(("name", "abc"))));
        }

        // ---------------- In ----------------

        [Fact]
        public void In_MatchesAnyMember()
        {
            var f = MetadataFilter.In("author", new object[] { "A", "B", "C" });
            Assert.True(f.Matches(Meta(("author", "B"))));
            Assert.False(f.Matches(Meta(("author", "Z"))));
        }

        [Fact]
        public void In_NumericMembers()
        {
            var f = MetadataFilter.In("year", new object[] { 2020, 2021 });
            Assert.True(f.Matches(Meta(("year", 2021L))));
            Assert.False(f.Matches(Meta(("year", 2019))));
        }

        [Fact]
        public void In_MissingKey_DoesNotMatch()
        {
            var f = MetadataFilter.In("author", new object[] { "A" });
            Assert.False(f.Matches(Meta(("other", "A"))));
        }

        // ---------------- Exists ----------------

        [Fact]
        public void Exists_MatchesWhenPresent()
        {
            var f = MetadataFilter.Exists("author");
            Assert.True(f.Matches(Meta(("author", "A"))));
            Assert.False(f.Matches(Meta(("other", "x"))));
        }

        // ---------------- And / Or / Not ----------------

        [Fact]
        public void And_RequiresAllOperands()
        {
            var f = MetadataFilter.Eq("category", "science").And(MetadataFilter.Gte("year", 2020));
            Assert.True(f.Matches(Meta(("category", "science"), ("year", 2021))));
            Assert.False(f.Matches(Meta(("category", "science"), ("year", 2019))));
            Assert.False(f.Matches(Meta(("category", "food"), ("year", 2021))));
        }

        [Fact]
        public void Or_RequiresAnyOperand()
        {
            var f = MetadataFilter.Eq("category", "science").Or(MetadataFilter.Eq("category", "math"));
            Assert.True(f.Matches(Meta(("category", "science"))));
            Assert.True(f.Matches(Meta(("category", "math"))));
            Assert.False(f.Matches(Meta(("category", "food"))));
        }

        [Fact]
        public void Not_NegatesOperand()
        {
            var f = MetadataFilter.Eq("archived", true).Not();
            Assert.True(f.Matches(Meta(("archived", false))));
            Assert.True(f.Matches(Meta(("other", "x"))));
            Assert.False(f.Matches(Meta(("archived", true))));
        }

        [Fact]
        public void Nested_AndOrNot_EvaluatesCorrectly()
        {
            // category == "science" AND (year >= 2020 OR author in ["A","B"]) AND NOT archived
            var f = MetadataFilter.Eq("category", "science")
                .And(MetadataFilter.Gte("year", 2020).Or(MetadataFilter.In("author", new object[] { "A", "B" })))
                .And(MetadataFilter.Eq("archived", true).Not());

            Assert.True(f.Matches(Meta(("category", "science"), ("year", 2022), ("archived", false))));
            Assert.True(f.Matches(Meta(("category", "science"), ("year", 2000), ("author", "A"), ("archived", false))));
            Assert.False(f.Matches(Meta(("category", "science"), ("year", 2000), ("author", "Z"), ("archived", false))));
            Assert.False(f.Matches(Meta(("category", "science"), ("year", 2022), ("archived", true))));
            Assert.False(f.Matches(Meta(("category", "food"), ("year", 2022), ("archived", false))));
        }

        [Fact]
        public void Or_ThreeOperands_ViaParams()
        {
            var f = MetadataFilter.Or(
                MetadataFilter.Eq("c", "a"),
                MetadataFilter.Eq("c", "b"),
                MetadataFilter.Eq("c", "c"));
            Assert.True(f.Matches(Meta(("c", "c"))));
            Assert.False(f.Matches(Meta(("c", "d"))));
        }

        [Fact]
        public void NullMetadata_TreatedAsEmpty()
        {
            Assert.False(MetadataFilter.Eq("k", "v").Matches(null!));
            Assert.True(MetadataFilter.Ne("k", "v").Matches(null!));
            Assert.False(MetadataFilter.Exists("k").Matches(null!));
        }

        // ---------------- Factory guards ----------------

        [Fact]
        public void Factories_RejectEmptyKey()
        {
            Assert.Throws<ArgumentException>(() => MetadataFilter.Eq("", "v"));
            Assert.Throws<ArgumentException>(() => MetadataFilter.Exists(""));
            Assert.Throws<ArgumentException>(() => MetadataFilter.In("", new object[] { 1 }));
        }

        [Fact]
        public void Operator_PropertyReflectsNodeKind()
        {
            Assert.Equal(MetadataFilterOperator.Eq, MetadataFilter.Eq("k", 1).Operator);
            Assert.Equal(MetadataFilterOperator.In, MetadataFilter.In("k", new object[] { 1 }).Operator);
            Assert.Equal(MetadataFilterOperator.Exists, MetadataFilter.Exists("k").Operator);
            Assert.Equal(MetadataFilterOperator.And, MetadataFilter.Eq("k", 1).And(MetadataFilter.Eq("j", 2)).Operator);
            Assert.Equal(MetadataFilterOperator.Or, MetadataFilter.Eq("k", 1).Or(MetadataFilter.Eq("j", 2)).Operator);
            Assert.Equal(MetadataFilterOperator.Not, MetadataFilter.Eq("k", 1).Not().Operator);
        }
    }
}
