using System;
using System.Collections.Generic;
using System.Linq;

using AiDotNet.RetrievalAugmentedGeneration.Filtering;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores;

/// <summary>
/// Translates an AiDotNet <see cref="MetadataFilter"/> expression tree into an Elasticsearch Query DSL
/// query clause (a <c>bool</c> query built from <c>term</c>/<c>terms</c>/<c>range</c>/<c>exists</c> leaf
/// queries). Extracted from <see cref="ElasticsearchDocumentStore{T}"/> so the query-generation logic can
/// be unit-tested without a live Elasticsearch cluster.
/// </summary>
/// <remarks>
/// <para>Translation rules (all metadata fields are addressed under the <c>metadata.</c> prefix):</para>
/// <list type="bullet">
///   <item><description><b>Eq</b> → <c>term</c>; <b>Ne</b> → <c>bool.must_not[term]</c>.</description></item>
///   <item><description><b>Gt/Gte/Lt/Lte</b> → <c>range</c> with <c>gt/gte/lt/lte</c>.</description></item>
///   <item><description><b>In</b> → <c>terms</c>; <b>Exists</b> → <c>exists</c>.</description></item>
///   <item><description><b>And</b> → <c>bool.must</c>; <b>Or</b> → <c>bool.should</c> (minimum_should_match 1); <b>Not</b> → <c>bool.must_not</c>.</description></item>
/// </list>
/// </remarks>
public static class ElasticsearchVectorFilterBuilder
{
    /// <summary>
    /// Builds the Elasticsearch query clause representing <paramref name="filter"/>. Returns a
    /// <c>match_all</c> query when <paramref name="filter"/> is <c>null</c>.
    /// </summary>
    public static object Build(MetadataFilter? filter)
    {
        if (filter == null)
            return new Dictionary<string, object> { ["match_all"] = new Dictionary<string, object>() };

        return BuildClause(filter);
    }

    private static object BuildClause(MetadataFilter filter)
    {
        switch (filter)
        {
            case ComparisonFilter comparison:
                return BuildComparison(comparison);
            case InFilter inFilter:
                return new Dictionary<string, object>
                {
                    ["terms"] = new Dictionary<string, object> { [Field(inFilter.Key)] = inFilter.Values.ToArray() }
                };
            case ExistsFilter existsFilter:
                return new Dictionary<string, object>
                {
                    ["exists"] = new Dictionary<string, object> { ["field"] = Field(existsFilter.Key) }
                };
            case NotFilter notFilter:
                return Bool("must_not", new[] { BuildClause(notFilter.Operand) });
            case LogicalFilter logical when logical.Operator == MetadataFilterOperator.And:
                return Bool("must", logical.Operands.Select(BuildClause).ToArray());
            case LogicalFilter logical:
            {
                var should = Bool("should", logical.Operands.Select(BuildClause).ToArray());
                ((Dictionary<string, object>)should["bool"])["minimum_should_match"] = 1;
                return should;
            }
            default:
                throw new NotSupportedException($"Unsupported metadata filter node: {filter.GetType().Name}");
        }
    }

    private static object BuildComparison(ComparisonFilter comparison)
    {
        switch (comparison.Operator)
        {
            case MetadataFilterOperator.Eq:
                return new Dictionary<string, object>
                {
                    ["term"] = new Dictionary<string, object> { [Field(comparison.Key)] = comparison.Value }
                };
            case MetadataFilterOperator.Ne:
                return Bool("must_not", new object[]
                {
                    new Dictionary<string, object>
                    {
                        ["term"] = new Dictionary<string, object> { [Field(comparison.Key)] = comparison.Value }
                    }
                });
            case MetadataFilterOperator.Gt:
                return Range(comparison.Key, "gt", comparison.Value);
            case MetadataFilterOperator.Gte:
                return Range(comparison.Key, "gte", comparison.Value);
            case MetadataFilterOperator.Lt:
                return Range(comparison.Key, "lt", comparison.Value);
            case MetadataFilterOperator.Lte:
                return Range(comparison.Key, "lte", comparison.Value);
            default:
                throw new NotSupportedException($"Unsupported comparison operator: {comparison.Operator}");
        }
    }

    private static object Range(string key, string op, object value)
    {
        return new Dictionary<string, object>
        {
            ["range"] = new Dictionary<string, object>
            {
                [Field(key)] = new Dictionary<string, object> { [op] = value }
            }
        };
    }

    private static Dictionary<string, object> Bool(string occurrence, object[] clauses)
    {
        return new Dictionary<string, object>
        {
            ["bool"] = new Dictionary<string, object> { [occurrence] = clauses }
        };
    }

    private static string Field(string key) => "metadata." + key;
}
