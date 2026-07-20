using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;

using AiDotNet.RetrievalAugmentedGeneration.Filtering;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores;

/// <summary>
/// Builds the RediSearch pre-filter expression from an AiDotNet metadata-filter dictionary, given the
/// set of fields declared in the index. Extracted from <see cref="RedisVLDocumentStore{T}"/> so the
/// query-generation logic can be unit-tested without a live Redis server.
/// </summary>
/// <remarks>
/// <para>Translation rules (declared fields only; anything else is reported for in-memory post-filtering):</para>
/// <list type="bullet">
///   <item><description><b>TAG field</b> → exact match <c>@name:{value}</c>; a collection value becomes an OR list <c>@name:{a|b}</c>. Booleans render as <c>true</c>/<c>false</c>.</description></item>
///   <item><description><b>NUMERIC field</b> → greater-than-or-equal range <c>@name:[value +inf]</c> (mirrors the core <c>MatchesFilters</c> gte semantics).</description></item>
/// </list>
/// <para>Tag values are escaped for RediSearch's tag syntax so punctuation and spaces are matched literally.</para>
/// </remarks>
public static class RedisVectorQueryBuilder
{
    /// <summary>
    /// Builds the RediSearch pre-filter expression (the part before <c>=&gt;[KNN ...]</c>).
    /// </summary>
    /// <param name="filters">The metadata filters; may be <c>null</c> or empty.</param>
    /// <param name="declaredFields">The fields declared in the index, keyed by name.</param>
    /// <param name="unpushedKeys">
    /// Receives the filter keys that could not be pushed to the server (their key was not declared) and
    /// therefore must be applied by in-memory post-filtering.
    /// </param>
    /// <returns>The pre-filter expression, or <c>*</c> when nothing can be pushed down.</returns>
    public static string BuildFilterExpression(
        IReadOnlyDictionary<string, object>? filters,
        IReadOnlyDictionary<string, RedisVectorFieldType> declaredFields,
        out List<string> unpushedKeys)
    {
        unpushedKeys = new List<string>();
        if (filters == null || filters.Count == 0)
        {
            return "*";
        }

        var clauses = new List<string>();
        foreach (var filter in filters)
        {
            if (!declaredFields.TryGetValue(filter.Key, out var fieldType))
            {
                unpushedKeys.Add(filter.Key);
                continue;
            }

            if (fieldType == RedisVectorFieldType.Numeric)
            {
                var d = System.Convert.ToDouble(filter.Value, CultureInfo.InvariantCulture);
                clauses.Add($"@{filter.Key}:[{d.ToString("R", CultureInfo.InvariantCulture)} +inf]");
            }
            else
            {
                clauses.Add($"@{filter.Key}:{{{FormatTagValue(filter.Value)}}}");
            }
        }

        return clauses.Count > 0 ? "(" + string.Join(" ", clauses) + ")" : "*";
    }

    /// <summary>
    /// Builds a RediSearch pre-filter expression from a rich boolean <see cref="MetadataFilter"/> tree.
    /// The whole expression is pushed down only when every referenced field is declared and every
    /// operator is supported by RediSearch (TAG equality/membership, NUMERIC comparisons/ranges, and
    /// <c>AND</c>/<c>OR</c>/<c>NOT</c> composition). Otherwise <c>*</c> is returned with
    /// <paramref name="fullyPushed"/> set to <c>false</c>, signalling the caller to evaluate the entire
    /// filter in memory over an over-fetched candidate set.
    /// </summary>
    public static string BuildFilterExpression(
        MetadataFilter? filter,
        IReadOnlyDictionary<string, RedisVectorFieldType> declaredFields,
        out bool fullyPushed)
    {
        if (filter == null)
        {
            fullyPushed = true;
            return "*";
        }

        if (TryBuildExpression(filter, declaredFields, out var expr))
        {
            fullyPushed = true;
            return expr;
        }

        fullyPushed = false;
        return "*";
    }

    private static bool TryBuildExpression(
        MetadataFilter filter,
        IReadOnlyDictionary<string, RedisVectorFieldType> declaredFields,
        out string expr)
    {
        expr = string.Empty;
        switch (filter)
        {
            case ComparisonFilter comparison:
                return declaredFields.TryGetValue(comparison.Key, out var compType)
                    && TryBuildComparison(comparison, compType, out expr);
            case InFilter inFilter:
                return declaredFields.TryGetValue(inFilter.Key, out var inType)
                    && TryBuildIn(inFilter, inType, out expr);
            case ExistsFilter:
                // RediSearch has no portable "field exists" predicate for HASH indexes.
                return false;
            case NotFilter notFilter:
                if (!TryBuildExpression(notFilter.Operand, declaredFields, out var inner))
                    return false;
                expr = "-(" + inner + ")";
                return true;
            case LogicalFilter logical:
                var parts = new List<string>();
                foreach (var operand in logical.Operands)
                {
                    if (!TryBuildExpression(operand, declaredFields, out var part))
                        return false;
                    parts.Add(part);
                }
                var separator = logical.Operator == MetadataFilterOperator.And ? " " : " | ";
                expr = "(" + string.Join(separator, parts) + ")";
                return true;
            default:
                return false;
        }
    }

    private static bool TryBuildComparison(ComparisonFilter comparison, RedisVectorFieldType fieldType, out string expr)
    {
        expr = string.Empty;
        if (fieldType == RedisVectorFieldType.Numeric)
        {
            if (!MetadataFilter.TryToDouble(comparison.Value, out var d))
                return false;
            var v = d.ToString("R", CultureInfo.InvariantCulture);
            switch (comparison.Operator)
            {
                case MetadataFilterOperator.Eq: expr = $"@{comparison.Key}:[{v} {v}]"; return true;
                case MetadataFilterOperator.Ne: expr = $"-@{comparison.Key}:[{v} {v}]"; return true;
                case MetadataFilterOperator.Gt: expr = $"@{comparison.Key}:[({v} +inf]"; return true;
                case MetadataFilterOperator.Gte: expr = $"@{comparison.Key}:[{v} +inf]"; return true;
                case MetadataFilterOperator.Lt: expr = $"@{comparison.Key}:[-inf ({v}]"; return true;
                case MetadataFilterOperator.Lte: expr = $"@{comparison.Key}:[-inf {v}]"; return true;
                default: return false;
            }
        }

        // TAG field: only exact equality / inequality are expressible.
        switch (comparison.Operator)
        {
            case MetadataFilterOperator.Eq: expr = $"@{comparison.Key}:{{{FormatTagValue(comparison.Value)}}}"; return true;
            case MetadataFilterOperator.Ne: expr = $"-@{comparison.Key}:{{{FormatTagValue(comparison.Value)}}}"; return true;
            default: return false;
        }
    }

    private static bool TryBuildIn(InFilter inFilter, RedisVectorFieldType fieldType, out string expr)
    {
        expr = string.Empty;
        if (inFilter.Values.Count == 0)
            return false;

        if (fieldType == RedisVectorFieldType.Numeric)
        {
            var ranges = new List<string>();
            foreach (var value in inFilter.Values)
            {
                if (!MetadataFilter.TryToDouble(value, out var d))
                    return false;
                var v = d.ToString("R", CultureInfo.InvariantCulture);
                ranges.Add($"@{inFilter.Key}:[{v} {v}]");
            }
            expr = "(" + string.Join(" | ", ranges) + ")";
            return true;
        }

        var items = inFilter.Values.Select(FormatTagValue);
        expr = $"@{inFilter.Key}:{{{string.Join("|", items)}}}";
        return true;
    }

    private static string FormatTagValue(object? value)
    {
        if (value is string s)
            return EscapeTag(s);

        if (value is bool b)
            return b ? "true" : "false";

        if (value is IEnumerable enumerable && value is not string)
        {
            var items = enumerable.Cast<object?>()
                .Select(o => o is bool eb ? (eb ? "true" : "false") : EscapeTag(o?.ToString() ?? string.Empty));
            return string.Join("|", items);
        }

        return EscapeTag(value?.ToString() ?? string.Empty);
    }

    /// <summary>
    /// Escapes a value for use inside a RediSearch <c>TAG</c> query (<c>{...}</c>). RediSearch treats a
    /// range of punctuation characters as tokenizer separators; each is prefixed with a backslash so the
    /// value matches literally.
    /// </summary>
    public static string EscapeTag(string value)
    {
        const string special = ",.<>{}[]\"':;!@#$%^&*()-+=~| ";
        var sb = new StringBuilder(value.Length);
        foreach (var c in value)
        {
            if (special.IndexOf(c) >= 0)
                sb.Append('\\');
            sb.Append(c);
        }

        return sb.ToString();
    }
}
