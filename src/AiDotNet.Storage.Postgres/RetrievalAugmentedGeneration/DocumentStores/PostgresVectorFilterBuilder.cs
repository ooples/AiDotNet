using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;

using AiDotNet.RetrievalAugmentedGeneration.Filtering;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores;

/// <summary>
/// Translates an AiDotNet metadata-filter dictionary into a parameterised PostgreSQL
/// <c>WHERE</c> clause over a <c>jsonb</c> <c>metadata</c> column. Extracted from
/// <see cref="PostgresVectorDocumentStore{T}"/> so the SQL-generation logic can be unit-tested
/// without a live database.
/// </summary>
/// <remarks>
/// <para>
/// Both the metadata keys and values are supplied as bound parameters (never string-interpolated),
/// so the generated SQL is injection-safe even for attacker-controlled metadata keys.
/// </para>
/// <para>Translation rules (mirroring the core <c>DocumentStoreBase.MatchesFilters</c> semantics):</para>
/// <list type="bullet">
///   <item><description><b>string / bool</b> → equality: <c>(metadata ->> @k) = @v</c> (bool compared as the text <c>true</c>/<c>false</c>).</description></item>
///   <item><description><b>numeric</b> → greater-than-or-equal range: <c>(metadata ->> @k)::numeric &gt;= @v</c>.</description></item>
///   <item><description><b>collection</b> → membership: <c>(metadata ->> @k) = ANY(@v)</c> (each element stringified).</description></item>
///   <item><description><b>anything else</b> → equality against its string form.</description></item>
/// </list>
/// </remarks>
public static class PostgresVectorFilterBuilder
{
    /// <summary>
    /// Builds the <c>WHERE</c> clause (including the leading <c>WHERE</c>, or an empty string when there
    /// are no filters) and appends the required parameters to <paramref name="parameters"/>.
    /// </summary>
    /// <param name="filters">The metadata filters; may be <c>null</c> or empty.</param>
    /// <param name="parameters">
    /// Receives the parameter name/value pairs to bind. Keys are the parameter names (e.g. <c>k0</c>,
    /// <c>v0</c>); values are the objects to bind.
    /// </param>
    /// <returns>The SQL <c>WHERE</c> fragment, or the empty string when no filters are supplied.</returns>
    public static string Build(IReadOnlyDictionary<string, object>? filters, IDictionary<string, object> parameters)
    {
        if (filters == null || filters.Count == 0)
        {
            return string.Empty;
        }

        var clauses = new List<string>();
        var index = 0;
        foreach (var filter in filters)
        {
            var keyParam = "k" + index.ToString(CultureInfo.InvariantCulture);
            var valParam = "v" + index.ToString(CultureInfo.InvariantCulture);
            parameters[keyParam] = filter.Key;

            var value = filter.Value;
            if (value is string s)
            {
                clauses.Add($"(metadata ->> @{keyParam}) = @{valParam}");
                parameters[valParam] = s;
            }
            else if (value is bool b)
            {
                clauses.Add($"(metadata ->> @{keyParam}) = @{valParam}");
                parameters[valParam] = b ? "true" : "false";
            }
            else if (IsNumeric(value))
            {
                clauses.Add($"(metadata ->> @{keyParam})::numeric >= @{valParam}");
                parameters[valParam] = System.Convert.ToDouble(value, CultureInfo.InvariantCulture);
            }
            else if (value is IEnumerable enumerable)
            {
                var items = enumerable.Cast<object?>()
                    .Select(o => o?.ToString() ?? string.Empty)
                    .ToArray();
                clauses.Add($"(metadata ->> @{keyParam}) = ANY(@{valParam})");
                parameters[valParam] = items;
            }
            else
            {
                clauses.Add($"(metadata ->> @{keyParam}) = @{valParam}");
                parameters[valParam] = value?.ToString() ?? string.Empty;
            }

            index++;
        }

        var sb = new StringBuilder(" WHERE ");
        sb.Append(string.Join(" AND ", clauses));
        return sb.ToString();
    }

    /// <summary>
    /// Builds a parameterised <c>WHERE</c> clause (including the leading <c>WHERE</c>) from a rich
    /// boolean <see cref="MetadataFilter"/> expression tree, appending the required bound parameters to
    /// <paramref name="parameters"/>. Returns the empty string when <paramref name="filter"/> is <c>null</c>.
    /// </summary>
    /// <remarks>
    /// Both metadata keys and values are supplied as bound parameters (never string-interpolated).
    /// Numeric comparisons cast the extracted text to <c>numeric</c>; existence uses
    /// <c>jsonb_exists(metadata, @k)</c>; <c>AND</c>/<c>OR</c>/<c>NOT</c> map to SQL boolean operators.
    /// </remarks>
    public static string Build(MetadataFilter? filter, IDictionary<string, object> parameters)
    {
        if (filter == null)
        {
            return string.Empty;
        }

        var counter = new int[1];
        var clause = BuildClause(filter, parameters, counter);
        return " WHERE " + clause;
    }

    private static string BuildClause(MetadataFilter filter, IDictionary<string, object> parameters, int[] counter)
    {
        switch (filter)
        {
            case ComparisonFilter comparison:
                return BuildComparison(comparison, parameters, counter);
            case InFilter inFilter:
            {
                var i = counter[0]++;
                var keyParam = "k" + i.ToString(CultureInfo.InvariantCulture);
                var valParam = "v" + i.ToString(CultureInfo.InvariantCulture);
                parameters[keyParam] = inFilter.Key;
                parameters[valParam] = inFilter.Values
                    .Select(o => o?.ToString() ?? string.Empty)
                    .ToArray();
                return $"(metadata ->> @{keyParam}) = ANY(@{valParam})";
            }
            case ExistsFilter existsFilter:
            {
                var i = counter[0]++;
                var keyParam = "k" + i.ToString(CultureInfo.InvariantCulture);
                parameters[keyParam] = existsFilter.Key;
                return $"jsonb_exists(metadata, @{keyParam})";
            }
            case NotFilter notFilter:
                return "(NOT (" + BuildClause(notFilter.Operand, parameters, counter) + "))";
            case LogicalFilter logical when logical.Operator == MetadataFilterOperator.And:
                return "(" + string.Join(" AND ", logical.Operands.Select(o => BuildClause(o, parameters, counter))) + ")";
            case LogicalFilter logical:
                return "(" + string.Join(" OR ", logical.Operands.Select(o => BuildClause(o, parameters, counter))) + ")";
            default:
                throw new NotSupportedException($"Unsupported metadata filter node: {filter.GetType().Name}");
        }
    }

    private static string BuildComparison(ComparisonFilter comparison, IDictionary<string, object> parameters, int[] counter)
    {
        var i = counter[0]++;
        var keyParam = "k" + i.ToString(CultureInfo.InvariantCulture);
        var valParam = "v" + i.ToString(CultureInfo.InvariantCulture);
        parameters[keyParam] = comparison.Key;

        var numeric = IsNumeric(comparison.Value);
        var left = numeric
            ? $"(metadata ->> @{keyParam})::numeric"
            : $"(metadata ->> @{keyParam})";

        if (numeric)
            parameters[valParam] = System.Convert.ToDouble(comparison.Value, CultureInfo.InvariantCulture);
        else if (comparison.Value is bool b)
            parameters[valParam] = b ? "true" : "false";
        else
            parameters[valParam] = comparison.Value?.ToString() ?? string.Empty;

        switch (comparison.Operator)
        {
            case MetadataFilterOperator.Eq:
                return $"{left} = @{valParam}";
            case MetadataFilterOperator.Ne:
                return $"{left} IS DISTINCT FROM @{valParam}";
            case MetadataFilterOperator.Gt:
                return $"{left} > @{valParam}";
            case MetadataFilterOperator.Gte:
                return $"{left} >= @{valParam}";
            case MetadataFilterOperator.Lt:
                return $"{left} < @{valParam}";
            case MetadataFilterOperator.Lte:
                return $"{left} <= @{valParam}";
            default:
                throw new NotSupportedException($"Unsupported comparison operator: {comparison.Operator}");
        }
    }

    private static bool IsNumeric(object? value)
    {
        return value is sbyte || value is byte || value is short || value is ushort
            || value is int || value is uint || value is long || value is ulong
            || value is float || value is double || value is decimal;
    }
}
