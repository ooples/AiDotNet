using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace AiDotNet.RetrievalAugmentedGeneration.Filtering;

/// <summary>
/// The kind of node in a <see cref="MetadataFilter"/> expression tree.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Every part of a filter is one of these operations. Leaf operations
/// (<c>Eq</c>, <c>Gt</c>, <c>In</c>, ...) test a single metadata field; the combinators
/// (<c>And</c>, <c>Or</c>, <c>Not</c>) glue smaller filters into bigger ones - exactly like the
/// filter languages of Pinecone, Qdrant and Weaviate.
/// </para>
/// </remarks>
public enum MetadataFilterOperator
{
    /// <summary>Field equals a value.</summary>
    Eq,

    /// <summary>Field does not equal a value (also matches when the field is absent).</summary>
    Ne,

    /// <summary>Field is strictly greater than a value.</summary>
    Gt,

    /// <summary>Field is greater than or equal to a value.</summary>
    Gte,

    /// <summary>Field is strictly less than a value.</summary>
    Lt,

    /// <summary>Field is less than or equal to a value.</summary>
    Lte,

    /// <summary>Field equals one of a set of values.</summary>
    In,

    /// <summary>Field is present in the metadata.</summary>
    Exists,

    /// <summary>Logical conjunction of child filters.</summary>
    And,

    /// <summary>Logical disjunction of child filters.</summary>
    Or,

    /// <summary>Logical negation of a child filter.</summary>
    Not,
}

/// <summary>
/// An immutable, provider-agnostic boolean metadata-filter expression tree. Supports equality,
/// inequality, full ranges, set membership, existence and arbitrary <c>AND</c>/<c>OR</c>/<c>NOT</c>
/// nesting, mirroring the filter capabilities of Pinecone, Qdrant, Weaviate, Milvus, Azure AI Search,
/// pgvector and Elasticsearch.
/// </summary>
/// <remarks>
/// <para>
/// Build filters with the static factory methods (<see cref="Eq(string, object)"/>,
/// <see cref="Range(string, object, object)"/>, <see cref="In(string, IEnumerable{object})"/>, ...) and
/// compose them with the fluent combinators (<see cref="And(MetadataFilter)"/>,
/// <see cref="Or(MetadataFilter)"/>, <see cref="Not()"/>). The resulting tree can be evaluated
/// in-memory with <see cref="Matches(IReadOnlyDictionary{string, object})"/> or translated to a
/// backend's native filter language by a document store.
/// </para>
/// <para><b>For Beginners:</b> This is a small, database-independent way to describe "which documents
/// do I want". For example:
/// <code>
/// // category == "science" AND year &gt;= 2020 AND (author in ["A","B"])
/// var filter = MetadataFilter.Eq("category", "science")
///     .And(MetadataFilter.Gte("year", 2020))
///     .And(MetadataFilter.In("author", new object[] { "A", "B" }));
/// </code>
/// The same <c>filter</c> object works against every store: an in-memory store evaluates it directly,
/// while Pinecone/Qdrant/etc. translate it into their own query syntax.
/// </para>
/// </remarks>
public abstract class MetadataFilter
{
    /// <summary>Gets the operator that identifies this node's kind.</summary>
    public abstract MetadataFilterOperator Operator { get; }

    /// <summary>
    /// Evaluates this filter against a metadata dictionary in memory.
    /// </summary>
    /// <param name="metadata">The document metadata to test; may be <c>null</c> (treated as empty).</param>
    /// <returns><c>true</c> when the metadata satisfies the filter; otherwise <c>false</c>.</returns>
    public abstract bool Matches(IReadOnlyDictionary<string, object> metadata);

    // ------------------------------------------------------------------
    // Leaf factories
    // ------------------------------------------------------------------

    /// <summary>Creates a filter that matches when <paramref name="key"/> equals <paramref name="value"/>.</summary>
    public static MetadataFilter Eq(string key, object value) => new ComparisonFilter(MetadataFilterOperator.Eq, key, value);

    /// <summary>Creates a filter that matches when <paramref name="key"/> is missing or does not equal <paramref name="value"/>.</summary>
    public static MetadataFilter Ne(string key, object value) => new ComparisonFilter(MetadataFilterOperator.Ne, key, value);

    /// <summary>Creates a filter that matches when <paramref name="key"/> is strictly greater than <paramref name="value"/>.</summary>
    public static MetadataFilter Gt(string key, object value) => new ComparisonFilter(MetadataFilterOperator.Gt, key, value);

    /// <summary>Creates a filter that matches when <paramref name="key"/> is greater than or equal to <paramref name="value"/>.</summary>
    public static MetadataFilter Gte(string key, object value) => new ComparisonFilter(MetadataFilterOperator.Gte, key, value);

    /// <summary>Creates a filter that matches when <paramref name="key"/> is strictly less than <paramref name="value"/>.</summary>
    public static MetadataFilter Lt(string key, object value) => new ComparisonFilter(MetadataFilterOperator.Lt, key, value);

    /// <summary>Creates a filter that matches when <paramref name="key"/> is less than or equal to <paramref name="value"/>.</summary>
    public static MetadataFilter Lte(string key, object value) => new ComparisonFilter(MetadataFilterOperator.Lte, key, value);

    /// <summary>
    /// Creates an inclusive range filter (<c>lower &lt;= key &lt;= upper</c>), the equivalent of
    /// <c>Gte(key, lower).And(Lte(key, upper))</c>.
    /// </summary>
    public static MetadataFilter Range(string key, object lower, object upper)
        => And(Gte(key, lower), Lte(key, upper));

    /// <summary>Creates a filter that matches when <paramref name="key"/> equals one of <paramref name="values"/>.</summary>
    public static MetadataFilter In(string key, IEnumerable<object> values) => new InFilter(key, values);

    /// <summary>Creates a filter that matches when <paramref name="key"/> equals one of <paramref name="values"/>.</summary>
    public static MetadataFilter In(string key, params object[] values) => new InFilter(key, values);

    /// <summary>Creates a filter that matches when <paramref name="key"/> is present in the metadata.</summary>
    public static MetadataFilter Exists(string key) => new ExistsFilter(key);

    // ------------------------------------------------------------------
    // Combinator factories
    // ------------------------------------------------------------------

    /// <summary>Combines two or more filters with logical <c>AND</c>.</summary>
    public static MetadataFilter And(params MetadataFilter[] filters) => new LogicalFilter(MetadataFilterOperator.And, filters);

    /// <summary>Combines two or more filters with logical <c>AND</c>.</summary>
    public static MetadataFilter And(IEnumerable<MetadataFilter> filters) => new LogicalFilter(MetadataFilterOperator.And, filters);

    /// <summary>Combines two or more filters with logical <c>OR</c>.</summary>
    public static MetadataFilter Or(params MetadataFilter[] filters) => new LogicalFilter(MetadataFilterOperator.Or, filters);

    /// <summary>Combines two or more filters with logical <c>OR</c>.</summary>
    public static MetadataFilter Or(IEnumerable<MetadataFilter> filters) => new LogicalFilter(MetadataFilterOperator.Or, filters);

    // ------------------------------------------------------------------
    // Fluent combinators
    // ------------------------------------------------------------------

    /// <summary>Returns a new filter that is the logical <c>AND</c> of this filter and <paramref name="other"/>.</summary>
    public MetadataFilter And(MetadataFilter other) => new LogicalFilter(MetadataFilterOperator.And, new[] { this, other });

    /// <summary>Returns a new filter that is the logical <c>OR</c> of this filter and <paramref name="other"/>.</summary>
    public MetadataFilter Or(MetadataFilter other) => new LogicalFilter(MetadataFilterOperator.Or, new[] { this, other });

    /// <summary>Returns a new filter that is the logical negation of this filter.</summary>
    public MetadataFilter Not() => new NotFilter(this);

    // ------------------------------------------------------------------
    // Shared evaluation helpers (used by the leaf nodes and reusable by
    // in-memory store implementations).
    // ------------------------------------------------------------------

    /// <summary>
    /// Compares two metadata values for equality using lenient, provider-like semantics: numbers are
    /// compared numerically (so <c>2020</c> and <c>2020L</c> and <c>"2020"</c> match), booleans by
    /// value, and everything else by ordinal string form.
    /// </summary>
    public static bool ValuesEqual(object? a, object? b)
    {
        if (a == null || b == null)
            return a == null && b == null;

        if (a is bool || b is bool)
            return BoolText(a) == BoolText(b);

        if (TryToDouble(a, out var da) && TryToDouble(b, out var db))
            return da.Equals(db);

        if (a.Equals(b))
            return true;

        return string.Equals(ToInvariantString(a), ToInvariantString(b), StringComparison.Ordinal);
    }

    /// <summary>
    /// Orders two metadata values, returning a negative/zero/positive result like
    /// <see cref="IComparable.CompareTo(object)"/>, or <c>null</c> when they are not comparable.
    /// Numbers are compared numerically; otherwise strings/other comparables are compared naturally.
    /// </summary>
    public static int? CompareValues(object? left, object? right)
    {
        if (left == null || right == null)
            return null;

        if (TryToDouble(left, out var dl) && TryToDouble(right, out var dr))
            return dl.CompareTo(dr);

        // Only order two strings against each other. A string vs a number (that didn't parse numerically
        // above) is NOT orderable — fall through to null so e.g. Gt("name", 5) never "matches" "abc".
        if (left is string ls && right is string rs)
            return string.CompareOrdinal(ls, rs);

        if (left.GetType() == right.GetType() && left is IComparable comparable)
            return comparable.CompareTo(right);

        return null;
    }

    private static string BoolText(object? value)
    {
        if (value is bool b)
            return b ? "true" : "false";
        return ToInvariantString(value).ToLowerInvariant();
    }

    /// <summary>
    /// Attempts to interpret a metadata value as a <see cref="double"/> (numeric types and numeric
    /// strings). Exposed so provider filter builders can share the same numeric-coercion rules.
    /// </summary>
    public static bool TryToDouble(object? value, out double result)
    {
        result = 0;
        switch (value)
        {
            case sbyte v: result = v; return true;
            case byte v: result = v; return true;
            case short v: result = v; return true;
            case ushort v: result = v; return true;
            case int v: result = v; return true;
            case uint v: result = v; return true;
            case long v: result = v; return true;
            case ulong v: result = v; return true;
            case float v: result = v; return true;
            case double v: result = v; return true;
            case decimal v: result = (double)v; return true;
            case string s:
                return double.TryParse(s, NumberStyles.Any, CultureInfo.InvariantCulture, out result);
            default:
                return false;
        }
    }

    /// <summary>
    /// Returns <c>true</c> when <paramref name="value"/> is a built-in numeric type. Exposed so provider
    /// filter builders can share the same numeric detection rules.
    /// </summary>
    public static bool IsNumeric(object? value)
    {
        return value is sbyte || value is byte || value is short || value is ushort
            || value is int || value is uint || value is long || value is ulong
            || value is float || value is double || value is decimal;
    }

    internal static string ToInvariantString(object? value)
    {
        if (value == null)
            return string.Empty;
        if (value is IFormattable formattable)
            return formattable.ToString(null, CultureInfo.InvariantCulture);
        return value.ToString() ?? string.Empty;
    }
}

/// <summary>
/// A leaf node comparing a single metadata field against a scalar value with one of the operators
/// <see cref="MetadataFilterOperator.Eq"/>, <see cref="MetadataFilterOperator.Ne"/>,
/// <see cref="MetadataFilterOperator.Gt"/>, <see cref="MetadataFilterOperator.Gte"/>,
/// <see cref="MetadataFilterOperator.Lt"/> or <see cref="MetadataFilterOperator.Lte"/>.
/// </summary>
public sealed class ComparisonFilter : MetadataFilter
{
    /// <summary>Initializes a new comparison node.</summary>
    /// <param name="op">The comparison operator.</param>
    /// <param name="key">The metadata field name.</param>
    /// <param name="value">The value to compare against.</param>
    public ComparisonFilter(MetadataFilterOperator op, string key, object value)
    {
        if (string.IsNullOrEmpty(key))
            throw new ArgumentException("Filter key cannot be null or empty", nameof(key));
        if (op != MetadataFilterOperator.Eq && op != MetadataFilterOperator.Ne
            && op != MetadataFilterOperator.Gt && op != MetadataFilterOperator.Gte
            && op != MetadataFilterOperator.Lt && op != MetadataFilterOperator.Lte)
        {
            throw new ArgumentException($"Operator '{op}' is not a comparison operator", nameof(op));
        }

        Operator = op;
        Key = key;
        Value = value;
    }

    /// <inheritdoc/>
    public override MetadataFilterOperator Operator { get; }

    /// <summary>Gets the metadata field name being compared.</summary>
    public string Key { get; }

    /// <summary>Gets the value being compared against.</summary>
    public object Value { get; }

    /// <inheritdoc/>
    public override bool Matches(IReadOnlyDictionary<string, object> metadata)
    {
        var present = metadata != null && metadata.TryGetValue(Key, out _);
        object? actual = present ? metadata![Key] : null;

        switch (Operator)
        {
            case MetadataFilterOperator.Eq:
                return present && ValuesEqual(actual, Value);
            case MetadataFilterOperator.Ne:
                return !(present && ValuesEqual(actual, Value));
            case MetadataFilterOperator.Gt:
                return present && CompareValues(actual, Value) is int gt && gt > 0;
            case MetadataFilterOperator.Gte:
                return present && CompareValues(actual, Value) is int gte && gte >= 0;
            case MetadataFilterOperator.Lt:
                return present && CompareValues(actual, Value) is int lt && lt < 0;
            case MetadataFilterOperator.Lte:
                return present && CompareValues(actual, Value) is int lte && lte <= 0;
            default:
                return false;
        }
    }
}

/// <summary>
/// A leaf node matching when a metadata field equals any value in a set (set membership).
/// </summary>
public sealed class InFilter : MetadataFilter
{
    /// <summary>Initializes a new set-membership node.</summary>
    /// <param name="key">The metadata field name.</param>
    /// <param name="values">The candidate values.</param>
    public InFilter(string key, IEnumerable<object> values)
    {
        if (string.IsNullOrEmpty(key))
            throw new ArgumentException("Filter key cannot be null or empty", nameof(key));
        if (values == null)
            throw new ArgumentNullException(nameof(values));

        Key = key;
        Values = values.ToList().AsReadOnly();
    }

    /// <inheritdoc/>
    public override MetadataFilterOperator Operator => MetadataFilterOperator.In;

    /// <summary>Gets the metadata field name being tested.</summary>
    public string Key { get; }

    /// <summary>Gets the candidate values.</summary>
    public IReadOnlyList<object> Values { get; }

    /// <inheritdoc/>
    public override bool Matches(IReadOnlyDictionary<string, object> metadata)
    {
        if (metadata == null || !metadata.TryGetValue(Key, out var actual))
            return false;

        foreach (var candidate in Values)
        {
            if (ValuesEqual(actual, candidate))
                return true;
        }

        return false;
    }
}

/// <summary>
/// A leaf node matching when a metadata field is present.
/// </summary>
public sealed class ExistsFilter : MetadataFilter
{
    /// <summary>Initializes a new existence node.</summary>
    /// <param name="key">The metadata field name that must be present.</param>
    public ExistsFilter(string key)
    {
        if (string.IsNullOrEmpty(key))
            throw new ArgumentException("Filter key cannot be null or empty", nameof(key));

        Key = key;
    }

    /// <inheritdoc/>
    public override MetadataFilterOperator Operator => MetadataFilterOperator.Exists;

    /// <summary>Gets the metadata field name that must be present.</summary>
    public string Key { get; }

    /// <inheritdoc/>
    public override bool Matches(IReadOnlyDictionary<string, object> metadata)
        => metadata != null && metadata.ContainsKey(Key);
}

/// <summary>
/// A combinator node joining child filters with logical <see cref="MetadataFilterOperator.And"/> or
/// <see cref="MetadataFilterOperator.Or"/>.
/// </summary>
public sealed class LogicalFilter : MetadataFilter
{
    /// <summary>Initializes a new logical combinator node.</summary>
    /// <param name="op">Either <see cref="MetadataFilterOperator.And"/> or <see cref="MetadataFilterOperator.Or"/>.</param>
    /// <param name="operands">The child filters; must contain at least one element.</param>
    public LogicalFilter(MetadataFilterOperator op, IEnumerable<MetadataFilter> operands)
    {
        if (op != MetadataFilterOperator.And && op != MetadataFilterOperator.Or)
            throw new ArgumentException($"Operator '{op}' is not a logical combinator", nameof(op));
        if (operands == null)
            throw new ArgumentNullException(nameof(operands));

        var list = operands.ToList();
        if (list.Count == 0)
            throw new ArgumentException("A logical filter requires at least one operand", nameof(operands));
        foreach (var operand in list)
        {
            if (operand == null)
                throw new ArgumentException("A logical filter operand cannot be null", nameof(operands));
        }

        Operator = op;
        Operands = list.AsReadOnly();
    }

    /// <inheritdoc/>
    public override MetadataFilterOperator Operator { get; }

    /// <summary>Gets the child filters.</summary>
    public IReadOnlyList<MetadataFilter> Operands { get; }

    /// <inheritdoc/>
    public override bool Matches(IReadOnlyDictionary<string, object> metadata)
    {
        if (Operator == MetadataFilterOperator.And)
        {
            foreach (var operand in Operands)
            {
                if (!operand.Matches(metadata))
                    return false;
            }

            return true;
        }

        foreach (var operand in Operands)
        {
            if (operand.Matches(metadata))
                return true;
        }

        return false;
    }
}

/// <summary>
/// A combinator node negating a single child filter.
/// </summary>
public sealed class NotFilter : MetadataFilter
{
    /// <summary>Initializes a new negation node.</summary>
    /// <param name="operand">The filter to negate.</param>
    public NotFilter(MetadataFilter operand)
    {
        Operand = operand ?? throw new ArgumentNullException(nameof(operand));
    }

    /// <inheritdoc/>
    public override MetadataFilterOperator Operator => MetadataFilterOperator.Not;

    /// <summary>Gets the negated child filter.</summary>
    public MetadataFilter Operand { get; }

    /// <inheritdoc/>
    public override bool Matches(IReadOnlyDictionary<string, object> metadata)
        => !Operand.Matches(metadata);
}
