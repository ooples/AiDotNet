using System;
using System.Text.RegularExpressions;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores;

/// <summary>
/// The RediSearch field type used to index a piece of document metadata so it can be filtered on.
/// </summary>
public enum RedisVectorFieldType
{
    /// <summary>A <c>TAG</c> field: exact-match filtering for strings and booleans (e.g. <c>@category:{science}</c>).</summary>
    Tag,

    /// <summary>A <c>NUMERIC</c> field: range filtering for numbers (e.g. <c>@year:[2020 +inf]</c>).</summary>
    Numeric,
}

/// <summary>
/// Declares a single metadata field that <see cref="RedisVLDocumentStore{T}"/> should index in its
/// RediSearch schema so that metadata filters on that key can be pushed down to the server.
/// </summary>
/// <remarks>
/// RediSearch on a HASH index only filters on fields declared at <c>FT.CREATE</c> time. Filters that
/// reference an undeclared key are still honoured, but by post-filtering the candidate set in memory
/// rather than in the server query.
/// </remarks>
public sealed class RedisVectorField
{
    private static readonly Regex NamePattern = new("^[A-Za-z_][A-Za-z0-9_]*$", RegexOptions.Compiled);

    /// <summary>Initializes a new field declaration.</summary>
    /// <param name="name">The metadata key / hash field name. Must be a plain identifier.</param>
    /// <param name="type">The RediSearch field type.</param>
    /// <exception cref="ArgumentException">Thrown when <paramref name="name"/> is not a valid identifier.</exception>
    public RedisVectorField(string name, RedisVectorFieldType type)
    {
        if (string.IsNullOrWhiteSpace(name) || !NamePattern.IsMatch(name))
            throw new ArgumentException(
                "Field name must be a valid identifier (letters, digits, underscore; not starting with a digit).",
                nameof(name));

        Name = name;
        Type = type;
    }

    /// <summary>Gets the metadata key / hash field name.</summary>
    public string Name { get; }

    /// <summary>Gets the RediSearch field type.</summary>
    public RedisVectorFieldType Type { get; }
}
