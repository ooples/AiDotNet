using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs;

/// <summary>An immutable program source treated as one evolvable candidate.</summary>
/// <remarks>
/// Identity and equality cover exact source text and language, excluding the optional description.
/// Display normalization is deliberately not semantic canonicalization: trimming whitespace or rewriting line
/// endings can change multiline string literals, preprocessing or language-specific behavior. Cosmetic-only edits
/// may therefore require another evaluation; falsely reusing a different program's score is not an acceptable tradeoff.
/// Sources must be nonblank, bounded and valid Unicode so UTF-8 serialization cannot collapse malformed surrogates.
/// <see cref="NormalizedSource"/> remains available for display and approximate descriptors, never execution/cache keys.
/// </remarks>
public sealed class ProgramGenome : IEquatable<ProgramGenome>
{
    /// <summary>The largest source length, in characters, that a genome may carry.</summary>
    public const int MaxSourceLength = 1_048_576;

    /// <summary>The largest description length, in characters, that a genome may carry.</summary>
    public const int MaxDescriptionLength = 4_096;

    private readonly int _hashCode;

    /// <summary>Initializes an immutable program genome.</summary>
    /// <param name="source">The program source text; must contain at least one non-white-space character.</param>
    /// <param name="language">The language the source is written in.</param>
    /// <param name="description">An optional bounded note describing this candidate, such as the change it makes.</param>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">
    /// <paramref name="source"/> is empty or white space, or <paramref name="description"/> exceeds
    /// <see cref="MaxDescriptionLength"/> characters.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="source"/> exceeds <see cref="MaxSourceLength"/> characters, or <paramref name="language"/> is
    /// not a defined enumeration value.
    /// </exception>
    public ProgramGenome(string source, ProgramLanguage language = ProgramLanguage.Generic, string? description = null)
    {
        Guard.NotNull(source);
        if (source.Length > MaxSourceLength)
            throw new ArgumentOutOfRangeException(nameof(source), source.Length,
                $"Program sources cannot exceed {MaxSourceLength} characters.");
        if (!Enum.IsDefined(typeof(ProgramLanguage), language)) throw new ArgumentOutOfRangeException(nameof(language));

        string normalized = ProgramText.Normalize(source);
        if (normalized.Length == 0)
            throw new ArgumentException("Program sources cannot be empty or white space.", nameof(source));
        if (description is not null && description.Length > MaxDescriptionLength)
            throw new ArgumentException(
                $"Program descriptions cannot exceed {MaxDescriptionLength} characters.", nameof(description));

        Source = source;
        NormalizedSource = normalized;
        Language = language;
        Description = description;
        ValidateSourceEncoding(source);
        Id = ComputeIdCore(source, language);
        _hashCode = ComputeHashCode(source, language);
    }

    /// <summary>Gets the source text exactly as supplied, including its original line endings.</summary>
    public string Source { get; }

    /// <summary>Gets the source after byte-order-mark removal, line-ending normalization, and trailing-space trimming.</summary>
    public string NormalizedSource { get; }

    /// <summary>Gets the language the source is written in.</summary>
    public ProgramLanguage Language { get; }

    /// <summary>Gets the optional bounded description of this candidate, or <c>null</c> when none was supplied.</summary>
    public string? Description { get; }

    /// <summary>Gets the versioned lowercase SHA-256 over exact <see cref="Source"/> and <see cref="Language"/>.</summary>
    /// <remarks>
    /// Two genomes share this value exactly when <see cref="Equals(ProgramGenome)"/> reports them equal, so the
    /// engine's duplicate set and evaluation cache can key on it safely. <see cref="Description"/> is excluded, so a
    /// description-only edit is the same candidate and is not evaluated twice.
    /// </remarks>
    public string Id { get; }

    /// <summary>Gets the number of lines in <see cref="NormalizedSource"/>.</summary>
    public int LineCount
    {
        get
        {
            int count = 1;
            foreach (char character in NormalizedSource)
            {
                if (character == '\n') count++;
            }

            return count;
        }
    }

    /// <summary>Returns a copy of this genome with a different source and the same language.</summary>
    /// <param name="source">The replacement source text.</param>
    /// <param name="description">An optional replacement description; <c>null</c> keeps the current one.</param>
    /// <returns>A new genome; this instance is unchanged.</returns>
    public ProgramGenome WithSource(string source, string? description = null) =>
        new(source, Language, description ?? Description);

    /// <summary>Returns a copy of this genome with a different description.</summary>
    /// <param name="description">The replacement description, or <c>null</c> to clear it.</param>
    /// <returns>A new genome; this instance is unchanged.</returns>
    public ProgramGenome WithDescription(string? description) => new(Source, Language, description);

    /// <summary>Normalizes source text the same way the constructor does, without building a genome.</summary>
    /// <param name="source">The text to normalize.</param>
    /// <returns>The normalized text.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is <c>null</c>.</exception>
    public static string Normalize(string source) => ProgramText.Normalize(source);

    /// <summary>Computes the identity a genome built from <paramref name="source"/> would have.</summary>
    /// <param name="source">The text to fingerprint.</param>
    /// <param name="language">
    /// The language the genome would carry; the default matches a genome constructed without one.
    /// </param>
    /// <returns>The versioned lowercase hexadecimal SHA-256 over exact source text and language.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="language"/> is not a defined value.</exception>
    public static string ComputeId(string source, ProgramLanguage language = ProgramLanguage.Generic)
    {
        Guard.NotNull(source);
        if (!Enum.IsDefined(typeof(ProgramLanguage), language)) throw new ArgumentOutOfRangeException(nameof(language));
        ValidateSourceEncoding(source);
        return ComputeIdCore(source, language);
    }

    /// <inheritdoc/>
    public bool Equals(ProgramGenome? other)
    {
        if (other is null) return false;
        if (ReferenceEquals(this, other)) return true;
        return string.Equals(Source, other.Source, StringComparison.Ordinal)
            && Language == other.Language;
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as ProgramGenome);

    /// <inheritdoc/>
    public override int GetHashCode() => _hashCode;

    /// <summary>Returns the identity and language, never the source text, so logs stay bounded.</summary>
    /// <returns>A short diagnostic label for this genome.</returns>
    public override string ToString() =>
        string.Concat(Id.Substring(0, 12), " (", Language.ToString(), ", ",
            NormalizedSource.Length.ToString(System.Globalization.CultureInfo.InvariantCulture), " chars)");

    /// <summary>Determines whether two genomes are value equal.</summary>
    /// <param name="left">The first genome, which may be <c>null</c>.</param>
    /// <param name="right">The second genome, which may be <c>null</c>.</param>
    /// <returns><c>true</c> when both are <c>null</c> or both describe the same candidate.</returns>
    public static bool operator ==(ProgramGenome? left, ProgramGenome? right) =>
        left is null ? right is null : left.Equals(right);

    /// <summary>Determines whether two genomes differ.</summary>
    /// <param name="left">The first genome, which may be <c>null</c>.</param>
    /// <param name="right">The second genome, which may be <c>null</c>.</param>
    /// <returns><c>true</c> when the genomes are not value equal.</returns>
    public static bool operator !=(ProgramGenome? left, ProgramGenome? right) => !(left == right);

    private static string ComputeIdCore(string source, ProgramLanguage language) =>
        EvolutionHash.Combine(new[] { "program-genome-v2-exact-source", language.ToString(), source });

    private static void ValidateSourceEncoding(string source)
    {
        for (int i = 0; i < source.Length; i++)
        {
            if (!char.IsSurrogate(source[i])) continue;
            if (!char.IsHighSurrogate(source[i]) || i + 1 >= source.Length || !char.IsLowSurrogate(source[i + 1]))
                throw new ArgumentException("Program source contains an unpaired Unicode surrogate.", nameof(source));
            i++;
        }
    }

    private static int ComputeHashCode(string normalizedSource, ProgramLanguage language)
    {
        unchecked
        {
            int hash = 17;
            hash = (hash * 31) + StringComparer.Ordinal.GetHashCode(normalizedSource);
            hash = (hash * 31) + (int)language;
            return hash;
        }
    }
}
