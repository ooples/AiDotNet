using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs;

/// <summary>An immutable program source treated as one evolvable candidate.</summary>
/// <remarks>
/// <para>
/// The genome carries the exact <see cref="Source"/> text the run will hand to an execution engine plus the
/// <see cref="Language"/> that decides comment markers, fence labels, and file extensions. <see cref="Id"/> is a
/// lowercase hexadecimal SHA-256 over <see cref="NormalizedSource"/> and <see cref="Language"/>, and normalization
/// strips a byte-order mark, rewrites CRLF and CR terminators as line feeds, trims trailing white space from every
/// line, and drops trailing blank lines. Two proposals that differ only in those incidental ways therefore share one
/// identity, which is exactly what <c>IEvolutionTask&lt;TGenome&gt;.CanonicalizeAsync</c> needs so the engine can
/// deduplicate them and reuse a cached evaluation instead of paying to run the same program twice.
/// </para>
/// <para>
/// Construction validates that the source is non-empty after normalization and no longer than
/// <see cref="MaxSourceLength"/> characters, so an unbounded model response cannot become an unbounded genome.
/// Identity and value equality cover exactly the same fields: <see cref="NormalizedSource"/> and
/// <see cref="Language"/>. Language belongs in both because it selects the interpreter, so byte-identical text in
/// two languages is two candidates that evaluate differently and must never share a cached result.
/// <see cref="Description"/> belongs in neither: it is the model's note about what a proposal changed, it cannot
/// alter how the program runs or scores, and putting it in equality while leaving it out of the identity is what
/// previously let two genomes be unequal yet share an <see cref="Id"/>. A run that genuinely wants a
/// description-carrying identity should fold it in at the task's canonicalization step instead.
/// </para>
/// <para><b>For Beginners:</b> This class is one candidate program in an evolutionary search: the code itself, the
/// language it is written in, and an optional note about what changed. The important part is <see cref="Id"/>, a
/// fingerprint computed from the code after cosmetic differences are removed. If a model reformats a file's line
/// endings but changes nothing else, the fingerprint stays the same and the search knows it has already tried that
/// program. Because the object never changes after construction, it is safe to keep in an archive, write to a
/// checkpoint, and share between threads.</para>
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
        Id = ComputeIdCore(normalized, language);
        _hashCode = ComputeHashCode(normalized, language);
    }

    /// <summary>Gets the source text exactly as supplied, including its original line endings.</summary>
    public string Source { get; }

    /// <summary>Gets the source after byte-order-mark removal, line-ending normalization, and trailing-space trimming.</summary>
    public string NormalizedSource { get; }

    /// <summary>Gets the language the source is written in.</summary>
    public ProgramLanguage Language { get; }

    /// <summary>Gets the optional bounded description of this candidate, or <c>null</c> when none was supplied.</summary>
    public string? Description { get; }

    /// <summary>Gets the lowercase hexadecimal SHA-256 over <see cref="NormalizedSource"/> and <see cref="Language"/>.</summary>
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
    /// <returns>The lowercase hexadecimal SHA-256 over the normalized text and the language.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="language"/> is not a defined value.</exception>
    public static string ComputeId(string source, ProgramLanguage language = ProgramLanguage.Generic)
    {
        Guard.NotNull(source);
        if (!Enum.IsDefined(typeof(ProgramLanguage), language)) throw new ArgumentOutOfRangeException(nameof(language));
        return ComputeIdCore(ProgramText.Normalize(source), language);
    }

    /// <inheritdoc/>
    public bool Equals(ProgramGenome? other)
    {
        if (other is null) return false;
        if (ReferenceEquals(this, other)) return true;
        return string.Equals(NormalizedSource, other.NormalizedSource, StringComparison.Ordinal)
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

    private static string ComputeIdCore(string normalizedSource, ProgramLanguage language) =>
        EvolutionHash.Combine(new[] { "program-genome-v1", language.ToString(), normalizedSource });

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
