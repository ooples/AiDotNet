using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.Evolution.Programs;

/// <summary>The start and end comment markers that delimit the editable region of a program.</summary>
/// <remarks>
/// <para>
/// A program-evolution run normally rewrites only part of a file: the imports, the harness, and the entry point
/// the evaluator calls have to survive untouched, while one clearly delimited region is offered to the model. The
/// reference OpenEvolve implementation hard-codes the Python-style markers <c>#&#160;EVOLVE-BLOCK-START</c> and
/// <c>#&#160;EVOLVE-BLOCK-END</c>, which are syntax errors in every C-like language. This value type carries the
/// marker pair explicitly, with <see cref="ForLanguage"/> supplying the right comment syntax per
/// <see cref="ProgramLanguage"/>, so the same evolution pipeline works for Python, C#, SQL, and anything else.
/// </para>
/// <para><b>For Beginners:</b> Think of these two strings as the tape around the part of a file that may be
/// changed. They are ordinary comments, so the file still compiles and runs with the tape in place. The default
/// pair uses <c>#</c>, which is how Python writes a comment; C#, Java, and JavaScript need <c>//</c> instead, and
/// SQL needs <c>--</c>. Use <see cref="ForLanguage"/> and you get the correct pair automatically; supply your own
/// pair through the constructor when your file format needs something different.</para>
/// </remarks>
public readonly struct EvolveBlockMarkers : IEquatable<EvolveBlockMarkers>
{
    private readonly string? _start;
    private readonly string? _end;

    /// <summary>Initializes a marker pair.</summary>
    /// <param name="start">The text that opens an evolve block; a line containing it starts the region.</param>
    /// <param name="end">The text that closes an evolve block; a line containing it ends the region.</param>
    /// <exception cref="ArgumentNullException"><paramref name="start"/> or <paramref name="end"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">
    /// A marker is empty or white space, or the two markers are equal, which would make a block impossible to close.
    /// </exception>
    public EvolveBlockMarkers(string start, string end)
    {
        if (start is null) throw new ArgumentNullException(nameof(start));
        if (end is null) throw new ArgumentNullException(nameof(end));
        if (string.IsNullOrWhiteSpace(start))
            throw new ArgumentException("The start marker cannot be empty or white space.", nameof(start));
        if (string.IsNullOrWhiteSpace(end))
            throw new ArgumentException("The end marker cannot be empty or white space.", nameof(end));
        if (string.Equals(start, end, StringComparison.Ordinal))
            throw new ArgumentException("The start and end markers must differ.", nameof(end));

        _start = start;
        _end = end;
    }

    /// <summary>Gets the text that opens an evolve block.</summary>
    public string Start => _start ?? EvolveBlock.DefaultStartMarker;

    /// <summary>Gets the text that closes an evolve block.</summary>
    public string End => _end ?? EvolveBlock.DefaultEndMarker;

    /// <summary>Gets the hash-comment marker pair used by Python, Ruby, shell, and YAML style sources.</summary>
    public static EvolveBlockMarkers Hash => new(EvolveBlock.DefaultStartMarker, EvolveBlock.DefaultEndMarker);

    /// <summary>Gets the double-slash marker pair used by C, C++, C#, Java, JavaScript, TypeScript, Go, and Rust.</summary>
    public static EvolveBlockMarkers Slash => new(EvolveBlock.SlashStartMarker, EvolveBlock.SlashEndMarker);

    /// <summary>Gets the double-dash marker pair used by SQL.</summary>
    public static EvolveBlockMarkers DoubleDash => new(EvolveBlock.SqlStartMarker, EvolveBlock.SqlEndMarker);

    /// <summary>Returns the marker pair whose comment syntax is valid in <paramref name="language"/>.</summary>
    /// <param name="language">The language whose comment syntax the markers must respect.</param>
    /// <returns>
    /// <see cref="Slash"/> for the C-like languages, <see cref="DoubleDash"/> for SQL, and <see cref="Hash"/> for
    /// Python and for <see cref="ProgramLanguage.Generic"/>.
    /// </returns>
    public static EvolveBlockMarkers ForLanguage(ProgramLanguage language)
    {
        switch (language)
        {
            case ProgramLanguage.CSharp:
            case ProgramLanguage.Java:
            case ProgramLanguage.JavaScript:
            case ProgramLanguage.TypeScript:
            case ProgramLanguage.CPlusPlus:
            case ProgramLanguage.C:
            case ProgramLanguage.Go:
            case ProgramLanguage.Rust:
                return Slash;
            case ProgramLanguage.SQL:
                return DoubleDash;
            default:
                return Hash;
        }
    }

    /// <inheritdoc/>
    public bool Equals(EvolveBlockMarkers other) =>
        string.Equals(Start, other.Start, StringComparison.Ordinal)
        && string.Equals(End, other.End, StringComparison.Ordinal);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is EvolveBlockMarkers other && Equals(other);

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        unchecked
        {
            return (StringComparer.Ordinal.GetHashCode(Start) * 397) ^ StringComparer.Ordinal.GetHashCode(End);
        }
    }

    /// <summary>Returns the marker pair in a stable, culture-independent form.</summary>
    /// <returns>The start marker, a pipe, and the end marker.</returns>
    public override string ToString() => string.Concat(Start, "|", End);

    /// <summary>Determines whether two marker pairs are equal.</summary>
    /// <param name="left">The first pair.</param>
    /// <param name="right">The second pair.</param>
    /// <returns><c>true</c> when both markers match ordinally.</returns>
    public static bool operator ==(EvolveBlockMarkers left, EvolveBlockMarkers right) => left.Equals(right);

    /// <summary>Determines whether two marker pairs differ.</summary>
    /// <param name="left">The first pair.</param>
    /// <param name="right">The second pair.</param>
    /// <returns><c>true</c> when either marker differs.</returns>
    public static bool operator !=(EvolveBlockMarkers left, EvolveBlockMarkers right) => !left.Equals(right);
}
