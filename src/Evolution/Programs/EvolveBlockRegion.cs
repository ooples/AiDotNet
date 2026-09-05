namespace AiDotNet.Evolution.Programs;

/// <summary>One matched evolve block, split into the text before it, the block body, and the text after it.</summary>
/// <remarks>
/// <para>
/// <see cref="Prefix"/> ends with the start-marker line and its terminator, <see cref="Suffix"/> begins at the
/// end-marker line, and <see cref="Body"/> is everything between them including its own trailing terminator when it
/// is non-empty. Concatenating the three parts therefore reproduces the original source byte for byte, and
/// <see cref="Rewrite"/> substitutes a new body while preserving every character outside the block: imports,
/// harness code, and the evaluator entry point cannot be lost by an over-eager rewrite.
/// </para>
/// <para><b>For Beginners:</b> Imagine a page with a paragraph highlighted for editing. This type hands you three
/// pieces: everything above the highlight, the highlighted paragraph itself, and everything below it. To produce
/// an edited page you only replace the middle piece and glue the three back together, which is what
/// <see cref="Rewrite"/> does for you. Splitting the file this way is what guarantees that changing the evolvable
/// region never disturbs the rest of the file.</para>
/// </remarks>
public readonly struct EvolveBlockRegion : IEquatable<EvolveBlockRegion>
{
    private readonly string? _prefix;
    private readonly string? _body;
    private readonly string? _suffix;
    private readonly string? _newLine;

    internal EvolveBlockRegion(string prefix, string body, string suffix, string newLine, int startLineIndex, int endLineIndex)
    {
        _prefix = prefix;
        _body = body;
        _suffix = suffix;
        _newLine = newLine;
        StartLineIndex = startLineIndex;
        EndLineIndex = endLineIndex;
    }

    /// <summary>Gets the source text up to and including the start-marker line and its terminator.</summary>
    public string Prefix => _prefix ?? string.Empty;

    /// <summary>Gets the block body, terminated by a line break unless it is empty.</summary>
    public string Body => _body ?? string.Empty;

    /// <summary>Gets the source text from the end-marker line to the end of the file.</summary>
    public string Suffix => _suffix ?? string.Empty;

    /// <summary>Gets the line terminator used by the source this region came from.</summary>
    public string NewLine => _newLine ?? "\n";

    /// <summary>Gets the zero-based index of the line holding the start marker.</summary>
    public int StartLineIndex { get; }

    /// <summary>Gets the zero-based index of the line holding the end marker.</summary>
    public int EndLineIndex { get; }

    /// <summary>Gets the number of body lines between the markers.</summary>
    public int BodyLineCount => Math.Max(0, EndLineIndex - StartLineIndex - 1);

    /// <summary>Gets whether this region carries no source text, which is the state of a default instance.</summary>
    public bool IsEmpty => _prefix is null && _body is null && _suffix is null;

    /// <summary>Rebuilds the whole source with <paramref name="replacement"/> in place of <see cref="Body"/>.</summary>
    /// <param name="replacement">
    /// The new block body. Line endings are converted to <see cref="NewLine"/> and a terminator is appended when
    /// the replacement does not already end with one, so the end marker always starts its own line.
    /// </param>
    /// <returns>The complete rewritten source.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="replacement"/> is <c>null</c>.</exception>
    public string Rewrite(string replacement)
    {
        if (replacement is null) throw new ArgumentNullException(nameof(replacement));
        if (replacement.Length == 0) return string.Concat(Prefix, Suffix);

        string newLine = NewLine;
        List<string> lines = ProgramText.SplitLines(replacement);
        bool trailing = ProgramText.EndsWithNewLine(replacement);
        if (trailing && lines.Count > 0 && lines[lines.Count - 1].Length == 0) lines.RemoveAt(lines.Count - 1);
        string body = ProgramText.JoinLines(lines, newLine, trailingNewLine: true);
        return string.Concat(Prefix, body, Suffix);
    }

    /// <summary>Reconstructs the original source this region was extracted from.</summary>
    /// <returns>The concatenation of <see cref="Prefix"/>, <see cref="Body"/>, and <see cref="Suffix"/>.</returns>
    public string ToSource() => string.Concat(Prefix, Body, Suffix);

    /// <inheritdoc/>
    public bool Equals(EvolveBlockRegion other) =>
        string.Equals(Prefix, other.Prefix, StringComparison.Ordinal)
        && string.Equals(Body, other.Body, StringComparison.Ordinal)
        && string.Equals(Suffix, other.Suffix, StringComparison.Ordinal)
        && StartLineIndex == other.StartLineIndex
        && EndLineIndex == other.EndLineIndex;

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is EvolveBlockRegion other && Equals(other);

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        unchecked
        {
            int hash = 17;
            hash = (hash * 31) + StringComparer.Ordinal.GetHashCode(Prefix);
            hash = (hash * 31) + StringComparer.Ordinal.GetHashCode(Body);
            hash = (hash * 31) + StringComparer.Ordinal.GetHashCode(Suffix);
            hash = (hash * 31) + StartLineIndex;
            hash = (hash * 31) + EndLineIndex;
            return hash;
        }
    }

    /// <summary>Returns the line span of this region without echoing any source text.</summary>
    /// <returns>A short diagnostic label naming the marker line indexes.</returns>
    public override string ToString() => string.Concat(
        "lines ", StartLineIndex.ToString(System.Globalization.CultureInfo.InvariantCulture),
        "-", EndLineIndex.ToString(System.Globalization.CultureInfo.InvariantCulture));

    /// <summary>Determines whether two regions describe the same split of the same source.</summary>
    /// <param name="left">The first region.</param>
    /// <param name="right">The second region.</param>
    /// <returns><c>true</c> when every part and both line indexes match.</returns>
    public static bool operator ==(EvolveBlockRegion left, EvolveBlockRegion right) => left.Equals(right);

    /// <summary>Determines whether two regions differ.</summary>
    /// <param name="left">The first region.</param>
    /// <param name="right">The second region.</param>
    /// <returns><c>true</c> when any part or line index differs.</returns>
    public static bool operator !=(EvolveBlockRegion left, EvolveBlockRegion right) => !left.Equals(right);
}
