namespace AiDotNet.Enums;

/// <summary>Selects how a candidate program's captured output is compared with an expected output.</summary>
/// <remarks>
/// <para>
/// Execution-based scoring compares text, and text comparison is where most false failures come from: a trailing
/// newline written by a print statement, Windows line endings from a container, or an extra space between columns
/// will all sink an otherwise correct program. Choosing the comparison explicitly keeps the fitness signal honest,
/// because the strictness is part of the evaluator's version identity rather than an accident of the runner.
/// </para>
/// <para><b>For Beginners:</b> To decide whether a generated program is correct you run it on an example input and
/// check that what it printed matches what you expected. This enum is how strict that check should be. Use
/// <see cref="Ordinal"/> when every character matters, <see cref="TrimmedOrdinal"/> to forgive leading and trailing
/// blank space, and <see cref="NormalizedWhitespace"/> when only the words matter and any run of spaces, tabs, or
/// newlines is as good as a single space. Stricter settings give a cleaner signal; looser settings avoid punishing
/// a correct program for cosmetic differences.</para>
/// </remarks>
public enum ProgramOutputComparison
{
    /// <summary>Byte-for-byte comparison after line endings are normalized to line feeds.</summary>
    Ordinal = 0,

    /// <summary>Ordinal comparison after leading and trailing white space is removed from the whole output.</summary>
    TrimmedOrdinal = 1,

    /// <summary>Case-insensitive ordinal comparison after leading and trailing white space is removed.</summary>
    TrimmedOrdinalIgnoreCase = 2,

    /// <summary>Comparison after every run of white space collapses to a single space and the ends are trimmed.</summary>
    NormalizedWhitespace = 3
}
