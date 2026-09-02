using System.Globalization;
using AiDotNet.Enums;
using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.Evolution.Programs;

/// <summary>Finds, extracts, and rewrites the marked editable region of a program source.</summary>
/// <remarks>
/// <para>
/// Program evolution keeps a file runnable by confining edits to a region fenced with comment markers, so imports,
/// fixtures, and the entry point an evaluator calls survive every generation. This class implements that fence:
/// <see cref="Extract(string, EvolveBlockMarkers)"/> splits the source around each marker pair,
/// <see cref="TryReplaceFirst"/> and <see cref="TryReplaceAll"/> substitute new bodies while preserving every
/// character outside the blocks, and <see cref="Wrap"/> adds markers to a file that has none, matching the way the
/// reference implementation prepares a bare code string.
/// </para>
/// <para>
/// Detection matches the reference implementation's substring test, so an indented marker or a marker followed by
/// a trailing comment is still recognized, and CRLF, CR, and LF sources are handled identically. Every method is
/// deterministic, performs no I/O, and reports malformed input through
/// <see cref="EvolveBlockExtractionResult.Status"/> or a <c>false</c> return value rather than by throwing;
/// exceptions are reserved for <c>null</c> arguments.
/// </para>
/// <para><b>For Beginners:</b> Suppose you want a language model to improve one function in a file without
/// touching the rest. You put a start comment above it and an end comment below it, and this class does the rest:
/// it finds those comments, hands you the code between them, and glues a new version back in place. If the markers
/// are missing or broken it tells you so instead of quietly mangling the file. Use
/// <see cref="EvolveBlockMarkers.ForLanguage"/> to get comment markers that are valid in your language.</para>
/// </remarks>
public static class EvolveBlock
{
    /// <summary>The default hash-comment start marker, matching the reference OpenEvolve implementation.</summary>
    public const string DefaultStartMarker = "# EVOLVE-BLOCK-START";

    /// <summary>The default hash-comment end marker, matching the reference OpenEvolve implementation.</summary>
    public const string DefaultEndMarker = "# EVOLVE-BLOCK-END";

    /// <summary>The double-slash start marker for C-like languages.</summary>
    public const string SlashStartMarker = "// EVOLVE-BLOCK-START";

    /// <summary>The double-slash end marker for C-like languages.</summary>
    public const string SlashEndMarker = "// EVOLVE-BLOCK-END";

    /// <summary>The double-dash start marker for SQL.</summary>
    public const string SqlStartMarker = "-- EVOLVE-BLOCK-START";

    /// <summary>The double-dash end marker for SQL.</summary>
    public const string SqlEndMarker = "-- EVOLVE-BLOCK-END";

    /// <summary>Scans <paramref name="source"/> for marker pairs and splits it around each one.</summary>
    /// <param name="source">The program source to scan.</param>
    /// <param name="markers">The marker pair to look for; the default instance uses the hash-comment markers.</param>
    /// <returns>The recovered regions plus a status and diagnostics describing any malformed markers.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is <c>null</c>.</exception>
    public static EvolveBlockExtractionResult Extract(string source, EvolveBlockMarkers markers = default)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));

        List<string> lines = ProgramText.SplitLines(source);
        string newLine = ProgramText.DetectNewLine(source);
        bool trailingNewLine = ProgramText.EndsWithNewLine(source);
        if (trailingNewLine && lines.Count > 0 && lines[lines.Count - 1].Length == 0) lines.RemoveAt(lines.Count - 1);

        string startMarker = markers.Start;
        string endMarker = markers.End;
        var regions = new List<EvolveBlockRegion>();
        var diagnostics = new List<string>();
        bool sawStart = false;
        bool unterminated = false;
        bool restarted = false;
        bool unmatchedEnd = false;
        int openLine = -1;

        for (int index = 0; index < lines.Count; index++)
        {
            string line = lines[index];
            if (line.IndexOf(startMarker, StringComparison.Ordinal) >= 0)
            {
                if (openLine >= 0)
                {
                    restarted = true;
                    AddDiagnostic(diagnostics,
                        "A second start marker on line " + Ordinal(index) +
                        " reopened the block started on line " + Ordinal(openLine) + "; the partial block was discarded.");
                }

                sawStart = true;
                openLine = index;
                continue;
            }

            if (openLine >= 0 && line.IndexOf(endMarker, StringComparison.Ordinal) >= 0)
            {
                regions.Add(BuildRegion(lines, newLine, trailingNewLine, openLine, index));
                openLine = -1;
                continue;
            }

            if (openLine < 0 && line.IndexOf(endMarker, StringComparison.Ordinal) >= 0)
            {
                unmatchedEnd = true;
                AddDiagnostic(diagnostics,
                    "An end marker on line " + Ordinal(index) + " has no matching start marker and was ignored.");
            }
        }

        if (openLine >= 0)
        {
            unterminated = true;
            AddDiagnostic(diagnostics,
                "The start marker on line " + Ordinal(openLine) + " was never closed; the trailing block was discarded.");
        }

        EvolveBlockStatus status;
        if (unterminated) status = EvolveBlockStatus.Unterminated;
        else if (restarted) status = EvolveBlockStatus.RestartedBlock;
        else if (unmatchedEnd) status = EvolveBlockStatus.UnmatchedEnd;
        else status = sawStart ? EvolveBlockStatus.Complete : EvolveBlockStatus.NotPresent;

        return new EvolveBlockExtractionResult(status, regions, diagnostics);
    }

    /// <summary>Scans <paramref name="source"/> using the comment markers that are valid in a language.</summary>
    /// <param name="source">The program source to scan.</param>
    /// <param name="language">The language whose comment syntax selects the marker pair.</param>
    /// <returns>The recovered regions plus a status and diagnostics describing any malformed markers.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is <c>null</c>.</exception>
    public static EvolveBlockExtractionResult Extract(string source, ProgramLanguage language) =>
        Extract(source, EvolveBlockMarkers.ForLanguage(language));

    /// <summary>Reports whether <paramref name="source"/> contains at least one start marker.</summary>
    /// <param name="source">The program source to inspect.</param>
    /// <param name="markers">The marker pair to look for.</param>
    /// <returns><c>true</c> when any line contains the start marker.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is <c>null</c>.</exception>
    public static bool ContainsStartMarker(string source, EvolveBlockMarkers markers = default)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        return source.IndexOf(markers.Start, StringComparison.Ordinal) >= 0;
    }

    /// <summary>Replaces the body of the first well-formed block, preserving everything outside it.</summary>
    /// <param name="source">The program source to rewrite.</param>
    /// <param name="replacement">The new body for the first block.</param>
    /// <param name="markers">The marker pair to look for.</param>
    /// <param name="rewritten">The rewritten source when a block was found; otherwise <paramref name="source"/>.</param>
    /// <returns><c>true</c> when a well-formed block existed and was rewritten.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> or <paramref name="replacement"/> is <c>null</c>.</exception>
    public static bool TryReplaceFirst(string source, string replacement, EvolveBlockMarkers markers, out string rewritten)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (replacement is null) throw new ArgumentNullException(nameof(replacement));

        EvolveBlockExtractionResult extraction = Extract(source, markers);
        if (!extraction.TryGetPrimaryRegion(out EvolveBlockRegion region))
        {
            rewritten = source;
            return false;
        }

        rewritten = region.Rewrite(replacement);
        return true;
    }

    /// <summary>Replaces the body of every well-formed block, preserving everything outside them.</summary>
    /// <param name="source">The program source to rewrite.</param>
    /// <param name="replacements">
    /// One replacement body per block, in the order the blocks appear; the count must match the number of blocks.
    /// </param>
    /// <param name="markers">The marker pair to look for.</param>
    /// <param name="rewritten">The rewritten source on success; otherwise <paramref name="source"/>.</param>
    /// <returns>
    /// <c>true</c> when the source contained exactly <c>replacements.Count</c> well-formed blocks and all were
    /// rewritten; <c>false</c> when the counts differ or no block exists, leaving the source unchanged.
    /// </returns>
    /// <exception cref="ArgumentNullException">
    /// <paramref name="source"/> or <paramref name="replacements"/> is <c>null</c>, or a replacement is <c>null</c>.
    /// </exception>
    public static bool TryReplaceAll(
        string source,
        IReadOnlyList<string> replacements,
        EvolveBlockMarkers markers,
        out string rewritten)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (replacements is null) throw new ArgumentNullException(nameof(replacements));
        for (int index = 0; index < replacements.Count; index++)
        {
            if (replacements[index] is null)
                throw new ArgumentNullException(nameof(replacements), "Replacement bodies cannot be null.");
        }

        EvolveBlockExtractionResult extraction = Extract(source, markers);
        if (extraction.Regions.Count == 0 || extraction.Regions.Count != replacements.Count)
        {
            rewritten = source;
            return false;
        }

        string current = source;
        for (int index = extraction.Regions.Count - 1; index >= 0; index--)
        {
            EvolveBlockExtractionResult pass = Extract(current, markers);
            if (pass.Regions.Count != extraction.Regions.Count)
            {
                rewritten = source;
                return false;
            }

            current = pass.Regions[index].Rewrite(replacements[index]);
        }

        rewritten = current;
        return true;
    }

    /// <summary>Wraps a source that has no start marker in one evolve block.</summary>
    /// <param name="source">The program source to fence.</param>
    /// <param name="markers">The marker pair to insert.</param>
    /// <returns>
    /// The source unchanged when it already contains a start marker; otherwise the source preceded by the start
    /// marker line and followed by the end marker line, using the source's own line terminator.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is <c>null</c>.</exception>
    public static string Wrap(string source, EvolveBlockMarkers markers = default)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (ContainsStartMarker(source, markers)) return source;

        string newLine = ProgramText.DetectNewLine(source);
        string body = ProgramText.EndsWithNewLine(source) ? source : source + newLine;
        return string.Concat(markers.Start, newLine, body, markers.End, newLine);
    }

    private static EvolveBlockRegion BuildRegion(
        List<string> lines,
        string newLine,
        bool trailingNewLine,
        int startLineIndex,
        int endLineIndex)
    {
        var prefixLines = new List<string>();
        for (int index = 0; index <= startLineIndex; index++) prefixLines.Add(lines[index]);
        string prefix = ProgramText.JoinLines(prefixLines, newLine, trailingNewLine: true);

        var bodyLines = new List<string>();
        for (int index = startLineIndex + 1; index < endLineIndex; index++) bodyLines.Add(lines[index]);
        string body = bodyLines.Count == 0 ? string.Empty : ProgramText.JoinLines(bodyLines, newLine, trailingNewLine: true);

        var suffixLines = new List<string>();
        for (int index = endLineIndex; index < lines.Count; index++) suffixLines.Add(lines[index]);
        string suffix = ProgramText.JoinLines(suffixLines, newLine, trailingNewLine);

        return new EvolveBlockRegion(prefix, body, suffix, newLine, startLineIndex, endLineIndex);
    }

    private static void AddDiagnostic(List<string> diagnostics, string message)
    {
        if (diagnostics.Count >= EvolveBlockExtractionResult.MaxDiagnostics) return;
        diagnostics.Add(message);
    }

    private static string Ordinal(int lineIndex) => (lineIndex + 1).ToString(CultureInfo.InvariantCulture);
}
