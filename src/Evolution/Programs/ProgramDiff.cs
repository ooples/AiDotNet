using System.Globalization;
using System.Text;
using AiDotNet.Configuration;
using AiDotNet.Enums;

namespace AiDotNet.Evolution.Programs;

/// <summary>Parses SEARCH/REPLACE edit blocks from a model response and applies them to a program.</summary>
/// <remarks>
/// <para>
/// The block format is the one the reference OpenEvolve implementation emits: a line containing
/// <c>&lt;&lt;&lt;&lt;&lt;&lt;&lt; SEARCH</c>, the text to find, a line of equals signs, the replacement text, and a
/// line containing <c>&gt;&gt;&gt;&gt;&gt;&gt;&gt; REPLACE</c>. Upstream that format is recognized with a single
/// regular expression that hard-codes bare line feeds, so a response with Windows line endings, an indented
/// marker, or a trailing space after a marker matches nothing and the whole iteration is reported as "no valid
/// diffs found". This parser is line based and tolerant of all three, while keeping the marker text configurable
/// through <see cref="ProgramDiffOptions"/>.
/// </para>
/// <para>
/// The second upstream weakness is silence on failure: <c>apply_diff</c> looks for the first exact line-window
/// match and, when there is none, moves to the next block without recording anything, so an iteration can produce
/// a child identical to its parent and still consume evaluator budget. Here every rejected block becomes a typed
/// <see cref="ProgramDiffFailure"/> naming the block and the reason, an empty replacement really deletes its lines
/// instead of leaving a blank one behind, an unchanged result is reported as a failure by default, and
/// <see cref="CreateUnifiedDiff"/> renders exactly what an application changed.
/// </para>
/// <para><b>For Beginners:</b> Large language models edit code by sending small "find this, replace it with that"
/// instructions rather than rewriting whole files. This class reads those instructions out of the model's answer
/// and carries them out. Crucially, when an instruction cannot be carried out — usually because the text it wants
/// to find is not in the file — you are told which one failed and why, so you can ask the model to try again with
/// that feedback instead of silently evolving nothing.</para>
/// </remarks>
public static class ProgramDiff
{
    private const int MaxLongestCommonSubsequenceLines = 400;

    /// <summary>Parses every SEARCH/REPLACE block in a model response.</summary>
    /// <param name="response">The raw model response text.</param>
    /// <param name="options">Marker text and limits; <c>null</c> uses the defaults.</param>
    /// <returns>The well-formed blocks plus a typed failure for every block that could not be parsed.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="response"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="options"/> carries invalid marker text.</exception>
    public static ProgramDiffParseResult Parse(string response, ProgramDiffOptions? options = null)
    {
        if (response is null) throw new ArgumentNullException(nameof(response));
        ProgramDiffOptions effective = options ?? new ProgramDiffOptions();
        effective.Validate();

        var blocks = new List<ProgramDiffBlock>();
        var failures = new List<ProgramDiffFailure>();

        if (!effective.AllowCarriageReturns && response.IndexOf('\r') >= 0)
        {
            failures.Add(new ProgramDiffFailure(
                ProgramDiffFailureReason.CarriageReturnRejected,
                "The response contains carriage returns while strict line-feed-only parsing is enabled."));
            return new ProgramDiffParseResult(blocks, failures);
        }

        List<string> lines = ProgramText.SplitLines(response);
        var searchBuffer = new List<string>();
        var replaceBuffer = new List<string>();
        int state = 0;
        int openLine = 0;

        for (int index = 0; index < lines.Count; index++)
        {
            string line = lines[index];
            bool isSearch = IsMarkerLine(line, effective.SearchMarker);
            bool isDivider = IsDividerLine(line, effective.DividerMarker);
            bool isReplace = IsMarkerLine(line, effective.ReplaceMarker);

            if (isSearch)
            {
                if (state != 0)
                {
                    failures.Add(new ProgramDiffFailure(
                        ProgramDiffFailureReason.MalformedBlock,
                        "A new SEARCH marker on line " + Line(index) + " interrupted the block opened on line " +
                        Line(openLine - 1) + "; the partial block was discarded.",
                        blocks.Count));
                }

                searchBuffer.Clear();
                replaceBuffer.Clear();
                state = 1;
                openLine = index + 1;
                continue;
            }

            if (state == 1 && isDivider)
            {
                state = 2;
                continue;
            }

            if (state == 2 && isReplace)
            {
                AddBlock(blocks, failures, effective, searchBuffer, replaceBuffer, openLine);
                searchBuffer.Clear();
                replaceBuffer.Clear();
                state = 0;
                continue;
            }

            if (state == 1 && isReplace)
            {
                failures.Add(new ProgramDiffFailure(
                    ProgramDiffFailureReason.MalformedBlock,
                    "The block opened on line " + Line(openLine - 1) + " reached its REPLACE marker on line " +
                    Line(index) + " without a divider line.",
                    blocks.Count));
                searchBuffer.Clear();
                replaceBuffer.Clear();
                state = 0;
                continue;
            }

            if (state == 0 && isReplace)
            {
                failures.Add(new ProgramDiffFailure(
                    ProgramDiffFailureReason.MalformedBlock,
                    "A REPLACE marker on line " + Line(index) + " has no matching SEARCH marker.",
                    blocks.Count));
                continue;
            }

            if (state == 1) searchBuffer.Add(line);
            else if (state == 2) replaceBuffer.Add(line);
        }

        if (state != 0)
        {
            failures.Add(new ProgramDiffFailure(
                ProgramDiffFailureReason.MalformedBlock,
                "The block opened on line " + Line(openLine - 1) + " was never closed by a REPLACE marker.",
                blocks.Count));
        }

        if (blocks.Count == 0 && failures.Count == 0)
        {
            failures.Add(new ProgramDiffFailure(
                ProgramDiffFailureReason.NoBlocksFound,
                "The response contains no SEARCH/REPLACE block."));
        }

        return new ProgramDiffParseResult(blocks, failures);
    }

    /// <summary>Applies parsed edit blocks to a program.</summary>
    /// <param name="source">The program the blocks edit.</param>
    /// <param name="blocks">The blocks to apply, in order.</param>
    /// <param name="options">
    /// Program-evolution settings supplying the diff behaviour and, when
    /// <see cref="ProgramEvolutionOptions.EnforceEvolveBlocks"/> is set, the marker pair that bounds legal edits;
    /// <c>null</c> uses the defaults.
    /// </param>
    /// <returns>The edited program, the number of blocks that applied, and a typed failure for each that did not.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> or <paramref name="blocks"/> is <c>null</c>, or a block is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="options"/> is invalid.</exception>
    public static ProgramDiffApplyResult Apply(
        string source,
        IReadOnlyList<ProgramDiffBlock> blocks,
        ProgramEvolutionOptions? options = null)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (blocks is null) throw new ArgumentNullException(nameof(blocks));
        ProgramEvolutionOptions effective = options ?? new ProgramEvolutionOptions();
        effective.Validate();
        ProgramDiffOptions diffOptions = effective.Diff;
        EvolveBlockMarkers markers = effective.ResolveEvolveBlockMarkers();

        string newLine = ProgramText.DetectNewLine(source);
        bool trailingNewLine = ProgramText.EndsWithNewLine(source);
        List<string> lines = ProgramText.SplitLines(source);
        if (trailingNewLine && lines.Count > 0 && lines[lines.Count - 1].Length == 0) lines.RemoveAt(lines.Count - 1);

        var failures = new List<ProgramDiffFailure>();
        int applied = 0;

        for (int blockIndex = 0; blockIndex < blocks.Count; blockIndex++)
        {
            ProgramDiffBlock block = blocks[blockIndex];
            if (block is null) throw new ArgumentNullException(nameof(blocks), "Edit blocks cannot be null.");

            List<string> searchLines = ProgramText.SplitLines(block.SearchText);
            if (block.SearchText.Length == 0 || searchLines.Count == 0)
            {
                failures.Add(new ProgramDiffFailure(
                    ProgramDiffFailureReason.EmptySearchText,
                    "Block " + Line(block.Ordinal) + " has an empty SEARCH section, which would match an arbitrary line.",
                    block.Ordinal));
                continue;
            }

            int match = FindWindow(lines, searchLines, fuzzy: false);
            if (match < 0 && diffOptions.FuzzyWhitespace) match = FindWindow(lines, searchLines, fuzzy: true);
            if (match < 0)
            {
                failures.Add(new ProgramDiffFailure(
                    ProgramDiffFailureReason.SearchTextNotFound,
                    "Block " + Line(block.Ordinal) + " could not be applied because its SEARCH text does not appear in the program.",
                    block.Ordinal,
                    Excerpt(block.SearchText, diffOptions.MaxFailureExcerptLength)));
                continue;
            }

            if (effective.EnforceEvolveBlocks
                && !IsInsideEvolveBlock(lines, newLine, trailingNewLine, markers, match, match + searchLines.Count - 1))
            {
                failures.Add(new ProgramDiffFailure(
                    ProgramDiffFailureReason.OutsideEvolveBlock,
                    "Block " + Line(block.Ordinal) + " matches outside every evolve block, and out-of-block edits are forbidden.",
                    block.Ordinal,
                    Excerpt(block.SearchText, diffOptions.MaxFailureExcerptLength)));
                continue;
            }

            lines.RemoveRange(match, searchLines.Count);
            if (block.ReplaceText.Length > 0) lines.InsertRange(match, ProgramText.SplitLines(block.ReplaceText));
            applied++;
        }

        string modified = ProgramText.JoinLines(lines, newLine, trailingNewLine && lines.Count > 0);
        bool changed = !string.Equals(source, modified, StringComparison.Ordinal);

        if (diffOptions.RejectWhenNoBlockApplied)
        {
            if (blocks.Count == 0)
            {
                failures.Add(new ProgramDiffFailure(
                    ProgramDiffFailureReason.NoBlocksFound,
                    "No edit blocks were supplied, so the program cannot change."));
            }
            else if (applied > 0 && !changed)
            {
                failures.Add(new ProgramDiffFailure(
                    ProgramDiffFailureReason.ResultUnchanged,
                    "Every applied block replaced text with itself, leaving the program byte identical to its parent."));
            }
        }

        bool success = failures.Count == 0 && applied > 0 && (!diffOptions.RejectWhenNoBlockApplied || changed);
        return new ProgramDiffApplyResult(source, modified, applied, failures, success);
    }

    /// <summary>Parses a model response and applies every block it contains in one step.</summary>
    /// <param name="source">The program the response edits.</param>
    /// <param name="response">The raw model response text.</param>
    /// <param name="options">Program-evolution settings; <c>null</c> uses the defaults.</param>
    /// <returns>
    /// The edited program with parse failures and application failures merged, so one result describes everything
    /// the caller has to report back to the model.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> or <paramref name="response"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException"><paramref name="options"/> is invalid.</exception>
    public static ProgramDiffApplyResult ApplyResponse(
        string source,
        string response,
        ProgramEvolutionOptions? options = null)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (response is null) throw new ArgumentNullException(nameof(response));
        ProgramEvolutionOptions effective = options ?? new ProgramEvolutionOptions();
        effective.Validate();

        ProgramDiffParseResult parsed = Parse(response, effective.Diff);
        ProgramDiffApplyResult appliedResult = Apply(source, parsed.Blocks, effective);
        if (parsed.Failures.Count == 0) return appliedResult;

        var merged = new List<ProgramDiffFailure>(parsed.Failures);
        foreach (ProgramDiffFailure failure in appliedResult.Failures)
        {
            if (failure.Reason == ProgramDiffFailureReason.NoBlocksFound
                && merged.Any(existing => existing.Reason == ProgramDiffFailureReason.NoBlocksFound))
            {
                continue;
            }

            merged.Add(failure);
        }

        return new ProgramDiffApplyResult(
            source,
            appliedResult.ModifiedSource,
            appliedResult.AppliedCount,
            merged,
            isSuccess: false);
    }

    /// <summary>Routes edit blocks to the program or to the changes description by where their SEARCH text occurs.</summary>
    /// <param name="blocks">The blocks parsed from one reply, in order.</param>
    /// <param name="source">The program the reply may edit.</param>
    /// <param name="changesDescription">The changes description the reply may edit.</param>
    /// <param name="options">Diff behaviour; <c>null</c> uses the defaults. Fuzzy whitespace applies to both targets.</param>
    /// <returns>The blocks belonging to each target, and a failure for every block whose target was unclear.</returns>
    /// <remarks>
    /// <para>
    /// A run that maintains a changes description shows the model two documents and asks it to edit both, but a
    /// SEARCH/REPLACE reply says only what text to find, never where. A block whose SEARCH text occurs in exactly one
    /// document belongs to that document. A block that occurs in both is refused with
    /// <see cref="ProgramDiffFailureReason.AmbiguousTarget"/> rather than guessed, because applying it to the wrong
    /// document is an edit nobody asked for that still looks like success.
    /// </para>
    /// <para>
    /// A block that matches neither is routed to the program, so the ordinary "SEARCH text not found" failure is
    /// reported by <see cref="Apply"/> against the document the model was most likely editing, with the excerpt and
    /// the block number it already reports. Splitting that message across two targets would say less, not more.
    /// </para>
    /// <para><b>For Beginners:</b> The model is editing a program and a short note describing its own changes. This
    /// works out which of the two each edit meant, and refuses an edit that could plausibly mean either.</para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">An argument other than <paramref name="options"/> is <c>null</c>.</exception>
    public static ProgramDiffTargetSplit SplitByTarget(
        IReadOnlyList<ProgramDiffBlock> blocks,
        string source,
        string changesDescription,
        ProgramEvolutionOptions? options = null)
    {
        if (blocks is null) throw new ArgumentNullException(nameof(blocks));
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (changesDescription is null) throw new ArgumentNullException(nameof(changesDescription));
        ProgramEvolutionOptions effective = options ?? new ProgramEvolutionOptions();
        bool fuzzy = effective.Diff.FuzzyWhitespace;

        List<string> sourceLines = ProgramText.SplitLines(source);
        List<string> descriptionLines = ProgramText.SplitLines(changesDescription);

        var programBlocks = new List<ProgramDiffBlock>();
        var descriptionBlocks = new List<ProgramDiffBlock>();
        var failures = new List<ProgramDiffFailure>();

        foreach (ProgramDiffBlock block in blocks)
        {
            if (block is null) throw new ArgumentNullException(nameof(blocks), "Edit blocks cannot be null.");
            List<string> searchLines = ProgramText.SplitLines(block.SearchText);
            if (block.SearchText.Length == 0 || searchLines.Count == 0)
            {
                // An empty SEARCH matches anywhere, so it has no target. Apply reports it precisely; routing it to
                // the program keeps that one message rather than adding a second, vaguer one here.
                programBlocks.Add(block);
                continue;
            }

            bool inSource = Matches(sourceLines, searchLines, fuzzy);
            bool inDescription = Matches(descriptionLines, searchLines, fuzzy);

            if (inSource && inDescription)
            {
                failures.Add(new ProgramDiffFailure(
                    ProgramDiffFailureReason.AmbiguousTarget,
                    "Block " + Line(block.Ordinal) + " has SEARCH text that occurs both in the program and in the " +
                    "changes description, so which one it edits is unclear. Extend the SEARCH text until it appears " +
                    "in only one of them.",
                    block.Ordinal,
                    Excerpt(block.SearchText, effective.Diff.MaxFailureExcerptLength)));
                continue;
            }

            if (inDescription) descriptionBlocks.Add(block);
            else programBlocks.Add(block);
        }

        return new ProgramDiffTargetSplit(programBlocks, descriptionBlocks, failures);
    }

    /// <summary>Reports whether a search window occurs in a document.</summary>
    private static bool Matches(List<string> lines, List<string> search, bool fuzzy)
    {
        if (search.Count == 0 || lines.Count < search.Count) return false;
        if (FindWindow(lines, search, fuzzy: false) >= 0) return true;
        return fuzzy && FindWindow(lines, search, fuzzy: true) >= 0;
    }

    /// <summary>Renders a human-readable summary of a set of edit blocks.</summary>
    /// <param name="blocks">The blocks to summarize.</param>
    /// <param name="maxLineLength">The longest line rendered before it is truncated with an ellipsis.</param>
    /// <param name="maxLines">The most lines rendered per section before the remainder is counted instead.</param>
    /// <returns>One paragraph per block, or an empty string when there are no blocks.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="blocks"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxLineLength"/> or <paramref name="maxLines"/> is not positive.</exception>
    public static string FormatSummary(IReadOnlyList<ProgramDiffBlock> blocks, int maxLineLength = 100, int maxLines = 30)
    {
        if (blocks is null) throw new ArgumentNullException(nameof(blocks));
        if (maxLineLength <= 0) throw new ArgumentOutOfRangeException(nameof(maxLineLength));
        if (maxLines <= 0) throw new ArgumentOutOfRangeException(nameof(maxLines));

        var builder = new StringBuilder();
        for (int index = 0; index < blocks.Count; index++)
        {
            ProgramDiffBlock block = blocks[index];
            if (block is null) throw new ArgumentNullException(nameof(blocks), "Edit blocks cannot be null.");
            List<string> searchLines = ProgramText.SplitLines(block.SearchText.Trim());
            List<string> replaceLines = ProgramText.SplitLines(block.ReplaceText.Trim());
            if (builder.Length > 0) builder.Append('\n');

            if (searchLines.Count == 1 && replaceLines.Count == 1)
            {
                builder.Append("Change ").Append(Line(index)).Append(": '")
                    .Append(ProgramText.Bound(searchLines[0], maxLineLength)).Append("' to '")
                    .Append(ProgramText.Bound(replaceLines[0], maxLineLength)).Append('\'');
                continue;
            }

            builder.Append("Change ").Append(Line(index)).Append(": Replace:\n")
                .Append(FormatBlockLines(searchLines, maxLineLength, maxLines))
                .Append("\nwith:\n")
                .Append(FormatBlockLines(replaceLines, maxLineLength, maxLines));
        }

        return builder.ToString();
    }

    /// <summary>Renders the difference between two program texts as a unified diff.</summary>
    /// <param name="original">The text before the change.</param>
    /// <param name="modified">The text after the change.</param>
    /// <param name="contextLines">The number of unchanged lines shown around each change.</param>
    /// <returns>A unified diff, or an empty string when the two texts are identical.</returns>
    /// <remarks>
    /// Changes are located with a longest-common-subsequence comparison after identical leading and trailing lines
    /// are removed. When more than 400 differing lines remain on either side the comparison falls back to a single
    /// replace hunk, which keeps the routine linear in memory for very large rewrites.
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="original"/> or <paramref name="modified"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="contextLines"/> is negative.</exception>
    public static string CreateUnifiedDiff(string original, string modified, int contextLines = 3)
    {
        if (original is null) throw new ArgumentNullException(nameof(original));
        if (modified is null) throw new ArgumentNullException(nameof(modified));
        if (contextLines < 0) throw new ArgumentOutOfRangeException(nameof(contextLines));
        if (string.Equals(original, modified, StringComparison.Ordinal)) return string.Empty;

        List<string> left = TrimTrailingBlank(ProgramText.SplitLines(original), original);
        List<string> right = TrimTrailingBlank(ProgramText.SplitLines(modified), modified);
        List<DiffRecord> records = BuildRecords(left, right);
        return RenderHunks(records, contextLines);
    }

    private static void AddBlock(
        List<ProgramDiffBlock> blocks,
        List<ProgramDiffFailure> failures,
        ProgramDiffOptions options,
        List<string> searchBuffer,
        List<string> replaceBuffer,
        int openLine)
    {
        string searchText = ProgramText.JoinLines(searchBuffer, ProgramText.LineFeedText, trailingNewLine: false).TrimEnd();
        string replaceText = ProgramText.JoinLines(replaceBuffer, ProgramText.LineFeedText, trailingNewLine: false).TrimEnd();

        if (searchText.Length == 0)
        {
            failures.Add(new ProgramDiffFailure(
                ProgramDiffFailureReason.EmptySearchText,
                "The block opened on line " + Line(openLine - 1) + " has an empty SEARCH section.",
                blocks.Count));
            return;
        }

        if (blocks.Count >= options.MaxBlocks)
        {
            failures.Add(new ProgramDiffFailure(
                ProgramDiffFailureReason.BlockLimitExceeded,
                "The response contains more than " + options.MaxBlocks.ToString(CultureInfo.InvariantCulture) +
                " edit blocks; the surplus was discarded.",
                blocks.Count,
                Excerpt(searchText, options.MaxFailureExcerptLength)));
            return;
        }

        blocks.Add(new ProgramDiffBlock(searchText, replaceText, blocks.Count, openLine));
    }

    private static bool IsMarkerLine(string line, string marker)
    {
        string trimmed = line.Trim();
        if (trimmed.Length < marker.Length) return false;
        if (string.CompareOrdinal(trimmed, 0, marker, 0, marker.Length) != 0) return false;
        for (int index = marker.Length; index < trimmed.Length; index++)
        {
            if (!char.IsWhiteSpace(trimmed[index])) return false;
        }

        return true;
    }

    private static bool IsDividerLine(string line, string marker)
    {
        string trimmed = line.Trim();
        if (trimmed.Length < marker.Length) return false;
        if (string.CompareOrdinal(trimmed, 0, marker, 0, marker.Length) != 0) return false;
        char repeated = marker[marker.Length - 1];
        for (int index = marker.Length; index < trimmed.Length; index++)
        {
            if (trimmed[index] != repeated && !char.IsWhiteSpace(trimmed[index])) return false;
        }

        return true;
    }

    private static int FindWindow(List<string> lines, List<string> search, bool fuzzy)
    {
        int limit = lines.Count - search.Count;
        for (int start = 0; start <= limit; start++)
        {
            bool matched = true;
            for (int offset = 0; offset < search.Count; offset++)
            {
                string candidate = lines[start + offset];
                string needle = search[offset];
                bool equal = fuzzy
                    ? string.Equals(FuzzyKey(candidate), FuzzyKey(needle), StringComparison.Ordinal)
                    : string.Equals(candidate, needle, StringComparison.Ordinal);
                if (!equal)
                {
                    matched = false;
                    break;
                }
            }

            if (matched) return start;
        }

        return -1;
    }

    private static string FuzzyKey(string line)
    {
        int indentLength = 0;
        while (indentLength < line.Length && (line[indentLength] == ' ' || line[indentLength] == '\t')) indentLength++;

        var builder = new StringBuilder(line.Length);
        builder.Append(line, 0, indentLength);
        bool pendingSpace = false;
        for (int index = indentLength; index < line.Length; index++)
        {
            char character = line[index];
            if (char.IsWhiteSpace(character))
            {
                pendingSpace = true;
                continue;
            }

            if (pendingSpace)
            {
                builder.Append(' ');
                pendingSpace = false;
            }

            builder.Append(character);
        }

        return builder.ToString();
    }

    private static bool IsInsideEvolveBlock(
        List<string> lines,
        string newLine,
        bool trailingNewLine,
        EvolveBlockMarkers markers,
        int firstLine,
        int lastLine)
    {
        string current = ProgramText.JoinLines(lines, newLine, trailingNewLine && lines.Count > 0);
        EvolveBlockExtractionResult extraction = EvolveBlock.Extract(current, markers);
        if (extraction.Regions.Count == 0) return true;

        foreach (EvolveBlockRegion region in extraction.Regions)
        {
            if (firstLine > region.StartLineIndex && lastLine < region.EndLineIndex) return true;
        }

        return false;
    }

    private static string Excerpt(string text, int maximumLength) =>
        ProgramText.Bound(ProgramText.Sanitize(text), maximumLength);

    private static string Line(int zeroBasedIndex) =>
        (zeroBasedIndex + 1).ToString(CultureInfo.InvariantCulture);

    private static string FormatBlockLines(List<string> lines, int maxLineLength, int maxLines)
    {
        if (lines.Count == 0 || (lines.Count == 1 && lines[0].Length == 0)) return "  (empty)";
        var builder = new StringBuilder();
        int rendered = Math.Min(lines.Count, maxLines);
        for (int index = 0; index < rendered; index++)
        {
            if (index > 0) builder.Append('\n');
            builder.Append("  ").Append(ProgramText.Bound(lines[index].TrimEnd(), maxLineLength));
        }

        if (lines.Count > maxLines)
        {
            builder.Append("\n  ... (")
                .Append((lines.Count - maxLines).ToString(CultureInfo.InvariantCulture))
                .Append(" more lines)");
        }

        return builder.ToString();
    }

    private static List<string> TrimTrailingBlank(List<string> lines, string source)
    {
        if (ProgramText.EndsWithNewLine(source) && lines.Count > 0 && lines[lines.Count - 1].Length == 0)
        {
            lines.RemoveAt(lines.Count - 1);
        }

        return lines;
    }

    private static List<DiffRecord> BuildRecords(List<string> left, List<string> right)
    {
        var records = new List<DiffRecord>();
        int prefix = 0;
        while (prefix < left.Count && prefix < right.Count
            && string.Equals(left[prefix], right[prefix], StringComparison.Ordinal))
        {
            prefix++;
        }

        int suffix = 0;
        while (suffix < left.Count - prefix && suffix < right.Count - prefix
            && string.Equals(left[left.Count - 1 - suffix], right[right.Count - 1 - suffix], StringComparison.Ordinal))
        {
            suffix++;
        }

        int leftLine = 1;
        int rightLine = 1;
        for (int index = 0; index < prefix; index++)
        {
            records.Add(new DiffRecord(DiffRecordKind.Equal, left[index], leftLine++, rightLine++));
        }

        int leftMiddle = left.Count - suffix - prefix;
        int rightMiddle = right.Count - suffix - prefix;

        if (leftMiddle > MaxLongestCommonSubsequenceLines || rightMiddle > MaxLongestCommonSubsequenceLines)
        {
            for (int index = 0; index < leftMiddle; index++)
            {
                records.Add(new DiffRecord(DiffRecordKind.Delete, left[prefix + index], leftLine++, rightLine));
            }

            for (int index = 0; index < rightMiddle; index++)
            {
                records.Add(new DiffRecord(DiffRecordKind.Insert, right[prefix + index], leftLine, rightLine++));
            }
        }
        else
        {
            AppendLongestCommonSubsequence(records, left, right, prefix, leftMiddle, rightMiddle, ref leftLine, ref rightLine);
        }

        for (int index = 0; index < suffix; index++)
        {
            records.Add(new DiffRecord(DiffRecordKind.Equal, left[left.Count - suffix + index], leftLine++, rightLine++));
        }

        return records;
    }

    private static void AppendLongestCommonSubsequence(
        List<DiffRecord> records,
        List<string> left,
        List<string> right,
        int offset,
        int leftCount,
        int rightCount,
        ref int leftLine,
        ref int rightLine)
    {
        var table = new int[leftCount + 1, rightCount + 1];
        for (int i = leftCount - 1; i >= 0; i--)
        {
            for (int j = rightCount - 1; j >= 0; j--)
            {
                table[i, j] = string.Equals(left[offset + i], right[offset + j], StringComparison.Ordinal)
                    ? table[i + 1, j + 1] + 1
                    : Math.Max(table[i + 1, j], table[i, j + 1]);
            }
        }

        int a = 0;
        int b = 0;
        while (a < leftCount || b < rightCount)
        {
            if (a < leftCount && b < rightCount
                && string.Equals(left[offset + a], right[offset + b], StringComparison.Ordinal))
            {
                records.Add(new DiffRecord(DiffRecordKind.Equal, left[offset + a], leftLine++, rightLine++));
                a++;
                b++;
            }
            else if (a < leftCount && (b == rightCount || table[a + 1, b] >= table[a, b + 1]))
            {
                records.Add(new DiffRecord(DiffRecordKind.Delete, left[offset + a], leftLine++, rightLine));
                a++;
            }
            else
            {
                records.Add(new DiffRecord(DiffRecordKind.Insert, right[offset + b], leftLine, rightLine++));
                b++;
            }
        }
    }

    private static string RenderHunks(List<DiffRecord> records, int contextLines)
    {
        var changeIndexes = new List<int>();
        for (int index = 0; index < records.Count; index++)
        {
            if (records[index].Kind != DiffRecordKind.Equal) changeIndexes.Add(index);
        }

        if (changeIndexes.Count == 0) return string.Empty;

        var builder = new StringBuilder();
        int position = 0;
        while (position < changeIndexes.Count)
        {
            int first = changeIndexes[position];
            int last = first;
            int lookahead = position + 1;
            while (lookahead < changeIndexes.Count && changeIndexes[lookahead] - last <= (contextLines * 2) + 1)
            {
                last = changeIndexes[lookahead];
                lookahead++;
            }

            int start = Math.Max(0, first - contextLines);
            int end = Math.Min(records.Count - 1, last + contextLines);
            AppendHunk(builder, records, start, end);
            position = lookahead;
        }

        return builder.ToString();
    }

    private static void AppendHunk(StringBuilder builder, List<DiffRecord> records, int start, int end)
    {
        int leftStart = 0;
        int rightStart = 0;
        int leftCount = 0;
        int rightCount = 0;
        for (int index = start; index <= end; index++)
        {
            DiffRecord record = records[index];
            if (record.Kind != DiffRecordKind.Insert)
            {
                if (leftCount == 0) leftStart = record.LeftLine;
                leftCount++;
            }

            if (record.Kind != DiffRecordKind.Delete)
            {
                if (rightCount == 0) rightStart = record.RightLine;
                rightCount++;
            }
        }

        if (leftCount == 0) leftStart = 0;
        if (rightCount == 0) rightStart = 0;

        builder.Append("@@ -").Append(leftStart.ToString(CultureInfo.InvariantCulture)).Append(',')
            .Append(leftCount.ToString(CultureInfo.InvariantCulture)).Append(" +")
            .Append(rightStart.ToString(CultureInfo.InvariantCulture)).Append(',')
            .Append(rightCount.ToString(CultureInfo.InvariantCulture)).Append(" @@\n");

        for (int index = start; index <= end; index++)
        {
            DiffRecord record = records[index];
            char marker = record.Kind == DiffRecordKind.Equal ? ' ' : record.Kind == DiffRecordKind.Delete ? '-' : '+';
            builder.Append(marker).Append(record.Text).Append('\n');
        }
    }

    private enum DiffRecordKind
    {
        Equal,
        Delete,
        Insert
    }

    private readonly struct DiffRecord
    {
        internal DiffRecord(DiffRecordKind kind, string text, int leftLine, int rightLine)
        {
            Kind = kind;
            Text = text;
            LeftLine = leftLine;
            RightLine = rightLine;
        }

        internal DiffRecordKind Kind { get; }
        internal string Text { get; }
        internal int LeftLine { get; }
        internal int RightLine { get; }
    }
}
