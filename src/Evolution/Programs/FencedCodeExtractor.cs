using AiDotNet.Enums;
using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.Evolution.Programs;

/// <summary>Extracts fenced code blocks from a model response and picks the best full rewrite.</summary>
/// <remarks>
/// <para>
/// The reference OpenEvolve implementation builds a regular expression by concatenating the language name into
/// <c>```&lt;language&gt;\n(.*?)```</c>, then falls back to <c>```(.*?)```</c> and finally to the whole response.
/// Three consequences follow: the first fence always wins even when it is a short illustrative snippet, a nested
/// fence terminates its parent early, and a caller cannot tell a labelled extraction from a raw-prose fallback.
/// This scanner walks the response line by line with CommonMark fence rules — an opening run of three or more
/// backticks or tildes is closed only by a run of at least the same length of the same character — so nested
/// fences survive, and it selects the longest block on the strongest available rung rather than the first block
/// on any rung.
/// </para>
/// <para>
/// Selection order is: a fence whose label resolves to the requested language, then an unlabelled fence, then any
/// other labelled fence, then the raw response. Within a rung the longest block wins and ties go to the earliest,
/// so the choice is reproducible. Content keeps its leading indentation, which matters for Python, and only
/// leading and trailing blank lines are removed.
/// </para>
/// <para><b>For Beginners:</b> Chat models normally return code wrapped in triple backticks. This class pulls the
/// code back out. It copes with the awkward cases: a model that shows a short example before the real answer, a
/// model that forgot the language name, and a model that nested one code block inside another. It also tells you
/// how it found the code, so you can retry when the model clearly ignored the requested format.</para>
/// </remarks>
public static class FencedCodeExtractor
{
    /// <summary>Extracts the best full rewrite from a model response.</summary>
    /// <param name="response">The raw model response text.</param>
    /// <param name="language">
    /// The language whose fence label is preferred. <see cref="ProgramLanguage.Generic"/> has no label, so an
    /// unlabelled fence becomes the strongest rung.
    /// </param>
    /// <param name="allowRawFallback">
    /// Whether a response with no fence at all yields the trimmed response text. When <c>false</c> such a response
    /// yields <see cref="FencedCodeSelectionSource.None"/> and empty code, which is the safer setting when the
    /// model was explicitly told to use a fence.
    /// </param>
    /// <returns>The selected code, the rung it came from, every block seen, and bounded diagnostics.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="response"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="language"/> is not a defined value.</exception>
    public static FencedCodeExtractionResult Extract(
        string response,
        ProgramLanguage language = ProgramLanguage.Generic,
        bool allowRawFallback = true)
    {
        if (response is null) throw new ArgumentNullException(nameof(response));
        if (!Enum.IsDefined(typeof(ProgramLanguage), language)) throw new ArgumentOutOfRangeException(nameof(language));

        var diagnostics = new List<string>();
        IReadOnlyList<FencedCodeBlock> blocks = ScanBlocks(response, diagnostics);

        FencedCodeBlock? labeled = SelectLongest(blocks, block => block.Language.HasValue && block.Language.Value == language);
        if (labeled is not null)
        {
            return new FencedCodeExtractionResult(
                labeled.Content, FencedCodeSelectionSource.LanguageLabeledFence, blocks, diagnostics);
        }

        FencedCodeBlock? unlabeled = SelectLongest(blocks, block => block.IsUnlabeled);
        if (unlabeled is not null)
        {
            if (language != ProgramLanguage.Generic)
            {
                diagnostics.Add("No fence was labelled '" + ProgramLanguageDetector.GetFenceLabel(language) +
                    "'; the longest unlabelled fence was used instead.");
            }

            return new FencedCodeExtractionResult(
                unlabeled.Content, FencedCodeSelectionSource.UnlabeledFence, blocks, diagnostics);
        }

        FencedCodeBlock? other = SelectLongest(blocks, block => !block.IsUnlabeled);
        if (other is not null)
        {
            diagnostics.Add("The only fences present were labelled '" + other.Label +
                "', which does not match the requested language.");
            return new FencedCodeExtractionResult(
                other.Content, FencedCodeSelectionSource.OtherLabeledFence, blocks, diagnostics);
        }

        string raw = TrimBlankLines(response);
        if (!allowRawFallback || raw.Length == 0)
        {
            diagnostics.Add("The response contains no fenced code block.");
            return new FencedCodeExtractionResult(string.Empty, FencedCodeSelectionSource.None, blocks, diagnostics);
        }

        diagnostics.Add("The response contains no fenced code block; the whole response was used verbatim.");
        return new FencedCodeExtractionResult(raw, FencedCodeSelectionSource.RawResponse, blocks, diagnostics);
    }

    /// <summary>Returns every fenced block in a response without selecting one.</summary>
    /// <param name="response">The raw model response text.</param>
    /// <returns>The blocks in the order their opening fences appear; empty when the response has no fence.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="response"/> is <c>null</c>.</exception>
    public static IReadOnlyList<FencedCodeBlock> ExtractBlocks(string response)
    {
        if (response is null) throw new ArgumentNullException(nameof(response));
        return ScanBlocks(response, new List<string>());
    }

    private static IReadOnlyList<FencedCodeBlock> ScanBlocks(string response, List<string> diagnostics)
    {
        List<string> lines = ProgramText.SplitLines(response);
        var blocks = new List<FencedCodeBlock>();

        int index = 0;
        while (index < lines.Count)
        {
            if (!TryReadOpeningFence(lines[index], out char fenceChar, out int fenceLength, out int indent, out string label))
            {
                index++;
                continue;
            }

            int openLine = index;
            var content = new List<string>();
            index++;
            bool closed = false;
            while (index < lines.Count)
            {
                if (IsClosingFence(lines[index], fenceChar, fenceLength))
                {
                    closed = true;
                    index++;
                    break;
                }

                content.Add(StripIndent(lines[index], indent));
                index++;
            }

            if (!closed)
            {
                diagnostics.Add("The fence opened on line " +
                    (openLine + 1).ToString(System.Globalization.CultureInfo.InvariantCulture) +
                    " was never closed; the remainder of the response was treated as its content.");
            }

            string text = TrimBlankLines(ProgramText.JoinLines(content, ProgramText.LineFeedText, trailingNewLine: false));
            if (text.Length > 0) blocks.Add(new FencedCodeBlock(label, text, openLine + 1, fenceLength));
        }

        return blocks;
    }

    private static bool TryReadOpeningFence(string line, out char fenceChar, out int fenceLength, out int indent, out string label)
    {
        fenceChar = '`';
        fenceLength = 0;
        indent = 0;
        label = string.Empty;

        int position = 0;
        while (position < line.Length && line[position] == ' ') position++;
        if (position > 3 || position >= line.Length) return false;

        char candidate = line[position];
        if (candidate != '`' && candidate != '~') return false;

        int run = 0;
        while (position + run < line.Length && line[position + run] == candidate) run++;
        if (run < 3) return false;

        string info = line.Substring(position + run).Trim();
        if (candidate == '`' && info.IndexOf('`') >= 0) return false;

        fenceChar = candidate;
        fenceLength = run;
        indent = position;
        label = info;
        return true;
    }

    private static bool IsClosingFence(string line, char fenceChar, int fenceLength)
    {
        int position = 0;
        while (position < line.Length && line[position] == ' ') position++;
        if (position > 3) return false;

        int run = 0;
        while (position + run < line.Length && line[position + run] == fenceChar) run++;
        if (run < fenceLength) return false;

        for (int index = position + run; index < line.Length; index++)
        {
            if (!char.IsWhiteSpace(line[index])) return false;
        }

        return true;
    }

    private static string StripIndent(string line, int indent)
    {
        int removed = 0;
        while (removed < indent && removed < line.Length && line[removed] == ' ') removed++;
        return removed == 0 ? line : line.Substring(removed);
    }

    private static string TrimBlankLines(string text)
    {
        List<string> lines = ProgramText.SplitLines(text);
        int first = 0;
        while (first < lines.Count && lines[first].Trim().Length == 0) first++;
        int last = lines.Count - 1;
        while (last >= first && lines[last].Trim().Length == 0) last--;
        if (last < first) return string.Empty;

        var kept = new List<string>();
        for (int index = first; index <= last; index++) kept.Add(lines[index]);
        return ProgramText.JoinLines(kept, ProgramText.LineFeedText, trailingNewLine: false).TrimEnd();
    }

    private static FencedCodeBlock? SelectLongest(IReadOnlyList<FencedCodeBlock> blocks, Func<FencedCodeBlock, bool> predicate)
    {
        FencedCodeBlock? best = null;
        foreach (FencedCodeBlock block in blocks)
        {
            if (!predicate(block)) continue;
            if (best is null || block.Content.Length > best.Content.Length) best = block;
        }

        return best;
    }
}
