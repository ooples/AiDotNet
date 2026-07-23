using System.Text.RegularExpressions;

namespace AiDotNet.PromptEngineering.Compression;

/// <summary>
/// Compressor that removes redundant phrases and verbose language from prompts.
/// </summary>
/// <remarks>
/// <para>
/// This compressor identifies and removes common patterns of redundant language
/// such as filler phrases, unnecessary qualifiers, and verbose constructions.
/// It preserves the semantic meaning while making the prompt more concise.
/// </para>
/// <para><b>For Beginners:</b> Removes wordy phrases without changing meaning.
///
/// Example:
/// <code>
/// var compressor = new RedundancyRemovalCompressor();
///
/// string verbose = "I would like you to please help me to analyze the following text";
/// string compressed = compressor.Compress(verbose);
/// // Result: "Analyze this text"
///
/// // What gets removed:
/// // - "I would like you to" → (removed)
/// // - "please help me to" → (removed)
/// // - "the following" → "this"
/// </code>
/// </para>
/// </remarks>
public class RedundancyRemovalCompressor : PromptCompressorBase
{
    /// <summary>
    /// Regex timeout to prevent ReDoS attacks.
    /// </summary>
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);

    private readonly List<(Regex Pattern, string Replacement)> _replacements;

    /// <summary>
    /// Initializes a new instance of the RedundancyRemovalCompressor class.
    /// </summary>
    /// <param name="tokenCounter">Optional custom token counter function.</param>
    public RedundancyRemovalCompressor(Func<string, int>? tokenCounter = null)
        : base("RedundancyRemoval", tokenCounter)
    {
        _replacements = InitializeReplacements();
    }

    /// <summary>
    /// Initializes the replacement patterns for redundancy removal.
    /// </summary>
    private static List<(Regex Pattern, string Replacement)> InitializeReplacements()
    {
        var patterns = new List<(string Pattern, string Replacement)>
        {
            // Polite but redundant phrases
            (@"\bplease\s+kindly\b", "please"),
            (@"\bkindly\s+please\b", "please"),
            (@"\bI\s+would\s+like\s+(you\s+)?to\b", ""),
            (@"\bcould\s+you\s+please\b", ""),
            (@"\bwould\s+you\s+please\b", ""),
            (@"\bcan\s+you\s+please\b", ""),
            (@"\bplease\s+help\s+me\s+to\b", ""),
            (@"\bhelp\s+me\s+to\b", ""),
            (@"\bI\s+need\s+you\s+to\b", ""),
            (@"\bI\s+want\s+you\s+to\b", ""),

            // Filler phrases
            (@"\bin\s+order\s+to\b", "to"),
            (@"\bfor\s+the\s+purpose\s+of\b", "to"),
            (@"\bwith\s+the\s+aim\s+of\b", "to"),
            (@"\bso\s+as\s+to\b", "to"),
            (@"\bdue\s+to\s+the\s+fact\s+that\b", "because"),
            (@"\bowing\s+to\s+the\s+fact\s+that\b", "because"),
            (@"\bin\s+light\s+of\s+the\s+fact\s+that\b", "because"),
            (@"\bgiven\s+the\s+fact\s+that\b", "since"),
            (@"\bas\s+a\s+result\s+of\b", "because"),
            (@"\bat\s+this\s+point\s+in\s+time\b", "now"),
            (@"\bat\s+the\s+present\s+time\b", "now"),
            (@"\bin\s+the\s+event\s+that\b", "if"),
            (@"\bin\s+the\s+case\s+that\b", "if"),
            (@"\bunder\s+circumstances\s+where\b", "when"),

            // Verbose references
            (@"\bthe\s+following\b", "this"),
            (@"\bthe\s+above\s+mentioned\b", "the"),
            (@"\bthe\s+aforementioned\b", "the"),
            (@"\bthe\s+previously\s+mentioned\b", "the"),
            (@"\bas\s+mentioned\s+above\b", ""),
            (@"\bas\s+stated\s+earlier\b", ""),
            (@"\bas\s+discussed\s+previously\b", ""),

            // Unnecessary qualifiers
            (@"\bvery\s+unique\b", "unique"),
            (@"\bcompletely\s+unique\b", "unique"),
            (@"\babsolutely\s+essential\b", "essential"),
            (@"\bvery\s+essential\b", "essential"),
            (@"\bextremely\s+important\b", "important"),
            (@"\bvery\s+important\b", "important"),
            (@"\bquite\s+obviously\b", "obviously"),
            (@"\bbasically\s+fundamental\b", "fundamental"),

            // Redundant word pairs
            (@"\beach\s+and\s+every\b", "every"),
            (@"\bone\s+and\s+only\b", "only"),
            (@"\bfirst\s+and\s+foremost\b", "first"),
            (@"\bvarious\s+different\b", "various"),
            (@"\bpast\s+history\b", "history"),
            (@"\bfuture\s+plans\b", "plans"),
            (@"\badvance\s+warning\b", "warning"),
            (@"\bfinal\s+outcome\b", "outcome"),
            (@"\bfree\s+gift\b", "gift"),
            (@"\bjoin\s+together\b", "join"),
            (@"\bmerge\s+together\b", "merge"),
            (@"\bcombine\s+together\b", "combine"),

            // Wordy instructions
            (@"\bmake\s+sure\s+that\s+you\b", ""),
            (@"\bensure\s+that\s+you\b", ""),
            (@"\bbe\s+sure\s+to\b", ""),
            (@"\bit\s+is\s+important\s+to\s+note\s+that\b", "note:"),
            (@"\bit\s+should\s+be\s+noted\s+that\b", "note:"),
            (@"\bplease\s+note\s+that\b", "note:"),

            // AI-specific redundancies
            (@"\byou\s+are\s+an?\s+AI\s+assistant\b", ""),
            (@"\bas\s+an?\s+AI\s+(language\s+)?model\b", ""),
            (@"\byour\s+task\s+is\s+to\b", ""),
            (@"\byour\s+goal\s+is\s+to\b", ""),
            (@"\byour\s+job\s+is\s+to\b", ""),
        };

        return patterns
            .Select(p => (new Regex(p.Pattern, RegexOptions.IgnoreCase | RegexOptions.Compiled, RegexTimeout), p.Replacement))
            .ToList();
    }

    /// <summary>
    /// Compresses the prompt by removing redundant phrases.
    /// </summary>
    protected override string CompressCore(string prompt, CompressionOptions options)
    {
        var result = prompt;

        // Handle variable preservation
        Dictionary<string, string>? variables = null;
        if (options.PreserveVariables)
        {
            variables = ExtractVariables(result);
            result = ReplaceVariablesWithPlaceholders(result, variables);
        }

        // Handle code block preservation
        Dictionary<string, string>? codeBlocks = null;
        if (options.PreserveCodeBlocks)
        {
            codeBlocks = ExtractCodeBlocks(result);
            result = ReplaceCodeBlocksWithPlaceholders(result, codeBlocks);
        }

        // Apply all replacement patterns
        foreach (var (pattern, replacement) in _replacements)
        {
            result = pattern.Replace(result, replacement);
        }

        // Clean up multiple spaces
        result = Regex.Replace(result, @"\s{2,}", " ", RegexOptions.None, RegexTimeout);

        // Clean up leading/trailing spaces from lines
        result = Regex.Replace(result, @"^\s+", "", RegexOptions.Multiline, RegexTimeout);
        result = Regex.Replace(result, @"\s+$", "", RegexOptions.Multiline, RegexTimeout);

        // Remove empty lines
        result = Regex.Replace(result, @"\n\s*\n", "\n", RegexOptions.None, RegexTimeout);

        // Restore code blocks
        if (codeBlocks != null)
        {
            result = RestoreCodeBlocks(result, codeBlocks);
        }

        // Restore variables
        if (variables != null)
        {
            result = RestoreVariables(result, variables);
        }

        return result.Trim();
    }
}
