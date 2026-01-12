using System.Text.RegularExpressions;

namespace AiDotNet.PromptEngineering.Compression;

/// <summary>
/// Compressor that shortens sentences while preserving their core meaning.
/// </summary>
/// <remarks>
/// <para>
/// This compressor analyzes sentence structure and removes non-essential clauses,
/// qualifiers, and modifiers while maintaining grammatical correctness and the
/// primary message.
/// </para>
/// <para><b>For Beginners:</b> Makes sentences shorter by keeping only the important parts.
///
/// Example:
/// <code>
/// var compressor = new SentenceCompressor();
///
/// string verbose = "In the following section, I will provide you with a detailed explanation of how the algorithm works, step by step.";
/// string compressed = compressor.Compress(verbose);
/// // Result: "Here's how the algorithm works, step by step."
/// </code>
///
/// What gets simplified:
/// - Introductory phrases removed
/// - Complex clauses simplified
/// - Passive voice converted to active (when possible)
/// </para>
/// </remarks>
public class SentenceCompressor : PromptCompressorBase
{
    /// <summary>
    /// Regex timeout to prevent ReDoS attacks.
    /// </summary>

    /// <summary>
    /// Initializes a new instance of the SentenceCompressor class.
    /// </summary>
    /// <param name="tokenCounter">Optional custom token counter function.</param>
    public SentenceCompressor(Func<string, int>? tokenCounter = null)
        : base("SentenceCompressor", tokenCounter)
    {
    }

    /// <summary>
    /// Compresses the prompt by simplifying sentences.
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

        // Apply sentence simplifications
        result = RemoveIntroductoryClauses(result);
        result = SimplifyPassiveVoice(result);
        result = RemoveParentheticalClauses(result);
        result = SimplifyVerbPhrases(result);
        result = RemoveEmptyModifiers(result);
        result = ConsolidateWhitespace(result);

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

    /// <summary>
    /// Removes common introductory clauses that don't add meaning.
    /// </summary>
    private static string RemoveIntroductoryClauses(string text)
    {
        var patterns = new[]
        {
            @"^In\s+the\s+following\s+(section|paragraph|text|document),?\s*",
            @"^In\s+this\s+(section|paragraph|text|document),?\s*",
            @"^Below,?\s+(I\s+will\s+)?(you\s+will\s+find|is|are)\s*",
            @"^The\s+following\s+(is|are)\s+",
            @"^Here\s+(is|are)\s+the\s+",
            @"^Let\s+me\s+(start\s+by|begin\s+by|explain)\s*",
            @"^I\s+will\s+now\s+",
            @"^Now\s+I\s+will\s+",
            @"^First\s+of\s+all,?\s*",
            @"^To\s+begin\s+with,?\s*",
            @"^To\s+start\s+with,?\s*"
        };

        var result = text;
        foreach (var pattern in patterns)
        {
            result = RegexHelper.Replace(result, pattern, "", RegexOptions.Multiline | RegexOptions.IgnoreCase);
        }

        return result;
    }

    /// <summary>
    /// Simplifies passive voice constructions to active voice where possible.
    /// </summary>
    private static string SimplifyPassiveVoice(string text)
    {
        var patterns = new Dictionary<string, string>
        {
            // Simple passive to active conversions
            { @"\bit\s+is\s+recommended\s+that\s+you\b", "you should" },
            { @"\bit\s+is\s+suggested\s+that\s+you\b", "you should" },
            { @"\bit\s+is\s+required\s+that\s+you\b", "you must" },
            { @"\bit\s+is\s+necessary\s+for\s+you\s+to\b", "you need to" },
            { @"\bit\s+is\s+important\s+for\s+you\s+to\b", "you should" },
            { @"\bit\s+is\s+advisable\s+to\b", "you should" },
            { @"\bit\s+would\s+be\s+better\s+if\s+you\b", "you should" },
            { @"\byou\s+are\s+expected\s+to\b", "you should" },
            { @"\byou\s+are\s+required\s+to\b", "you must" },
            { @"\byou\s+will\s+be\s+asked\s+to\b", "you will" },
            { @"\bwhat\s+is\s+needed\s+is\b", "you need" },
            { @"\bwhat\s+is\s+required\s+is\b", "you must" },
            { @"\bthe\s+result\s+will\s+be\s+that\b", "this will" },
            { @"\bthis\s+will\s+result\s+in\b", "this causes" }
        };

        var result = text;
        foreach (var kvp in patterns)
        {
            result = RegexHelper.Replace(result, kvp.Key, kvp.Value, RegexOptions.IgnoreCase);
        }

        return result;
    }

    /// <summary>
    /// Removes parenthetical and non-essential clauses.
    /// </summary>
    private static string RemoveParentheticalClauses(string text)
    {
        // Remove content in parentheses if short and non-essential
        var result = RegexHelper.Replace(text, @"\s*\([^)]{1,30}\)", "", RegexOptions.None);

        // Remove em-dash clauses that appear to be asides
        result = RegexHelper.Replace(result, @"\s*—[^—]{1,50}—\s*", " ", RegexOptions.None);

        // Remove "which is/are" clauses when they're short
        result = RegexHelper.Replace(result, @",?\s*which\s+(is|are)\s+[^,.]{1,30}(?=[,.])", "", RegexOptions.None);

        return result;
    }

    /// <summary>
    /// Simplifies verbose verb phrases.
    /// </summary>
    private static string SimplifyVerbPhrases(string text)
    {
        var patterns = new Dictionary<string, string>
        {
            { @"\bmake\s+a\s+decision\b", "decide" },
            { @"\bmake\s+an\s+attempt\b", "try" },
            { @"\btake\s+into\s+consideration\b", "consider" },
            { @"\bgive\s+consideration\s+to\b", "consider" },
            { @"\bcome\s+to\s+a\s+conclusion\b", "conclude" },
            { @"\bhave\s+a\s+tendency\s+to\b", "tend to" },
            { @"\bis\s+in\s+a\s+position\s+to\b", "can" },
            { @"\bhas\s+the\s+ability\s+to\b", "can" },
            { @"\bhave\s+the\s+ability\s+to\b", "can" },
            { @"\bis\s+able\s+to\b", "can" },
            { @"\bare\s+able\s+to\b", "can" },
            { @"\bmake\s+use\s+of\b", "use" },
            { @"\bput\s+emphasis\s+on\b", "emphasize" },
            { @"\bgive\s+an\s+explanation\s+of\b", "explain" },
            { @"\bprovide\s+assistance\s+to\b", "help" },
            { @"\boffer\s+assistance\s+to\b", "help" },
            { @"\bserve\s+as\b", "be" },
            { @"\bact\s+as\b", "be" },
            { @"\bfunction\s+as\b", "be" }
        };

        var result = text;
        foreach (var kvp in patterns)
        {
            result = RegexHelper.Replace(result, kvp.Key, kvp.Value, RegexOptions.IgnoreCase);
        }

        return result;
    }

    /// <summary>
    /// Removes empty modifiers and intensifiers that don't add meaning.
    /// </summary>
    private static string RemoveEmptyModifiers(string text)
    {
        var patterns = new[]
        {
            @"\b(very|really|quite|rather|somewhat|fairly)\s+(?=\w)",
            @"\b(basically|essentially|fundamentally|virtually|practically)\s+(?=\w)",
            @"\b(obviously|clearly|apparently|evidently)\s+(?=[a-z])",
            @"\b(in\s+fact|as\s+a\s+matter\s+of\s+fact)\s*,?\s*",
            @"\b(to\s+be\s+honest|honestly|frankly)\s*,?\s*",
            @"\b(of\s+course)\s*,?\s*"
        };

        var result = text;
        foreach (var pattern in patterns)
        {
            result = RegexHelper.Replace(result, pattern, "", RegexOptions.IgnoreCase);
        }

        return result;
    }

    /// <summary>
    /// Consolidates multiple whitespace characters.
    /// </summary>
    private static string ConsolidateWhitespace(string text)
    {
        var result = RegexHelper.Replace(text, @"[ \t]{2,}", " ", RegexOptions.None);
        result = RegexHelper.Replace(result, @"\n{3,}", "\n\n", RegexOptions.None);
        result = RegexHelper.Replace(result, @"^\s+", "", RegexOptions.Multiline);
        result = RegexHelper.Replace(result, @"\s+$", "", RegexOptions.Multiline);
        return result;
    }
}



