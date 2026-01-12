using System.Text.RegularExpressions;

namespace AiDotNet.PromptEngineering.Compression;

/// <summary>
/// Compressor that removes common stop words to reduce prompt length.
/// </summary>
/// <remarks>
/// <para>
/// This compressor removes frequently occurring words that often don't add significant
/// meaning to the prompt (articles, prepositions, auxiliary verbs, etc.). It's more aggressive
/// than redundancy removal and may slightly reduce readability.
/// </para>
/// <para><b>For Beginners:</b> Removes common words like "the", "a", "is" to make prompts shorter.
///
/// Example:
/// <code>
/// var compressor = new StopWordRemovalCompressor();
///
/// string original = "Please analyze the document and provide a summary of the main points";
/// string compressed = compressor.Compress(original);
/// // Result: "analyze document provide summary main points"
/// </code>
///
/// When to use:
/// - When maximum compression is needed
/// - When the AI can infer meaning from keywords
/// - Not recommended for conversational prompts
/// </para>
/// </remarks>
public class StopWordRemovalCompressor : PromptCompressorBase
{
    /// <summary>
    /// Regex timeout to prevent ReDoS attacks.
    /// </summary>

    private readonly HashSet<string> _stopWords;
    private readonly AggressivenessLevel _level;

    /// <summary>
    /// Defines the aggressiveness level of stop word removal.
    /// </summary>
    public enum AggressivenessLevel
    {
        /// <summary>
        /// Removes only basic articles and auxiliaries.
        /// </summary>
        Light,

        /// <summary>
        /// Removes common stop words (default).
        /// </summary>
        Medium,

        /// <summary>
        /// Removes most stop words including prepositions.
        /// </summary>
        Aggressive
    }

    /// <summary>
    /// Initializes a new instance of the StopWordRemovalCompressor class.
    /// </summary>
    /// <param name="level">The aggressiveness level for stop word removal.</param>
    /// <param name="tokenCounter">Optional custom token counter function.</param>
    public StopWordRemovalCompressor(
        AggressivenessLevel level = AggressivenessLevel.Medium,
        Func<string, int>? tokenCounter = null)
        : base("StopWordRemoval", tokenCounter)
    {
        _level = level;
        _stopWords = InitializeStopWords(level);
    }

    /// <summary>
    /// Initializes the set of stop words based on aggressiveness level.
    /// </summary>
    private static HashSet<string> InitializeStopWords(AggressivenessLevel level)
    {
        var lightWords = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "a", "an", "the",
            "is", "are", "was", "were", "be", "been", "being",
            "do", "does", "did", "doing",
            "have", "has", "had", "having"
        };

        if (level == AggressivenessLevel.Light)
        {
            return lightWords;
        }

        var mediumWords = new HashSet<string>(lightWords, StringComparer.OrdinalIgnoreCase)
        {
            // Pronouns
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
            "you", "your", "yours", "yourself", "yourselves",
            "he", "him", "his", "himself",
            "she", "her", "hers", "herself",
            "it", "its", "itself",
            "they", "them", "their", "theirs", "themselves",

            // Conjunctions
            "and", "but", "or", "nor", "for", "yet", "so",

            // Common words
            "this", "that", "these", "those",
            "am", "will", "would", "could", "should", "can", "may", "might", "must",
            "just", "very", "really", "actually", "basically", "simply"
        };

        if (level == AggressivenessLevel.Medium)
        {
            return mediumWords;
        }

        // Aggressive level
        var aggressiveWords = new HashSet<string>(mediumWords, StringComparer.OrdinalIgnoreCase)
        {
            // Prepositions
            "in", "on", "at", "to", "from", "of", "with", "by",
            "into", "onto", "upon", "out", "off",
            "up", "down", "over", "under", "above", "below",
            "through", "between", "among", "during", "before", "after",
            "about", "around", "against", "along", "across",

            // More auxiliaries and modals
            "shall", "need", "ought", "used",

            // Determiners
            "some", "any", "each", "every", "all", "both", "neither", "either",
            "few", "many", "much", "more", "most", "less", "least",

            // Adverbs
            "here", "there", "where", "when", "how", "why", "what", "which", "who",
            "not", "no", "yes", "also", "only", "then", "than", "now"
        };

        return aggressiveWords;
    }

    /// <summary>
    /// Compresses the prompt by removing stop words.
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

        // Split into words while preserving structure
        var lines = result.Split(new[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);
        var processedLines = new List<string>();

        foreach (var line in lines)
        {
            var words = RegexHelper.Split(line, @"(\s+|[.,!?;:])", RegexOptions.None);
            var filteredWords = new List<string>();

            foreach (var word in words)
            {
                var trimmedWord = word.Trim();

                // Keep punctuation and whitespace
                if (string.IsNullOrWhiteSpace(trimmedWord) || RegexHelper.IsMatch(trimmedWord, @"^[.,!?;:\s]+$", RegexOptions.None))
                {
                    filteredWords.Add(word);
                    continue;
                }

                // Keep placeholders
                if (trimmedWord.StartsWith("__VAR_") || trimmedWord.StartsWith("__CODE_"))
                {
                    filteredWords.Add(word);
                    continue;
                }

                // Check if it's a stop word
                if (!_stopWords.Contains(trimmedWord))
                {
                    filteredWords.Add(word);
                }
            }

            var processedLine = string.Join("", filteredWords);
            processedLine = RegexHelper.Replace(processedLine, @"\s{2,}", " ", RegexOptions.None).Trim();

            if (!string.IsNullOrWhiteSpace(processedLine))
            {
                processedLines.Add(processedLine);
            }
        }

        result = string.Join("\n", processedLines);

        // Clean up multiple punctuation
        result = RegexHelper.Replace(result, @"([.,!?;:])\s*\1+", "$1", RegexOptions.None);

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



