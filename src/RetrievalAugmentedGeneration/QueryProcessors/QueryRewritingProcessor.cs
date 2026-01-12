using System.Linq;
using System.Text.RegularExpressions;
using AiDotNet.Interfaces;

namespace AiDotNet.RetrievalAugmentedGeneration.QueryProcessors;

/// <summary>
/// Rewrites queries for clarity and completeness, especially in conversational contexts.
/// </summary>
/// <typeparam name="T">The numeric data type for generator operations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// This processor transforms conversational or context-dependent queries into standalone,
/// clear questions. It's particularly useful in multi-turn conversations where queries
/// reference previous context.
/// </para>
/// <para><b>For Beginners:</b> Makes incomplete questions complete by adding missing context.
/// 
/// Conversational Examples:
/// - User: "Tell me about transformers"
/// - User: "What about their applications?" → "What are the applications of transformers?"
/// 
/// Clarity Examples:
/// - "how r cars made" → "how are cars manufactured"
/// - "wht is AI" → "what is artificial intelligence"
/// 
/// This makes your searches clearer and gets better results!
/// </para>
/// </remarks>
public class QueryRewritingProcessor<T> : QueryProcessorBase
{
    private readonly IGenerator<T>? _llmGenerator;
    private readonly List<string> _conversationHistory;

    /// <summary>
    /// Initializes a new instance of the QueryRewritingProcessor class.
    /// </summary>
    /// <param name="llmGenerator">Optional LLM generator for sophisticated query rewriting.</param>
    public QueryRewritingProcessor(IGenerator<T>? llmGenerator = null)
    {
        _llmGenerator = llmGenerator;
        _conversationHistory = new List<string>();
    }

    /// <summary>
    /// Adds a query to the conversation history for context-aware rewriting.
    /// </summary>
    /// <param name="query">The query to add to history.</param>
    public void AddToHistory(string query)
    {
        if (!string.IsNullOrWhiteSpace(query))
        {
            _conversationHistory.Add(query);

            if (_conversationHistory.Count > 5)
            {
                _conversationHistory.RemoveAt(0);
            }
        }
    }

    protected override string ProcessQueryCore(string query)
    {
        if (string.IsNullOrWhiteSpace(query))
            return query;

        var processedQuery = ApplyBasicRewrites(query);

        if (_llmGenerator != null && _conversationHistory.Count > 0)
        {
            processedQuery = RewriteWithContext(processedQuery);
        }

        AddToHistory(processedQuery);
        return processedQuery;
    }

    private string ApplyBasicRewrites(string query)
    {
        var rewritten = query;

        // Case-insensitive replacements for common text speak
        rewritten = RegexHelper.Replace(rewritten, @"\br\b", "are", RegexOptions.IgnoreCase);
        rewritten = RegexHelper.Replace(rewritten, @"\bu\b", "you", RegexOptions.IgnoreCase);
        rewritten = RegexHelper.Replace(rewritten, @"\bwht\b", "what", RegexOptions.IgnoreCase);
        rewritten = RegexHelper.Replace(rewritten, @"\bhw\b", "how", RegexOptions.IgnoreCase);
        rewritten = RegexHelper.Replace(rewritten, @" w/ ", " with ", RegexOptions.IgnoreCase);
        rewritten = RegexHelper.Replace(rewritten, @" w/o ", " without ", RegexOptions.IgnoreCase);

        return rewritten.Trim();
    }

    private string RewriteWithContext(string query)
    {
        var contextualKeywords = new[] { "it", "they", "them", "their", "that", "this", "these", "those", "what about", "how about" };

        var lowerQuery = query.ToLowerInvariant();
        var needsContext = contextualKeywords.Any(keyword => lowerQuery.Contains(keyword));

        if (!needsContext || _conversationHistory.Count == 0)
            return query;

        // Use LLM for contextual rewriting if available
        if (_llmGenerator != null)
        {
            var historyContext = string.Join("\n", _conversationHistory.Skip(Math.Max(0, _conversationHistory.Count - 3)).Select((q, i) => $"{i + 1}. {q}"));
            var prompt = $@"Given the conversation history:
{historyContext}

Rewrite the following query to be self-contained by resolving any pronouns or contextual references:
Query: {query}

Rewritten query:";

            var rewritten = _llmGenerator.Generate(prompt);
            return string.IsNullOrWhiteSpace(rewritten) ? query : rewritten.Trim();
        }

        // Fallback to rule-based rewriting
        var lastContext = _conversationHistory[_conversationHistory.Count - 1];

        if (lowerQuery.StartsWith("what about") || lowerQuery.StartsWith("how about"))
        {
            var topic = ExtractTopic(lastContext);
            if (!string.IsNullOrEmpty(topic))
            {
                // Anchor pattern to match only at the start of the query
                var result = RegexHelper.Replace(query, @"^what about\b", $"what about {topic} and", System.Text.RegularExpressions.RegexOptions.IgnoreCase);
                result = RegexHelper.Replace(result, @"^how about\b", $"how about {topic} and", System.Text.RegularExpressions.RegexOptions.IgnoreCase);
                return result;
            }
        }

        return query;
    }

    private static string ExtractTopic(string query)
    {
        if (string.IsNullOrWhiteSpace(query))
            return string.Empty;

        // Remove common question words and punctuation
        var commonWords = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "what", "is", "are", "the", "a", "an", "how", "why", "when", "where", "who",
            "do", "does", "did", "can", "could", "would", "should", "will", "about", "tell",
            "me", "you", "your", "it", "they", "them", "this", "that", "these", "those",
            "for", "in", "on", "at", "to", "from", "with", "of", "by", "as"
        };

        var words = query.Split(new[] { ' ', '?', '.', ',', '!', ';', ':' }, StringSplitOptions.RemoveEmptyEntries)
            .Where(w => !commonWords.Contains(w) && w.Length > 2)
            .ToList();

        if (words.Count == 0)
            return string.Empty;

        // For multi-word topics, take the first significant noun phrase (first 1-3 content words)
        // This handles cases like "transformers", "neural networks", "machine learning models"
        if (words.Count >= 2)
        {
            // Check if the first two words form a compound term (capitalized or technical terms)
            var firstTwo = string.Join(" ", words.Take(2));
            if (words[0].Length > 3 && words[1].Length > 3)
                return firstTwo;
        }

        // Return the first significant word as the topic
        return words[0];
    }
}



