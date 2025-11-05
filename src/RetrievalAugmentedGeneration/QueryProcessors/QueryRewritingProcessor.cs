using AiDotNet.Interfaces;

namespace AiDotNet.RetrievalAugmentedGeneration.QueryProcessors;

/// <summary>
/// Rewrites queries for clarity and completeness, especially in conversational contexts.
/// </summary>
/// <typeparam name="T">The numeric data type for computations.</typeparam>
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

        rewritten = rewritten.Replace(" r ", " are ");
        rewritten = rewritten.Replace(" u ", " you ");
        rewritten = rewritten.Replace("wht ", "what ");
        rewritten = rewritten.Replace("hw ", "how ");
        rewritten = rewritten.Replace(" w/ ", " with ");
        rewritten = rewritten.Replace(" w/o ", " without ");

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
            var historyContext = string.Join("\n", _conversationHistory.TakeLast(3).Select((q, i) => $"{i + 1}. {q}"));
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
                return query.Replace("what about", $"what about {topic} and", StringComparison.OrdinalIgnoreCase)
                           .Replace("how about", $"how about {topic} and", StringComparison.OrdinalIgnoreCase);
            }
        }

        return query;
    }

    private static string ExtractTopic(string query)
    {
        var words = query.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
        
        if (words.Length > 2)
        {
            return words[words.Length - 1];
        }

        return string.Empty;
    }
}
