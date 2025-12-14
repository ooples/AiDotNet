namespace AiDotNet.PromptEngineering.Analysis;

/// <summary>
/// Analyzer that provides accurate token counting and cost estimation for prompts.
/// </summary>
/// <remarks>
/// <para>
/// This analyzer focuses on token counting and cost estimation, supporting various
/// tokenization methods and model-specific pricing.
/// </para>
/// <para><b>For Beginners:</b> Counts how many tokens your prompt uses and estimates costs.
///
/// Example:
/// <code>
/// var analyzer = new TokenCountAnalyzer("gpt-4");
/// var metrics = analyzer.Analyze("Explain quantum computing in simple terms");
///
/// Console.WriteLine($"Tokens: {metrics.TokenCount}");
/// Console.WriteLine($"Estimated cost: ${metrics.EstimatedCost}");
/// </code>
///
/// Supports different models with their pricing:
/// - GPT-4: $0.03/1K tokens
/// - GPT-3.5-Turbo: $0.001/1K tokens
/// - Claude: $0.008/1K tokens
/// </para>
/// </remarks>
public class TokenCountAnalyzer : PromptAnalyzerBase
{
    private static readonly Dictionary<string, decimal> ModelPricing = new(StringComparer.OrdinalIgnoreCase)
    {
        { "gpt-4", 0.03m },
        { "gpt-4-turbo", 0.01m },
        { "gpt-4o", 0.005m },
        { "gpt-3.5-turbo", 0.001m },
        { "gpt-3.5-turbo-16k", 0.003m },
        { "claude-3-opus", 0.015m },
        { "claude-3-sonnet", 0.003m },
        { "claude-3-haiku", 0.00025m },
        { "claude-2", 0.008m },
        { "gemini-pro", 0.00025m },
        { "gemini-1.5-pro", 0.00125m }
    };

    /// <summary>
    /// Initializes a new instance of the TokenCountAnalyzer class.
    /// </summary>
    /// <param name="modelName">The target model for token counting.</param>
    /// <param name="tokenCounter">Optional custom token counter function.</param>
    public TokenCountAnalyzer(string modelName = "gpt-4", Func<string, int>? tokenCounter = null)
        : base(
            "TokenCountAnalyzer",
            modelName,
            GetModelPrice(modelName),
            tokenCounter)
    {
    }

    /// <summary>
    /// Gets the price per 1000 tokens for a given model.
    /// </summary>
    /// <param name="modelName">The model name.</param>
    /// <returns>The price per 1000 tokens in USD.</returns>
    public static decimal GetModelPrice(string modelName)
    {
        if (ModelPricing.TryGetValue(modelName, out var price))
        {
            return price;
        }

        // Try partial match
        foreach (var kvp in ModelPricing)
        {
            if (modelName.Contains(kvp.Key, StringComparison.OrdinalIgnoreCase))
            {
                return kvp.Value;
            }
        }

        // Default to GPT-4 pricing
        return 0.03m;
    }

    /// <summary>
    /// Gets the list of supported models.
    /// </summary>
    /// <returns>A list of model names with known pricing.</returns>
    public static IReadOnlyList<string> GetSupportedModels()
    {
        return ModelPricing.Keys.ToList().AsReadOnly();
    }

    /// <summary>
    /// Creates an analyzer pre-configured for GPT-4.
    /// </summary>
    public static TokenCountAnalyzer ForGpt4() => new("gpt-4");

    /// <summary>
    /// Creates an analyzer pre-configured for GPT-3.5-Turbo.
    /// </summary>
    public static TokenCountAnalyzer ForGpt35Turbo() => new("gpt-3.5-turbo");

    /// <summary>
    /// Creates an analyzer pre-configured for Claude.
    /// </summary>
    public static TokenCountAnalyzer ForClaude() => new("claude-3-sonnet");

    /// <summary>
    /// Creates an analyzer pre-configured for Gemini.
    /// </summary>
    public static TokenCountAnalyzer ForGemini() => new("gemini-pro");
}
