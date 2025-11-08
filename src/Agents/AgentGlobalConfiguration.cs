namespace AiDotNet.Agents;

/// <summary>
/// Global configuration for agent assistance across the application.
/// </summary>
public static class AgentGlobalConfiguration
{
    private static readonly Dictionary<LLMProvider, string> _apiKeys = new();

    /// <summary>
    /// Configured API keys by provider.
    /// </summary>
    public static IReadOnlyDictionary<LLMProvider, string> ApiKeys => _apiKeys;

    /// <summary>
    /// Default LLM provider to use if not specified.
    /// </summary>
    public static LLMProvider DefaultProvider { get; set; } = LLMProvider.OpenAI;

    /// <summary>
    /// Configures global agent settings.
    /// </summary>
    /// <param name="configure">Action to configure settings.</param>
    public static void Configure(Action<AgentGlobalConfigurationBuilder> configure)
    {
        var builder = new AgentGlobalConfigurationBuilder();
        configure(builder);
        builder.Apply();
    }

    internal static void SetApiKey(LLMProvider provider, string apiKey)
    {
        _apiKeys[provider] = apiKey;
    }
}
