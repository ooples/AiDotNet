namespace AiDotNet.Agents;

/// <summary>
/// Configuration for AI agent assistance during model building.
/// </summary>
/// <typeparam name="T">The numeric type used for model parameters.</typeparam>
public class AgentConfiguration<T>
{
    /// <summary>
    /// The API key for the LLM provider. Can be null if using environment variables.
    /// </summary>
    public string? ApiKey { get; set; }

    /// <summary>
    /// The LLM provider to use (OpenAI, Anthropic, Azure).
    /// </summary>
    public LLMProvider Provider { get; set; } = LLMProvider.OpenAI;

    /// <summary>
    /// Whether agent assistance is enabled.
    /// </summary>
    public bool IsEnabled { get; set; }

    /// <summary>
    /// Azure OpenAI endpoint (only needed if Provider is AzureOpenAI).
    /// </summary>
    public string? AzureEndpoint { get; set; }

    /// <summary>
    /// Azure OpenAI deployment name (only needed if Provider is AzureOpenAI).
    /// </summary>
    public string? AzureDeployment { get; set; }
}
