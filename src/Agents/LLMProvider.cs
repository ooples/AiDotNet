namespace AiDotNet.Agents;

/// <summary>
/// Supported LLM providers for agent assistance.
/// </summary>
public enum LLMProvider
{
    /// <summary>
    /// OpenAI GPT models (GPT-3.5, GPT-4, GPT-4-turbo, GPT-4o).
    /// </summary>
    OpenAI,

    /// <summary>
    /// Anthropic Claude models (Claude 2, Claude 3 family).
    /// </summary>
    Anthropic,

    /// <summary>
    /// Azure-hosted OpenAI models with enterprise features.
    /// </summary>
    AzureOpenAI
}
