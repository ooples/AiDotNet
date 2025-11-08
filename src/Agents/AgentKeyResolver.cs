using AiDotNet.Enums;
using AiDotNet.Models;

namespace AiDotNet.Agents;

/// <summary>
/// Resolves API keys from multiple sources with priority ordering.
/// </summary>
public static class AgentKeyResolver
{
    /// <summary>
    /// Resolves API key in this priority order:
    /// 1. Explicit parameter (if provided)
    /// 2. Stored in AgentConfiguration (from build phase)
    /// 3. Global configuration (if set)
    /// 4. Environment variable
    /// 5. Throw exception if none found
    /// </summary>
    public static string ResolveApiKey<T>(
        string? explicitKey = null,
        AgentConfiguration<T>? storedConfig = null,
        LLMProvider provider = LLMProvider.OpenAI)
    {
        // 1. Explicit parameter takes highest priority
        if (!string.IsNullOrWhiteSpace(explicitKey))
            return explicitKey;

        // 2. Check stored config from build phase
        if (storedConfig?.ApiKey != null)
            return storedConfig.ApiKey;

        // 3. Check global configuration
        if (AgentGlobalConfiguration.ApiKeys.TryGetValue(provider, out var globalKey) && !string.IsNullOrWhiteSpace(globalKey))
            return globalKey;

        // 4. Check environment variables
        var envVarName = provider switch
        {
            LLMProvider.OpenAI => "OPENAI_API_KEY",
            LLMProvider.Anthropic => "ANTHROPIC_API_KEY",
            LLMProvider.AzureOpenAI => "AZURE_OPENAI_KEY",
            _ => null
        };

        if (envVarName != null)
        {
            var envKey = Environment.GetEnvironmentVariable(envVarName);
            if (!string.IsNullOrWhiteSpace(envKey))
                return envKey;
        }

        // 5. No key found - throw helpful error
        throw new InvalidOperationException(
            $"No API key found for {provider}. Please provide via:\n" +
            $"1. Explicit parameter: .ConfigureAgentAssistance(apiKey: \"...\")\n" +
            $"2. Global config: AgentGlobalConfiguration.Configure(...)\n" +
            $"3. Environment variable: {envVarName}");
    }
}
