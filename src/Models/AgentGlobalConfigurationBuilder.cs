using AiDotNet.Agents;
using AiDotNet.Enums;

namespace AiDotNet.Models;

/// <summary>
/// Builder for global agent configuration.
/// </summary>
public class AgentGlobalConfigurationBuilder
{
    private readonly Dictionary<LLMProvider, string> _keys = new();
    private LLMProvider _defaultProvider = LLMProvider.OpenAI;

    public AgentGlobalConfigurationBuilder ConfigureOpenAI(string apiKey)
    {
        _keys[LLMProvider.OpenAI] = apiKey;
        return this;
    }

    public AgentGlobalConfigurationBuilder ConfigureAnthropic(string apiKey)
    {
        _keys[LLMProvider.Anthropic] = apiKey;
        return this;
    }

    public AgentGlobalConfigurationBuilder ConfigureAzureOpenAI(string apiKey)
    {
        _keys[LLMProvider.AzureOpenAI] = apiKey;
        return this;
    }

    public AgentGlobalConfigurationBuilder UseDefaultProvider(LLMProvider provider)
    {
        _defaultProvider = provider;
        return this;
    }

    internal void Apply()
    {
        foreach (var (provider, key) in _keys)
        {
            AgentGlobalConfiguration.SetApiKey(provider, key);
        }
        AgentGlobalConfiguration.DefaultProvider = _defaultProvider;
    }
}
