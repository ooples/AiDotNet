using AiDotNet.Interfaces;

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

/// <summary>
/// Options for customizing what the agent helps with during building.
/// </summary>
public class AgentAssistanceOptions
{
    /// <summary>
    /// Enable agent to analyze data characteristics.
    /// </summary>
    public bool EnableDataAnalysis { get; set; } = true;

    /// <summary>
    /// Enable agent to suggest model types if not already configured.
    /// </summary>
    public bool EnableModelSelection { get; set; } = true;

    /// <summary>
    /// Enable agent to recommend hyperparameter values.
    /// </summary>
    public bool EnableHyperparameterTuning { get; set; } = false;

    /// <summary>
    /// Enable agent to analyze feature importance and suggest feature selection.
    /// </summary>
    public bool EnableFeatureAnalysis { get; set; } = false;

    /// <summary>
    /// Enable agent to provide advice for meta-learning configurations.
    /// </summary>
    public bool EnableMetaLearningAdvice { get; set; } = false;

    /// <summary>
    /// Default options: data analysis and model selection enabled.
    /// </summary>
    public static AgentAssistanceOptions Default => new AgentAssistanceOptions
    {
        EnableDataAnalysis = true,
        EnableModelSelection = true,
        EnableHyperparameterTuning = false,
        EnableFeatureAnalysis = false,
        EnableMetaLearningAdvice = false
    };

    /// <summary>
    /// Minimal options: only model selection enabled.
    /// </summary>
    public static AgentAssistanceOptions Minimal => new AgentAssistanceOptions
    {
        EnableDataAnalysis = false,
        EnableModelSelection = true,
        EnableHyperparameterTuning = false,
        EnableFeatureAnalysis = false,
        EnableMetaLearningAdvice = false
    };

    /// <summary>
    /// Comprehensive options: everything enabled.
    /// </summary>
    public static AgentAssistanceOptions Comprehensive => new AgentAssistanceOptions
    {
        EnableDataAnalysis = true,
        EnableModelSelection = true,
        EnableHyperparameterTuning = true,
        EnableFeatureAnalysis = true,
        EnableMetaLearningAdvice = true
    };

    /// <summary>
    /// Creates a new options builder for fluent configuration.
    /// </summary>
    public static AgentAssistanceOptionsBuilder Create() => new AgentAssistanceOptionsBuilder();
}

/// <summary>
/// Fluent builder for agent assistance options.
/// </summary>
public class AgentAssistanceOptionsBuilder
{
    private readonly AgentAssistanceOptions _options = new AgentAssistanceOptions
    {
        // Start with nothing enabled
        EnableDataAnalysis = false,
        EnableModelSelection = false,
        EnableHyperparameterTuning = false,
        EnableFeatureAnalysis = false,
        EnableMetaLearningAdvice = false
    };

    public AgentAssistanceOptionsBuilder EnableDataAnalysis()
    {
        _options.EnableDataAnalysis = true;
        return this;
    }

    public AgentAssistanceOptionsBuilder EnableModelSelection()
    {
        _options.EnableModelSelection = true;
        return this;
    }

    public AgentAssistanceOptionsBuilder EnableHyperparameterTuning()
    {
        _options.EnableHyperparameterTuning = true;
        return this;
    }

    public AgentAssistanceOptionsBuilder EnableFeatureAnalysis()
    {
        _options.EnableFeatureAnalysis = true;
        return this;
    }

    public AgentAssistanceOptionsBuilder EnableMetaLearningAdvice()
    {
        _options.EnableMetaLearningAdvice = true;
        return this;
    }

    public AgentAssistanceOptionsBuilder DisableDataAnalysis()
    {
        _options.EnableDataAnalysis = false;
        return this;
    }

    public AgentAssistanceOptionsBuilder DisableModelSelection()
    {
        _options.EnableModelSelection = false;
        return this;
    }

    public AgentAssistanceOptionsBuilder DisableHyperparameterTuning()
    {
        _options.EnableHyperparameterTuning = false;
        return this;
    }

    public AgentAssistanceOptionsBuilder DisableFeatureAnalysis()
    {
        _options.EnableFeatureAnalysis = false;
        return this;
    }

    public AgentAssistanceOptionsBuilder DisableMetaLearningAdvice()
    {
        _options.EnableMetaLearningAdvice = false;
        return this;
    }

    public AgentAssistanceOptions Build() => _options;

    public static implicit operator AgentAssistanceOptions(AgentAssistanceOptionsBuilder builder)
        => builder._options;
}

/// <summary>
/// Contains the agent's recommendations after analyzing the data and configuration.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public class AgentRecommendation<T, TInput, TOutput>
{
    /// <summary>
    /// Data analysis insights from the agent.
    /// </summary>
    public string? DataAnalysis { get; set; }

    /// <summary>
    /// Suggested model type (e.g., "RidgeRegression", "RandomForest").
    /// </summary>
    public string? SuggestedModelType { get; set; }

    /// <summary>
    /// Agent's reasoning for model selection.
    /// </summary>
    public string? ModelSelectionReasoning { get; set; }

    /// <summary>
    /// Suggested hyperparameter values.
    /// </summary>
    public Dictionary<string, object>? SuggestedHyperparameters { get; set; }

    /// <summary>
    /// Agent's reasoning for hyperparameter tuning.
    /// </summary>
    public string? TuningReasoning { get; set; }

    /// <summary>
    /// Feature selection and importance recommendations.
    /// </summary>
    public string? FeatureRecommendations { get; set; }

    /// <summary>
    /// Complete reasoning trace from all agent operations.
    /// </summary>
    public string? ReasoningTrace { get; set; }
}

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
            $"1. Explicit parameter: .WithAgentAssistance(apiKey: \"...\")\n" +
            $"2. Global config: AgentGlobalConfiguration.Configure(...)\n" +
            $"3. Environment variable: {envVarName}");
    }
}

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

/// <summary>
/// Builder for global agent configuration.
/// </summary>
public class AgentGlobalConfigurationBuilder
{
    private readonly Dictionary<LLMProvider, string> _keys = new();
    private LLMProvider _defaultProvider = LLMProvider.OpenAI;

    public AgentGlobalConfigurationBuilder WithOpenAI(string apiKey)
    {
        _keys[LLMProvider.OpenAI] = apiKey;
        return this;
    }

    public AgentGlobalConfigurationBuilder WithAnthropic(string apiKey)
    {
        _keys[LLMProvider.Anthropic] = apiKey;
        return this;
    }

    public AgentGlobalConfigurationBuilder WithAzureOpenAI(string apiKey)
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
