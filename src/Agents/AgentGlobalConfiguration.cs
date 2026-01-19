using AiDotNet.Enums;
using AiDotNet.Models;

namespace AiDotNet.Agents;

/// <summary>
/// Provides application-wide configuration for AI agent assistance, including API keys and default provider settings.
/// </summary>
/// <remarks>
/// <para>
/// This static class manages global configuration settings for AI agent assistance across your entire application.
/// It stores API keys for different LLM providers (OpenAI, Anthropic, Azure OpenAI) and allows you to set a default
/// provider to use when one isn't explicitly specified. Configuration set through this class applies to all
/// AiModelBuilder instances unless overridden at the individual builder level. This centralized approach
/// eliminates the need to specify API keys for each model you build and ensures consistent settings across your
/// application.
/// </para>
/// <para><b>For Beginners:</b> This class lets you set up your AI agent credentials once at application startup,
/// so you don't have to provide them every time you build a model.
///
/// Think of it as your application's central AI configuration hub:
/// - **Set API keys once**: Configure credentials for all AI providers you want to use
/// - **Choose a default provider**: Pick which AI service to use by default (OpenAI, Anthropic, or Azure)
/// - **Use everywhere**: Every model builder automatically uses these settings
/// - **Override when needed**: Individual builders can still override these global settings
///
/// Why use global configuration:
/// - Convenience: Set credentials once instead of repeating them for every model
/// - Consistency: Ensures all models use the same AI provider and settings
/// - Security: Keeps API keys in one place, easier to manage and secure
/// - Flexibility: Can still override on a per-model basis when needed
///
/// For example, at application startup you might do:
/// <code>
/// AgentGlobalConfiguration.Configure(config => config
///     .ConfigureOpenAI("your-api-key")
///     .UseDefaultProvider(LLMProvider.OpenAI));
/// </code>
///
/// Then later when building models, the agent automatically uses those credentials:
/// <code>
/// var result = await new AiModelBuilder&lt;double&gt;()
///     .ConfigureAgentAssistance(options => options.EnableModelSelection())
///     .BuildAsync();  // Automatically uses global OpenAI configuration
/// </code>
///
/// This is especially useful in production applications where you want centralized credential management
/// and don't want to scatter API keys throughout your codebase.
/// </para>
/// </remarks>
public static class AgentGlobalConfiguration
{
    private static readonly Dictionary<LLMProvider, string> _apiKeys = new();
    private static readonly object _lock = new object();
    private static LLMProvider _defaultProvider = LLMProvider.OpenAI;

    /// <summary>
    /// Gets a read-only dictionary of configured API keys indexed by LLM provider.
    /// </summary>
    /// <value>A read-only dictionary mapping LLMProvider enum values to their corresponding API key strings.</value>
    /// <remarks>
    /// <para>
    /// This property provides read-only access to all API keys that have been configured through the Configure method.
    /// The dictionary is keyed by LLMProvider (OpenAI, Anthropic, AzureOpenAI) and contains the API key strings for
    /// each provider. Only providers that have been explicitly configured will have entries in this dictionary.
    /// This property is useful for checking which providers are configured or for diagnostic purposes, but cannot
    /// be used to modify the keys directly - use the Configure method instead.
    /// </para>
    /// <para><b>For Beginners:</b> This shows you which AI providers you've configured with API keys.
    ///
    /// You can check this property to see:
    /// - Which AI services have been set up (OpenAI, Anthropic, Azure)
    /// - Whether a specific provider has been configured
    /// - What API keys are currently in use (for debugging)
    ///
    /// For example:
    /// <code>
    /// if (AgentGlobalConfiguration.ApiKeys.ContainsKey(LLMProvider.OpenAI))
    /// {
    ///     Console.WriteLine("OpenAI is configured and ready to use");
    /// }
    /// </code>
    ///
    /// Note: This is read-only. To add or change API keys, use the Configure() method instead.
    ///
    /// Security reminder: Be careful when logging or displaying this property, as it contains sensitive API keys.
    /// Never commit code that logs or exposes these values in production.
    /// </para>
    /// </remarks>
    public static IReadOnlyDictionary<LLMProvider, string> ApiKeys
    {
        get
        {
            lock (_lock)
            {
                return new Dictionary<LLMProvider, string>(_apiKeys);
            }
        }
    }

    /// <summary>
    /// Gets or sets the default LLM provider to use when one is not explicitly specified.
    /// </summary>
    /// <value>An LLMProvider enum value indicating the default provider. Defaults to OpenAI.</value>
    /// <remarks>
    /// <para>
    /// This property determines which LLM provider will be used when agent assistance is enabled but no specific
    /// provider is specified in the model builder's AgentConfiguration. It provides a convenient way to set a
    /// consistent default across your application while still allowing individual builders to override this choice.
    /// The default value is LLMProvider.OpenAI. If you primarily use a different provider, setting this property
    /// eliminates the need to specify the provider for each model.
    /// </para>
    /// <para><b>For Beginners:</b> This sets which AI service to use by default when you don't specify one.
    ///
    /// The default provider:
    /// - Is used whenever you enable agent assistance without choosing a specific provider
    /// - Can be changed at any time (usually set once at application startup)
    /// - Defaults to OpenAI if you don't set it
    /// - Can be overridden for individual models if needed
    ///
    /// When to set this:
    /// - You primarily use one AI provider across your application
    /// - You want consistency in which AI service is used
    /// - You want to avoid specifying the provider every time
    ///
    /// For example:
    /// <code>
    /// // At application startup, set Anthropic as your default
    /// AgentGlobalConfiguration.DefaultProvider = LLMProvider.Anthropic;
    ///
    /// // Later, when building models, Anthropic is automatically used
    /// var result = await builder
    ///     .ConfigureAgentAssistance(options => options.EnableModelSelection())
    ///     .BuildAsync();  // Uses Anthropic by default
    ///
    /// // But you can still override for specific models
    /// var result2 = await builder
    ///     .ConfigureAgentAssistance(
    ///         options => options.EnableModelSelection(),
    ///         provider: LLMProvider.OpenAI)  // Override to use OpenAI
    ///     .BuildAsync();
    /// </code>
    ///
    /// Choose your default based on which provider you've configured and prefer to use most often.
    /// </para>
    /// </remarks>
    public static LLMProvider DefaultProvider
    {
        get
        {
            lock (_lock)
            {
                return _defaultProvider;
            }
        }
        set
        {
            lock (_lock)
            {
                _defaultProvider = value;
            }
        }
    }

    /// <summary>
    /// Configures global agent settings using a fluent builder pattern.
    /// </summary>
    /// <param name="configure">An action that receives a builder and configures the agent settings.</param>
    /// <remarks>
    /// <para>
    /// This method provides a fluent, type-safe way to configure global agent settings. It creates a
    /// AgentGlobalConfigurationBuilder instance, passes it to the provided action for configuration, then applies
    /// all the configured settings. This is the recommended way to set up global agent configuration as it provides
    /// a clean, readable syntax with IntelliSense support and compile-time checking.
    /// </para>
    /// <para><b>For Beginners:</b> This method lets you set up all your AI agent configuration in one place using
    /// a clean, readable syntax.
    ///
    /// How to use it:
    /// <code>
    /// AgentGlobalConfiguration.Configure(config => config
    ///     .ConfigureOpenAI("your-openai-api-key")
    ///     .ConfigureAnthropic("your-anthropic-api-key")
    ///     .UseDefaultProvider(LLMProvider.OpenAI));
    /// </code>
    ///
    /// The fluent builder pattern:
    /// - Provides a clean, readable way to configure multiple settings
    /// - Chains methods together for concise code
    /// - Gives you IntelliSense (autocomplete) for available options
    /// - Catches configuration errors at compile time
    ///
    /// Common configuration scenarios:
    ///
    /// 1. Configure a single provider:
    /// <code>
    /// AgentGlobalConfiguration.Configure(config => config
    ///     .ConfigureOpenAI("sk-..."));
    /// </code>
    ///
    /// 2. Configure multiple providers with a default:
    /// <code>
    /// AgentGlobalConfiguration.Configure(config => config
    ///     .ConfigureOpenAI("sk-...")
    ///     .ConfigureAnthropic("sk-ant-...")
    ///     .UseDefaultProvider(LLMProvider.Anthropic));
    /// </code>
    ///
    /// 3. Configure Azure OpenAI:
    /// <code>
    /// AgentGlobalConfiguration.Configure(config => config
    ///     .ConfigureAzureOpenAI("your-azure-key")
    ///     .UseDefaultProvider(LLMProvider.AzureOpenAI));
    /// </code>
    ///
    /// Best practice: Call this method once at application startup (e.g., in your Main method or Startup.cs)
    /// to set up credentials for all subsequent model building operations.
    /// </para>
    /// </remarks>
    public static void Configure(Action<AgentGlobalConfigurationBuilder> configure)
    {
        var builder = new AgentGlobalConfigurationBuilder();
        configure(builder);
        builder.Apply();
    }

    /// <summary>
    /// Sets the API key for a specific LLM provider (internal use by builder).
    /// </summary>
    /// <param name="provider">The LLM provider for which to set the API key.</param>
    /// <param name="apiKey">The API key string for the specified provider.</param>
    /// <remarks>
    /// This internal method is called by the AgentGlobalConfigurationBuilder to store configured API keys.
    /// It is not intended for direct use - use the Configure() method with the fluent builder instead.
    /// Thread-safe for concurrent access.
    /// </remarks>
    internal static void SetApiKey(LLMProvider provider, string apiKey)
    {
        lock (_lock)
        {
            _apiKeys[provider] = apiKey;
        }
    }
}
