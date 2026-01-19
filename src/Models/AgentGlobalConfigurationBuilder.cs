using AiDotNet.Agents;
using AiDotNet.Enums;

namespace AiDotNet.Models;

/// <summary>
/// Provides a fluent interface for configuring global AI agent settings including API keys and default providers.
/// </summary>
/// <remarks>
/// <para>
/// This builder class is used internally by AgentGlobalConfiguration.Configure() to provide a type-safe, fluent
/// API for setting up global agent configuration. It allows you to configure API keys for multiple LLM providers
/// (OpenAI, Anthropic, Azure OpenAI) and specify which provider should be used by default. The builder pattern
/// enables method chaining for clean, readable configuration code. Once all settings are configured, the Apply()
/// method transfers them to the AgentGlobalConfiguration static class where they're stored for use across the application.
/// </para>
/// <para><b>For Beginners:</b> This class helps you set up your AI agent credentials using a clean, readable syntax.
///
/// You don't create this class directly - instead, you receive it through AgentGlobalConfiguration.Configure():
/// <code>
/// AgentGlobalConfiguration.Configure(builder => builder
///     .ConfigureOpenAI("your-api-key")
///     .UseDefaultProvider(LLMProvider.OpenAI));
/// </code>
///
/// The builder pattern:
/// - Makes configuration code easy to read
/// - Lets you chain multiple setup calls together
/// - Provides IntelliSense (autocomplete) to guide you
/// - Ensures all settings are valid before they're applied
///
/// Think of it as a temporary workspace where you specify all your settings. Once you're done configuring,
/// the builder transfers everything to the global configuration where it's stored and used by your models.
///
/// For example:
/// <code>
/// // At application startup
/// AgentGlobalConfiguration.Configure(config => config
///     .ConfigureOpenAI("sk-...")           // Set OpenAI API key
///     .ConfigureAnthropic("sk-ant-...")    // Set Anthropic API key
///     .UseDefaultProvider(LLMProvider.OpenAI));  // Choose default
///
/// // All three methods return the builder, so they chain together
/// // When the lambda completes, Apply() is called automatically
/// // Your settings are now stored globally for all models to use
/// </code>
///
/// This approach keeps your configuration code organized and readable, especially when setting up multiple providers.
/// </para>
/// </remarks>
public class AgentGlobalConfigurationBuilder
{
    private readonly Dictionary<LLMProvider, string> _keys = new();
    private LLMProvider _defaultProvider = LLMProvider.OpenAI;

    /// <summary>
    /// Configures the API key for OpenAI's language models (GPT-3.5, GPT-4, etc.).
    /// </summary>
    /// <param name="apiKey">Your OpenAI API key, obtainable from platform.openai.com.</param>
    /// <returns>The current builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// This method sets the API key that will be used for all OpenAI API calls when agent assistance is enabled.
    /// The key should be obtained from your OpenAI account at platform.openai.com. Once configured, this key
    /// will be used automatically by any AiModelBuilder that enables agent assistance with the OpenAI provider,
    /// unless explicitly overridden at the builder level.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up your OpenAI credentials so the AI agent can use OpenAI's models.
    ///
    /// How to get an OpenAI API key:
    /// 1. Sign up at platform.openai.com
    /// 2. Navigate to API keys in your account settings
    /// 3. Click "Create new secret key"
    /// 4. Copy the key (it starts with "sk-") and save it securely
    /// 5. Use it here in your configuration
    ///
    /// Example:
    /// <code>
    /// AgentGlobalConfiguration.Configure(config => config
    ///     .ConfigureOpenAI("sk-proj-...your-key-here..."));
    /// </code>
    ///
    /// Security tips:
    /// - Never commit your API key to source control (use environment variables or secret management)
    /// - Don't share your key publicly or in screenshots
    /// - Rotate keys periodically for security
    /// - Use OpenAI's usage limits and monitoring to prevent unexpected charges
    ///
    /// Once configured, all your models can use OpenAI's GPT models for agent assistance without
    /// needing to provide the key again.
    /// </para>
    /// </remarks>
    public AgentGlobalConfigurationBuilder ConfigureOpenAI(string apiKey)
    {
        if (string.IsNullOrWhiteSpace(apiKey))
        {
            throw new ArgumentException("OpenAI API key cannot be null or whitespace.", nameof(apiKey));
        }

        _keys[LLMProvider.OpenAI] = apiKey;
        return this;
    }

    /// <summary>
    /// Configures the API key for Anthropic's Claude language models.
    /// </summary>
    /// <param name="apiKey">Your Anthropic API key, obtainable from console.anthropic.com.</param>
    /// <returns>The current builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// This method sets the API key that will be used for all Anthropic API calls when agent assistance is enabled.
    /// The key should be obtained from your Anthropic account at console.anthropic.com. Once configured, this key
    /// will be used automatically by any AiModelBuilder that enables agent assistance with the Anthropic provider,
    /// unless explicitly overridden at the builder level. Anthropic's Claude models are known for detailed explanations
    /// and large context windows.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up your Anthropic credentials so the AI agent can use Claude models.
    ///
    /// How to get an Anthropic API key:
    /// 1. Sign up at console.anthropic.com
    /// 2. Navigate to API keys in your account settings
    /// 3. Click to create a new key
    /// 4. Copy the key (it starts with "sk-ant-") and save it securely
    /// 5. Use it here in your configuration
    ///
    /// Example:
    /// <code>
    /// AgentGlobalConfiguration.Configure(config => config
    ///     .ConfigureAnthropic("sk-ant-...your-key-here..."));
    /// </code>
    ///
    /// Why choose Anthropic:
    /// - Claude models provide very detailed, thoughtful explanations
    /// - Large context windows can handle more information at once
    /// - Known for being particularly helpful and safe
    /// - Good for educational use cases where detailed reasoning matters
    ///
    /// Security tips:
    /// - Never commit your API key to source control
    /// - Use environment variables or secret management in production
    /// - Monitor your usage to control costs
    ///
    /// Once configured, all your models can use Anthropic's Claude for agent assistance.
    /// </para>
    /// </remarks>
    public AgentGlobalConfigurationBuilder ConfigureAnthropic(string apiKey)
    {
        if (string.IsNullOrWhiteSpace(apiKey))
        {
            throw new ArgumentException("Anthropic API key cannot be null or whitespace.", nameof(apiKey));
        }

        _keys[LLMProvider.Anthropic] = apiKey;
        return this;
    }

    /// <summary>
    /// Configures the API key for Azure OpenAI Service.
    /// </summary>
    /// <param name="apiKey">Your Azure OpenAI API key, obtainable from the Azure Portal.</param>
    /// <returns>The current builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// This method sets the API key that will be used for all Azure OpenAI API calls when agent assistance is enabled.
    /// The key should be obtained from your Azure OpenAI resource in the Azure Portal. Note that Azure OpenAI requires
    /// additional configuration beyond just the API key - you'll also need to specify the endpoint URL and deployment
    /// name when configuring individual models. Azure OpenAI provides the same models as OpenAI but with enterprise
    /// features like regional data residency, enhanced security, and integration with Azure services.
    /// </para>
    /// <para><b>For Beginners:</b> This sets up your Azure OpenAI credentials for enterprise-grade AI assistance.
    ///
    /// How to get an Azure OpenAI API key:
    /// 1. Create an Azure account and get approved for Azure OpenAI Service
    /// 2. Create an Azure OpenAI resource in the Azure Portal
    /// 3. Navigate to "Keys and Endpoint" in your resource
    /// 4. Copy one of the keys shown
    /// 5. Also note your endpoint URL - you'll need it later
    /// 6. Deploy a model (like GPT-4) and note the deployment name
    /// 7. Use the key here in your global configuration
    ///
    /// Example:
    /// <code>
    /// AgentGlobalConfiguration.Configure(config => config
    ///     .ConfigureAzureOpenAI("your-azure-key")
    ///     .UseDefaultProvider(LLMProvider.AzureOpenAI));
    /// </code>
    ///
    /// Why choose Azure OpenAI:
    /// - Enterprise-grade security and compliance (SOC 2, HIPAA, etc.)
    /// - Data stays in your specified Azure region
    /// - Integrated billing with other Azure services
    /// - Same powerful GPT models as OpenAI
    /// - Virtual network support and private endpoints
    ///
    /// Important notes:
    /// - Azure OpenAI requires approval (apply at azure.microsoft.com/products/cognitive-services/openai-service)
    /// - You need to specify endpoint and deployment name when building models
    /// - Pricing may differ from direct OpenAI API
    ///
    /// This is the preferred option for enterprise applications with strict compliance requirements.
    /// </para>
    /// </remarks>
    public AgentGlobalConfigurationBuilder ConfigureAzureOpenAI(string apiKey)
    {
        if (string.IsNullOrWhiteSpace(apiKey))
        {
            throw new ArgumentException("Azure OpenAI API key cannot be null or whitespace.", nameof(apiKey));
        }

        _keys[LLMProvider.AzureOpenAI] = apiKey;
        return this;
    }

    /// <summary>
    /// Sets which LLM provider should be used by default when one is not explicitly specified.
    /// </summary>
    /// <param name="provider">The LLMProvider to use as the default (OpenAI, Anthropic, or AzureOpenAI).</param>
    /// <returns>The current builder instance for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// This method specifies which LLM provider will be used by default across all model builders when agent
    /// assistance is enabled but no specific provider is chosen. This eliminates the need to specify the provider
    /// for each individual model, while still allowing overrides when needed. Choose the provider you'll use most
    /// frequently as your default. If not specified, OpenAI is used as the default.
    /// </para>
    /// <para><b>For Beginners:</b> This tells the system which AI service to use when you don't specify one.
    ///
    /// Setting a default provider:
    /// - Saves you from specifying the provider every time you build a model
    /// - Ensures consistency across your application
    /// - Can still be overridden for specific models when needed
    ///
    /// Example:
    /// <code>
    /// AgentGlobalConfiguration.Configure(config => config
    ///     .ConfigureOpenAI("sk-...")
    ///     .ConfigureAnthropic("sk-ant-...")
    ///     .UseDefaultProvider(LLMProvider.Anthropic));  // Use Anthropic by default
    /// </code>
    ///
    /// How to choose your default:
    /// - If you primarily use OpenAI: LLMProvider.OpenAI
    /// - If you primarily use Anthropic: LLMProvider.Anthropic
    /// - If you're in an enterprise Azure environment: LLMProvider.AzureOpenAI
    ///
    /// What happens:
    /// <code>
    /// // Uses your default provider (Anthropic in the example above)
    /// var result = await builder
    ///     .ConfigureAgentAssistance(options => options.EnableModelSelection())
    ///     .BuildAsync();
    ///
    /// // Override to use a different provider for this specific model
    /// var result2 = await builder
    ///     .ConfigureAgentAssistance(
    ///         options => options.EnableModelSelection(),
    ///         provider: LLMProvider.OpenAI)  // Use OpenAI just for this one
    ///     .BuildAsync();
    /// </code>
    ///
    /// Choose whichever provider best fits your needs, budget, and compliance requirements.
    /// </para>
    /// </remarks>
    public AgentGlobalConfigurationBuilder UseDefaultProvider(LLMProvider provider)
    {
        _defaultProvider = provider;
        return this;
    }

    /// <summary>
    /// Applies all configured settings to the global configuration (internal use only).
    /// </summary>
    /// <remarks>
    /// <para>
    /// This internal method transfers all configured API keys and the default provider setting from the builder
    /// to the AgentGlobalConfiguration static class where they're stored for use by the application. This method
    /// is called automatically by AgentGlobalConfiguration.Configure() after the configuration action completes.
    /// It should not be called directly by user code.
    /// </para>
    /// <para><b>For Beginners:</b> This method finalizes your configuration and stores it globally.
    ///
    /// You don't call this method yourself - it's called automatically when your configuration is complete:
    /// <code>
    /// AgentGlobalConfiguration.Configure(config => config
    ///     .ConfigureOpenAI("sk-...")
    ///     .UseDefaultProvider(LLMProvider.OpenAI));
    /// // Apply() is called automatically here, storing your settings
    /// </code>
    ///
    /// What it does:
    /// - Takes all the API keys you configured
    /// - Takes your default provider choice
    /// - Stores them in AgentGlobalConfiguration
    /// - Makes them available to all your model builders
    ///
    /// This happens behind the scenes, so you can focus on just configuring your settings
    /// without worrying about how they're stored.
    /// </para>
    /// </remarks>
    internal void Apply()
    {
        foreach (var kvp in _keys)
        {
            AgentGlobalConfiguration.SetApiKey(kvp.Key, kvp.Value);
        }
        AgentGlobalConfiguration.DefaultProvider = _defaultProvider;
    }
}
