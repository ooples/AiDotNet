using AiDotNet.Enums;
using AiDotNet.Models;

namespace AiDotNet.Agents;

/// <summary>
/// Provides intelligent API key resolution by searching multiple configuration sources in priority order.
/// </summary>
/// <remarks>
/// <para>
/// This static utility class resolves API keys for LLM providers by checking multiple configuration sources in
/// a well-defined priority order. It enables flexible credential management by supporting explicit parameters,
/// stored configurations, global application settings, and environment variables. The priority system ensures that
/// more specific configurations (like explicit parameters) override more general ones (like environment variables),
/// giving developers fine-grained control while maintaining convenient defaults. If no API key can be found in any
/// source, the resolver throws a detailed exception explaining all available configuration options.
/// </para>
/// <para><b>For Beginners:</b> This class automatically finds your API keys by checking several places in order.
///
/// Think of it as a smart key finder that looks for your API key in multiple locations:
/// 1. **Explicit parameter**: Key provided directly in your code (highest priority)
/// 2. **Stored configuration**: Key saved during model building
/// 3. **Global configuration**: Key set at application startup via AgentGlobalConfiguration
/// 4. **Environment variable**: Key stored in system environment variables (most secure)
///
/// Why this is useful:
/// - **Flexibility**: You can provide keys in different ways for different situations
/// - **Security**: Supports environment variables which keeps keys out of your code
/// - **Convenience**: Set once globally and use everywhere
/// - **Override capability**: Can override global settings for specific models when needed
/// - **Clear errors**: If no key is found, you get helpful instructions on how to fix it
///
/// For example, in development you might use:
/// <code>
/// // Option 1: Direct in code (quick for testing, not secure for production)
/// var result = await builder
///     .ConfigureAgentAssistance(
///         options => options.EnableModelSelection(),
///         apiKey: "sk-test-key")
///     .BuildAsync();
///
/// // Option 2: Global configuration (set once at startup)
/// AgentGlobalConfiguration.Configure(config => config
///     .ConfigureOpenAI("your-api-key"));
///
/// // Option 3: Environment variable (most secure for production)
/// // Set OPENAI_API_KEY=your-key in your environment
/// // Then no code changes needed, just:
/// var result = await builder
///     .ConfigureAgentAssistance(options => options.EnableModelSelection())
///     .BuildAsync();
/// </code>
///
/// The resolver automatically checks all these locations and uses the first key it finds, following the
/// priority order. This means you can set a global default but override it for specific models when needed.
/// </para>
/// </remarks>
public static class AgentKeyResolver
{
    /// <summary>
    /// Resolves an API key for the specified LLM provider by checking multiple sources in priority order.
    /// </summary>
    /// <typeparam name="T">The numeric type used in the agent configuration.</typeparam>
    /// <param name="explicitKey">An explicitly provided API key (highest priority), or null to check other sources.</param>
    /// <param name="storedConfig">A stored AgentConfiguration from the build phase, or null if not available.</param>
    /// <param name="provider">The LLM provider for which to resolve the API key. Defaults to OpenAI.</param>
    /// <returns>The resolved API key string from the highest-priority available source.</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when no API key can be found in any source. The exception message provides detailed instructions
    /// on how to configure the API key using each available method.
    /// </exception>
    /// <remarks>
    /// <para>
    /// This method searches for API keys in the following priority order, returning the first valid key found:
    ///
    /// 1. **Explicit parameter**: If explicitKey is provided and non-empty, it's used immediately
    /// 2. **Stored configuration**: If storedConfig contains an API key, it's used
    /// 3. **Global configuration**: If the key exists in AgentGlobalConfiguration.ApiKeys, it's used
    /// 4. **Environment variable**: Checks standard environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, AZURE_OPENAI_KEY)
    /// 5. **Exception**: If no key is found, throws InvalidOperationException with helpful guidance
    ///
    /// This priority system allows specific configurations to override more general ones, providing flexibility
    /// while maintaining sensible defaults. For example, you can set a global key for most models but override
    /// it with an explicit parameter for specific cases.
    /// </para>
    /// <para><b>For Beginners:</b> This method finds your API key by looking in several places in order.
    ///
    /// How it works:
    /// <code>
    /// // The resolver checks in this order:
    ///
    /// // 1. Explicit key (you provided it directly)
    /// var key = ResolveApiKey(explicitKey: "sk-direct-key");  // Uses this immediately
    ///
    /// // 2. Stored config (saved during model building)
    /// var config = new AgentConfiguration&lt;double&gt; { ApiKey = "sk-stored" };
    /// var key = ResolveApiKey(storedConfig: config);  // Uses stored key if no explicit key
    ///
    /// // 3. Global config (set at application startup)
    /// AgentGlobalConfiguration.Configure(c => c.ConfigureOpenAI("sk-global"));
    /// var key = ResolveApiKey();  // Uses global key if nothing more specific
    ///
    /// // 4. Environment variable (set in system environment)
    /// // If OPENAI_API_KEY is set in environment, uses that
    /// var key = ResolveApiKey();  // Checks environment if no other source
    /// </code>
    ///
    /// Priority example:
    /// <code>
    /// // Even if you have global config and environment variable set,
    /// // an explicit key takes priority:
    /// AgentGlobalConfiguration.Configure(c => c.ConfigureOpenAI("global-key"));
    /// Environment.SetEnvironmentVariable("OPENAI_API_KEY", "env-key");
    ///
    /// var key = ResolveApiKey(explicitKey: "explicit-key");
    /// // Result: "explicit-key" (highest priority wins)
    ///
    /// var key2 = ResolveApiKey();
    /// // Result: "global-key" (next highest priority when no explicit key)
    /// </code>
    ///
    /// If no key is found anywhere, you get a clear error message telling you exactly how to fix it:
    /// <code>
    /// try {
    ///     var key = ResolveApiKey();  // No key configured anywhere
    /// }
    /// catch (InvalidOperationException ex) {
    ///     // Exception message lists all ways to provide the key:
    ///     // "No API key found for OpenAI. Please provide via:
    ///     //  1. Explicit parameter: .ConfigureAgentAssistance(apiKey: "...")
    ///     //  2. Global config: AgentGlobalConfiguration.Configure(...)
    ///     //  3. Environment variable: OPENAI_API_KEY"
    /// }
    /// </code>
    ///
    /// This automatic resolution means you can use the most convenient method for your situation while
    /// maintaining security and flexibility.
    /// </para>
    /// </remarks>
    public static string ResolveApiKey<T>(
        string? explicitKey = null,
        AgentConfiguration<T>? storedConfig = null,
        LLMProvider provider = LLMProvider.OpenAI)
    {
        // 1. Explicit parameter takes highest priority
        if (explicitKey != null && !string.IsNullOrWhiteSpace(explicitKey))
            return explicitKey;

        // 2. Check stored config from build phase
        if (storedConfig?.ApiKey != null && !string.IsNullOrWhiteSpace(storedConfig.ApiKey))
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
