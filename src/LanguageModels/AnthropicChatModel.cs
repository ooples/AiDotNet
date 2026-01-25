using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using AiDotNet.Interfaces;
using Newtonsoft.Json;
using Newtonsoft.Json.Serialization;

namespace AiDotNet.LanguageModels;

/// <summary>
/// Implementation of IChatModel for Anthropic's Claude models (Claude 2, Claude 3 family).
/// Supports the Messages API with configurable parameters.
/// </summary>
/// <typeparam name="T">The numeric type used for model parameters and operations (e.g., double, float).</typeparam>
/// <remarks>
/// For Beginners:
/// This class lets you use Anthropic's Claude models in your code.
///
/// What you need:
/// - An Anthropic API key (get one at console.anthropic.com)
/// - Internet connection (calls Anthropic's cloud API)
/// - Budget (API calls cost money, but there's a free tier for testing)
///
/// Supported models:
/// - claude-3-opus-20240229: Most capable, best for complex tasks ($15/$75 per million tokens)
/// - claude-3-sonnet-20240229: Balanced performance/cost ($3/$15 per million tokens)
/// - claude-3-haiku-20240307: Fastest, most affordable ($0.25/$1.25 per million tokens)
/// - claude-2.1: Previous generation, still very capable ($8/$24 per million tokens)
///
/// Why choose Claude:
/// - Longer context windows (up to 200K tokens)
/// - Strong reasoning capabilities
/// - More nuanced understanding
/// - Better at following complex instructions
/// - Good for analysis, writing, coding
///
/// Example usage:
/// <code>
/// var model = new AnthropicChatModel&lt;double&gt;("your-api-key-here");
///
/// // Simple question
/// string answer = await model.GenerateAsync("Explain quantum entanglement");
///
/// // With custom settings
/// var customModel = new AnthropicChatModel&lt;double&gt;(
///     apiKey: "your-api-key",
///     modelName: "claude-3-opus-20240229",
///     temperature: 0.7,
///     maxTokens: 2048
/// );
/// </code>
///
/// API key security:
/// - Never commit API keys to git
/// - Use environment variables or secure vaults
/// - Monitor usage at console.anthropic.com
/// </remarks>
public class AnthropicChatModel<T> : ChatModelBase<T>
{
    private readonly string _apiKey;
    private readonly string _endpoint;
    private readonly double _temperature;
    private readonly int _maxTokens;
    private readonly double _topP;
    private readonly int _topK;

    private static readonly JsonSerializerSettings JsonOptions = new()
    {
        ContractResolver = new DefaultContractResolver { NamingStrategy = new SnakeCaseNamingStrategy() },
        NullValueHandling = NullValueHandling.Ignore,
        Formatting = Formatting.None
    };

    /// <summary>
    /// Initializes a new instance of the <see cref="AnthropicChatModel{T}"/> class.
    /// </summary>
    /// <param name="apiKey">Your Anthropic API key.</param>
    /// <param name="modelName">The model to use (default: claude-3-sonnet-20240229).</param>
    /// <param name="temperature">Controls randomness (0.0 = deterministic, 1.0 = creative). Default: 0.7.</param>
    /// <param name="maxTokens">Maximum tokens to generate. Default: 2048.</param>
    /// <param name="topP">Nucleus sampling parameter (0.0-1.0). Default: 1.0.</param>
    /// <param name="topK">Top-K sampling parameter. Default: 0 (disabled).</param>
    /// <param name="httpClient">Optional HTTP client.</param>
    /// <param name="endpoint">Optional custom API endpoint.</param>
    /// <remarks>
    /// For Beginners:
    ///
    /// **Required parameter:**
    /// - apiKey: Your secret key from Anthropic (keep this safe!)
    ///
    /// **Model selection:**
    /// - "claude-3-opus-20240229": Most intelligent, expensive, best quality
    /// - "claude-3-sonnet-20240229": Balanced (recommended for most uses)
    /// - "claude-3-haiku-20240307": Fastest, cheapest, still very good
    /// - "claude-2.1": Previous generation, lower cost
    ///
    /// **Temperature (creativity control):**
    /// - 0.0: Very focused, deterministic
    /// - 0.7: Balanced (default)
    /// - 1.0: Maximum creativity
    ///
    /// **MaxTokens (response length):**
    /// - 256: Short answer
    /// - 1024: Medium answer
    /// - 2048: Long answer (default)
    /// - 4096: Very detailed response
    /// - Note: Claude supports up to 4096 output tokens
    ///
    /// **Advanced parameters:**
    /// - topP: Alternative sampling method (usually leave at 1.0)
    /// - topK: Limits vocabulary considered (0 = disabled)
    /// </remarks>
    public AnthropicChatModel(
        string apiKey,
        string modelName = "claude-3-sonnet-20240229",
        double temperature = 0.7,
        int maxTokens = 2048,
        double topP = 1.0,
        int topK = 0,
        HttpClient? httpClient = null,
        string? endpoint = null)
        : base(httpClient, GetMaxContextTokens(modelName ?? throw new ArgumentNullException(nameof(modelName), "Model name cannot be null.")), maxTokens)
    {
        ValidateApiKey(apiKey);

        if (string.IsNullOrWhiteSpace(modelName))
        {
            throw new ArgumentException("Model name cannot be empty or whitespace.", nameof(modelName));
        }

        if (temperature < 0 || temperature > 1)
        {
            throw new ArgumentException("Temperature must be between 0 and 1.", nameof(temperature));
        }

        if (maxTokens < 1 || maxTokens > 4096)
        {
            throw new ArgumentException("Max tokens must be between 1 and 4096.", nameof(maxTokens));
        }

        if (topP < 0 || topP > 1)
        {
            throw new ArgumentException("TopP must be between 0 and 1.", nameof(topP));
        }

        if (topK < 0)
        {
            throw new ArgumentException("TopK must be non-negative.", nameof(topK));
        }

        _apiKey = apiKey;
        _endpoint = endpoint ?? "https://api.anthropic.com/v1/messages";
        _temperature = temperature;
        _maxTokens = maxTokens;
        _topP = topP;
        _topK = topK;

        ModelName = modelName;
        MaxGenerationTokens = maxTokens;
    }

    /// <inheritdoc/>
    protected override async Task<string> GenerateAsyncCore(string prompt, CancellationToken cancellationToken)
    {
        // Build the request
        var requestPayload = new AnthropicRequest
        {
            Model = ModelName,
            Messages = new[]
            {
                new AnthropicMessage
                {
                    Role = "user",
                    Content = prompt
                }
            },
            MaxTokens = _maxTokens,
            Temperature = _temperature,
            TopP = _topP,
            TopK = _topK > 0 ? _topK : null
        };

        var jsonContent = JsonConvert.SerializeObject(requestPayload, JsonOptions);
        var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

        // Create request message with scoped headers
        using var request = new HttpRequestMessage(HttpMethod.Post, _endpoint)
        {
            Content = content
        };
        request.Headers.Add("x-api-key", _apiKey);
        request.Headers.Add("anthropic-version", "2023-06-01");

        // Make the API call with cancellation support
        using var response = await HttpClient.SendAsync(request, cancellationToken);

        // Check for errors
        if (!response.IsSuccessStatusCode)
        {
            var errorContent = await response.Content.ReadAsStringAsync();
#if NET5_0_OR_GREATER
            throw new HttpRequestException(
                $"Anthropic API request failed with status {response.StatusCode}: {errorContent}",
                null,
                response.StatusCode);
#else
            throw new HttpRequestException(
                $"Anthropic API request failed with status {response.StatusCode}: {errorContent}");
#endif
        }

        // Parse the response
        var responseContent = await response.Content.ReadAsStringAsync();
        var anthropicResponse = JsonConvert.DeserializeObject<AnthropicResponse>(responseContent, JsonOptions);

        if (anthropicResponse?.Content == null || anthropicResponse.Content.Length == 0)
        {
            throw new InvalidOperationException("Anthropic API returned no content in response.");
        }

        // Extract text from content blocks
        var textContent = anthropicResponse.Content
            .Where(c => c.Type == "text" && !string.IsNullOrEmpty(c.Text))
            .Select(c => c.Text)
            .ToArray();

        if (textContent.Length == 0)
        {
            throw new InvalidOperationException("Anthropic API returned no text content.");
        }

        return string.Join("\n", textContent);
    }

    /// <summary>
    /// Gets the maximum context window size for a given model.
    /// </summary>
    /// <param name="modelName">The model name.</param>
    /// <returns>The maximum context tokens for the model.</returns>
    private static int GetMaxContextTokens(string modelName)
    {
        return modelName.ToLowerInvariant() switch
        {
            var m when m.Contains("claude-3-opus") => 200000,
            var m when m.Contains("claude-3-sonnet") => 200000,
            var m when m.Contains("claude-3-haiku") => 200000,
            var m when m.Contains("claude-2.1") => 200000,
            var m when m.Contains("claude-2.0") => 100000,
            var m when m.Contains("claude-2") => 100000,
            var m when m.Contains("claude-instant") => 100000,
            _ => 100000 // Default fallback
        };
    }

    #region Anthropic API Models

    /// <summary>
    /// Represents an Anthropic Messages API request.
    /// </summary>
    private class AnthropicRequest
    {
        [JsonProperty("model")]
        public string Model { get; set; } = "";

        [JsonProperty("messages")]
        public AnthropicMessage[] Messages { get; set; } = Array.Empty<AnthropicMessage>();

        [JsonProperty("max_tokens")]
        public int MaxTokens { get; set; }

        [JsonProperty("temperature")]
        public double Temperature { get; set; }

        [JsonProperty("top_p")]
        public double TopP { get; set; }

        [JsonProperty("top_k")]
        public int? TopK { get; set; }
    }

    /// <summary>
    /// Represents a message in the Anthropic Messages API.
    /// </summary>
    private class AnthropicMessage
    {
        [JsonProperty("role")]
        public string Role { get; set; } = "";

        [JsonProperty("content")]
        public string Content { get; set; } = "";
    }

    /// <summary>
    /// Represents an Anthropic Messages API response.
    /// </summary>
    private class AnthropicResponse
    {
        [JsonProperty("id")]
        public string? Id { get; set; }

        [JsonProperty("type")]
        public string? Type { get; set; }

        [JsonProperty("role")]
        public string? Role { get; set; }

        [JsonProperty("content")]
        public AnthropicContent[]? Content { get; set; }

        [JsonProperty("model")]
        public string? Model { get; set; }

        [JsonProperty("stop_reason")]
        public string? StopReason { get; set; }

        [JsonProperty("usage")]
        public AnthropicUsage? Usage { get; set; }
    }

    /// <summary>
    /// Represents a content block in the Anthropic API response.
    /// </summary>
    private class AnthropicContent
    {
        [JsonProperty("type")]
        public string? Type { get; set; }

        [JsonProperty("text")]
        public string? Text { get; set; }
    }

    /// <summary>
    /// Represents token usage information in the Anthropic API response.
    /// </summary>
    private class AnthropicUsage
    {
        [JsonProperty("input_tokens")]
        public int InputTokens { get; set; }

        [JsonProperty("output_tokens")]
        public int OutputTokens { get; set; }
    }

    #endregion
}
