using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using AiDotNet.Interfaces;
using AiDotNet.LanguageModels.Models;
using Newtonsoft.Json;
using Newtonsoft.Json.Serialization;

namespace AiDotNet.LanguageModels;

/// <summary>
/// Implementation of IChatModel for OpenAI's GPT models (GPT-3.5-turbo, GPT-4, GPT-4-turbo).
/// Supports the Chat Completions API with configurable parameters.
/// </summary>
/// <typeparam name="T">The numeric type used for model parameters and operations (e.g., double, float).</typeparam>
/// <remarks>
/// For Beginners:
/// This class lets you use OpenAI's GPT models (like ChatGPT) in your code.
///
/// What you need:
/// - An OpenAI API key (get one at platform.openai.com)
/// - Internet connection (calls OpenAI's cloud API)
/// - Some budget (API calls cost money, but it's very affordable for testing)
///
/// Supported models:
/// - gpt-3.5-turbo: Fast, cheap, good for most tasks ($0.0005/1K tokens)
/// - gpt-4: Most capable, slower, more expensive ($0.03/1K tokens)
/// - gpt-4-turbo: Fast GPT-4, lower cost ($0.01/1K tokens)
/// - gpt-4o: Optimized multimodal model
///
/// Example usage:
/// <code>
/// var model = new OpenAIChatModel&lt;double&gt;("your-api-key-here");
///
/// // Simple question
/// string answer = await model.GenerateAsync("What is machine learning?");
///
/// // With custom settings
/// var customModel = new OpenAIChatModel&lt;double&gt;(
///     apiKey: "your-api-key",
///     modelName: "gpt-4",
///     temperature: 0.7,  // More creative
///     maxTokens: 500     // Longer responses
/// );
/// </code>
///
/// Cost-saving tips:
/// - Use gpt-3.5-turbo for simple tasks
/// - Set maxTokens to limit response length
/// - Cache responses when appropriate
/// - Monitor usage at platform.openai.com/usage
/// </remarks>
public class OpenAIChatModel<T> : ChatModelBase<T>
{
    private readonly string _apiKey;
    private readonly string _endpoint;
    private readonly double _temperature;
    private readonly int _maxTokens;
    private readonly double _topP;
    private readonly double _frequencyPenalty;
    private readonly double _presencePenalty;

    private static readonly JsonSerializerSettings JsonSettings = new()
    {
        ContractResolver = new DefaultContractResolver
        {
            NamingStrategy = new SnakeCaseNamingStrategy()
        },
        NullValueHandling = NullValueHandling.Ignore,
        Formatting = Formatting.None
    };

    /// <summary>
    /// Initializes a new instance of the <see cref="OpenAIChatModel{T}"/> class.
    /// </summary>
    /// <param name="apiKey">Your OpenAI API key.</param>
    /// <param name="modelName">The model to use (default: gpt-3.5-turbo).</param>
    /// <param name="temperature">Controls randomness (0.0 = deterministic, 2.0 = very creative). Default: 0.7.</param>
    /// <param name="maxTokens">Maximum tokens to generate. Default: 2048.</param>
    /// <param name="topP">Nucleus sampling parameter (0.0-1.0). Default: 1.0.</param>
    /// <param name="frequencyPenalty">Penalize frequent tokens (-2.0 to 2.0). Default: 0.0.</param>
    /// <param name="presencePenalty">Penalize tokens based on presence (-2.0 to 2.0). Default: 0.0.</param>
    /// <param name="httpClient">Optional HTTP client.</param>
    /// <param name="endpoint">Optional custom API endpoint (for Azure OpenAI or proxies).</param>
    /// <remarks>
    /// For Beginners:
    ///
    /// **Required parameter:**
    /// - apiKey: Your secret key from OpenAI (keep this safe!)
    ///
    /// **Model selection:**
    /// - "gpt-3.5-turbo": Fast, cheap, good enough for most tasks
    /// - "gpt-4": Smarter, better at complex reasoning
    /// - "gpt-4-turbo": Fast GPT-4 variant
    /// - "gpt-4o": Latest optimized model
    ///
    /// **Temperature (creativity control):**
    /// - 0.0: Very focused, deterministic, same answer every time
    /// - 0.7: Balanced (default for most use cases)
    /// - 1.5+: Very creative, more varied, but might be less accurate
    ///
    /// **MaxTokens (response length):**
    /// - 100: Short answer (1-2 sentences)
    /// - 500: Medium answer (1-2 paragraphs)
    /// - 2048: Long answer (1-2 pages)
    /// - Note: More tokens = higher cost
    ///
    /// **Advanced parameters** (usually leave as defaults):
    /// - topP: Alternative to temperature for controlling randomness
    /// - frequencyPenalty: Reduce repetitive text
    /// - presencePenalty: Encourage topic diversity
    /// </remarks>
    public OpenAIChatModel(
        string apiKey,
        string modelName = "gpt-3.5-turbo",
        double temperature = 0.7,
        int maxTokens = 2048,
        double topP = 1.0,
        double frequencyPenalty = 0.0,
        double presencePenalty = 0.0,
        HttpClient? httpClient = null,
        string? endpoint = null)
        : base(httpClient, GetMaxContextTokens(modelName), maxTokens)
    {
        ValidateApiKey(apiKey);

        if (temperature < 0 || temperature > 2)
        {
            throw new ArgumentException("Temperature must be between 0 and 2.", nameof(temperature));
        }

        if (topP < 0 || topP > 1)
        {
            throw new ArgumentException("TopP must be between 0 and 1.", nameof(topP));
        }

        if (frequencyPenalty < -2 || frequencyPenalty > 2)
        {
            throw new ArgumentException("Frequency penalty must be between -2 and 2.", nameof(frequencyPenalty));
        }

        if (presencePenalty < -2 || presencePenalty > 2)
        {
            throw new ArgumentException("Presence penalty must be between -2 and 2.", nameof(presencePenalty));
        }

        _apiKey = apiKey;
        _endpoint = endpoint ?? "https://api.openai.com/v1/chat/completions";
        _temperature = temperature;
        _maxTokens = maxTokens;
        _topP = topP;
        _frequencyPenalty = frequencyPenalty;
        _presencePenalty = presencePenalty;

        ModelName = modelName;
        MaxGenerationTokens = maxTokens;
    }

    /// <inheritdoc/>
    protected override async Task<string> GenerateAsyncCore(string prompt, CancellationToken cancellationToken)
    {
        // Build the request
        var requestPayload = new OpenAIRequest
        {
            Model = ModelName,
            Messages =
            [
                new OpenAIMessage
                {
                    Role = "user",
                    Content = prompt
                }
            ],
            Temperature = _temperature,
            MaxTokens = _maxTokens,
            TopP = _topP,
            FrequencyPenalty = _frequencyPenalty,
            PresencePenalty = _presencePenalty
        };

        var jsonContent = JsonConvert.SerializeObject(requestPayload, JsonSettings);
        var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

        // Create request message with scoped headers
        using var request = new HttpRequestMessage(HttpMethod.Post, _endpoint)
        {
            Content = content
        };
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);

        // Make the API call with cancellation support
        using var response = await HttpClient.SendAsync(request, cancellationToken);

        // Check for errors
        if (!response.IsSuccessStatusCode)
        {
            var errorContent = await response.Content.ReadAsStringAsync();
#if NET5_0_OR_GREATER
            throw new HttpRequestException(
                $"OpenAI API request failed with status {response.StatusCode}: {errorContent}",
                null,
                response.StatusCode);
#else
            throw new HttpRequestException(
                $"OpenAI API request failed with status {response.StatusCode}: {errorContent}");
#endif
        }

        // Parse the response
        var responseContent = await response.Content.ReadAsStringAsync();
        var openAIResponse = JsonConvert.DeserializeObject<OpenAIResponse>(responseContent, JsonSettings);

        if (openAIResponse?.Choices == null || openAIResponse.Choices.Length == 0)
        {
            throw new InvalidOperationException("OpenAI API returned no choices in response.");
        }

        var choice = openAIResponse.Choices[0];
        if (choice?.Message?.Content == null || choice.Message.Content.Length == 0)
        {
            throw new InvalidOperationException("OpenAI API returned empty message content.");
        }

        return choice.Message.Content;
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
            // GPT-3.5 models
            "gpt-3.5-turbo" => 4096,
            "gpt-3.5-turbo-16k" => 16384,

            // GPT-4 base models
            "gpt-4" => 8192,
            "gpt-4-32k" => 32768,

            // GPT-4 Turbo models (128k context)
            "gpt-4-turbo" => 128000,
            "gpt-4-turbo-preview" => 128000,
            "gpt-4-1106-preview" => 128000,
            "gpt-4-0125-preview" => 128000,

            // GPT-4o models (128k context)
            "gpt-4o" => 128000,
            "gpt-4o-mini" => 128000,
            "gpt-4o-2024-05-13" => 128000,
            "gpt-4o-2024-08-06" => 128000,
            "gpt-4o-2024-11-20" => 128000,

            // o1 reasoning models (128k context)
            "o1" => 128000,
            "o1-preview" => 128000,
            "o1-mini" => 128000,

            // o3 reasoning models (200k context)
            "o3-mini" => 200000,

            _ => 4096 // Default fallback
        };
    }
}
