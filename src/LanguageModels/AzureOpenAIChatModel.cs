using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using AiDotNet.Interfaces;
using Newtonsoft.Json;
using Newtonsoft.Json.Serialization;

namespace AiDotNet.LanguageModels;

/// <summary>
/// Implementation of IChatModel for Azure OpenAI Service.
/// Supports GPT-3.5 and GPT-4 models deployed on Azure with enterprise features.
/// </summary>
/// <typeparam name="T">The numeric type used for model parameters and operations (e.g., double, float).</typeparam>
/// <remarks>
/// For Beginners:
/// This class lets you use OpenAI's GPT models that are hosted on Microsoft Azure.
///
/// Why use Azure OpenAI instead of OpenAI directly?
/// - **Enterprise features**: Better SLAs, support contracts, compliance
/// - **Data residency**: Control where your data is processed geographically
/// - **Private networking**: VNet integration, private endpoints
/// - **Azure integration**: Works with Azure AD, Key Vault, Monitor
/// - **Compliance**: HIPAA, SOC 2, ISO 27001 certified
/// - **No waitlist**: Easier access for enterprise customers
///
/// What you need:
/// - An Azure subscription
/// - Azure OpenAI resource (request access at aka.ms/oai/access)
/// - Deployed model (deploy in Azure OpenAI Studio)
/// - API key or Azure AD authentication
/// - Endpoint URL (from Azure portal)
/// - Deployment name (what you named your model deployment)
///
/// Example usage:
/// <code>
/// var model = new AzureOpenAIChatModel&lt;double&gt;(
///     endpoint: "https://your-resource.openai.azure.com",
///     apiKey: "your-azure-api-key",
///     deploymentName: "gpt-4-deployment",  // Your deployment name in Azure
///     apiVersion: "2024-02-15-preview"     // API version
/// );
///
/// string answer = await model.GenerateAsync("Explain cloud computing");
/// </code>
///
/// Setup steps:
/// 1. Create Azure OpenAI resource in Azure portal
/// 2. Deploy a model (GPT-3.5 or GPT-4) in Azure OpenAI Studio
/// 3. Get endpoint URL and API key from "Keys and Endpoint" section
/// 4. Use your deployment name (NOT the model name)
/// </remarks>
public class AzureOpenAIChatModel<T> : ChatModelBase<T>
{
    private readonly string _apiKey;
    private readonly string _endpoint;
    private readonly string _deploymentName;
    private readonly string _apiVersion;
    private readonly double _temperature;
    private readonly int _maxTokens;
    private readonly double _topP;
    private readonly double _frequencyPenalty;
    private readonly double _presencePenalty;

    private static readonly JsonSerializerSettings JsonOptions = new()
    {
        ContractResolver = new DefaultContractResolver { NamingStrategy = new SnakeCaseNamingStrategy() },
        NullValueHandling = NullValueHandling.Ignore,
        Formatting = Formatting.None
    };

    /// <summary>
    /// Initializes a new instance of the <see cref="AzureOpenAIChatModel{T}"/> class.
    /// </summary>
    /// <param name="endpoint">Your Azure OpenAI endpoint (e.g., "https://your-resource.openai.azure.com").</param>
    /// <param name="apiKey">Your Azure OpenAI API key.</param>
    /// <param name="deploymentName">The name of your deployed model in Azure.</param>
    /// <param name="apiVersion">The API version to use (default: "2024-02-15-preview").</param>
    /// <param name="temperature">Controls randomness (0.0 = deterministic, 2.0 = very creative). Default: 0.7.</param>
    /// <param name="maxTokens">Maximum tokens to generate. Default: 2048.</param>
    /// <param name="topP">Nucleus sampling parameter (0.0-1.0). Default: 1.0.</param>
    /// <param name="frequencyPenalty">Penalize frequent tokens (-2.0 to 2.0). Default: 0.0.</param>
    /// <param name="presencePenalty">Penalize tokens based on presence (-2.0 to 2.0). Default: 0.0.</param>
    /// <param name="maxContextTokens">Maximum context window size. Default: 8192.</param>
    /// <param name="httpClient">Optional HTTP client.</param>
    /// <remarks>
    /// For Beginners:
    ///
    /// **Required parameters:**
    /// - endpoint: Your Azure OpenAI resource URL
    ///   Find this in Azure Portal → Your OpenAI Resource → Keys and Endpoint
    ///   Format: "https://YOUR-RESOURCE-NAME.openai.azure.com"
    ///
    /// - apiKey: Your API key
    ///   Find this in Azure Portal → Your OpenAI Resource → Keys and Endpoint
    ///   Click "Show Keys" to reveal
    ///
    /// - deploymentName: Name YOU chose when deploying the model
    ///   Find this in Azure OpenAI Studio → Deployments
    ///   This is NOT "gpt-4" - it's the name you gave it (e.g., "my-gpt4-deployment")
    ///
    /// **Optional parameters:**
    /// - apiVersion: Azure API version (use default unless you need a specific version)
    ///   Latest versions support newest features
    ///
    /// - temperature, maxTokens, topP, penalties: Same as OpenAI
    ///   See OpenAIChatModel documentation for details
    ///
    /// **Finding your values:**
    /// 1. Go to portal.azure.com
    /// 2. Navigate to your Azure OpenAI resource
    /// 3. Click "Keys and Endpoint" in left menu
    /// 4. Copy: Endpoint, Key 1 (as apiKey)
    /// 5. Click "Model deployments" → "Manage Deployments"
    /// 6. Find your deployment name
    /// </remarks>
    public AzureOpenAIChatModel(
        string endpoint,
        string apiKey,
        string deploymentName,
        string apiVersion = "2024-02-15-preview",
        double temperature = 0.7,
        int maxTokens = 2048,
        double topP = 1.0,
        double frequencyPenalty = 0.0,
        double presencePenalty = 0.0,
        int maxContextTokens = 8192,
        HttpClient? httpClient = null)
        : base(httpClient, maxContextTokens, maxTokens)
    {
        if (string.IsNullOrWhiteSpace(endpoint))
        {
            throw new ArgumentException("Endpoint cannot be null or empty.", nameof(endpoint));
        }

        ValidateApiKey(apiKey, nameof(apiKey));

        if (string.IsNullOrWhiteSpace(deploymentName))
        {
            throw new ArgumentException("Deployment name cannot be null or empty.", nameof(deploymentName));
        }

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

        // Ensure endpoint doesn't end with slash
        _endpoint = endpoint.TrimEnd('/');
        _apiKey = apiKey;
        _deploymentName = deploymentName;
        _apiVersion = apiVersion;
        _temperature = temperature;
        _maxTokens = maxTokens;
        _topP = topP;
        _frequencyPenalty = frequencyPenalty;
        _presencePenalty = presencePenalty;

        ModelName = $"azure-{deploymentName}";
        MaxGenerationTokens = maxTokens;
    }

    /// <inheritdoc/>
    protected override async Task<string> GenerateAsyncCore(string prompt, CancellationToken cancellationToken)
    {
        // Build the Azure OpenAI endpoint URL
        var url = $"{_endpoint}/openai/deployments/{_deploymentName}/chat/completions?api-version={_apiVersion}";

        // Build the request (same format as OpenAI)
        var requestPayload = new AzureOpenAIRequest
        {
            Messages = new[]
            {
                new AzureOpenAIMessage
                {
                    Role = "user",
                    Content = prompt
                }
            },
            Temperature = _temperature,
            MaxTokens = _maxTokens,
            TopP = _topP,
            FrequencyPenalty = _frequencyPenalty,
            PresencePenalty = _presencePenalty
        };

        var jsonContent = JsonConvert.SerializeObject(requestPayload, JsonOptions);
        var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

        // Create request message with scoped headers
        using var request = new HttpRequestMessage(HttpMethod.Post, url)
        {
            Content = content
        };
        request.Headers.Add("api-key", _apiKey);

        // Make the API call with cancellation support
        using var response = await HttpClient.SendAsync(request, cancellationToken);

        // Check for errors
        if (!response.IsSuccessStatusCode)
        {
            var errorContent = await response.Content.ReadAsStringAsync();
#if NET5_0_OR_GREATER
            throw new HttpRequestException(
                $"Azure OpenAI API request failed with status {response.StatusCode}: {errorContent}",
                null,
                response.StatusCode);
#else
            throw new HttpRequestException(
                $"Azure OpenAI API request failed with status {response.StatusCode}: {errorContent}");
#endif
        }

        // Parse the response (same format as OpenAI)
        var responseContent = await response.Content.ReadAsStringAsync();
        var azureResponse = JsonConvert.DeserializeObject<AzureOpenAIResponse>(responseContent, JsonOptions);

        if (azureResponse?.Choices == null || azureResponse.Choices.Length == 0)
        {
            throw new InvalidOperationException("Azure OpenAI API returned no choices in response.");
        }

        var choice = azureResponse.Choices[0];
        if (choice?.Message?.Content == null || choice.Message.Content.Length == 0)
        {
            throw new InvalidOperationException("Azure OpenAI API returned empty message content.");
        }

        return choice.Message.Content;
    }

    #region Azure OpenAI API Models

    /// <summary>
    /// Represents an Azure OpenAI Chat Completions API request.
    /// </summary>
    private class AzureOpenAIRequest
    {
        [JsonProperty("messages")]
        public AzureOpenAIMessage[] Messages { get; set; } = Array.Empty<AzureOpenAIMessage>();

        [JsonProperty("temperature")]
        public double Temperature { get; set; }

        [JsonProperty("max_tokens")]
        public int MaxTokens { get; set; }

        [JsonProperty("top_p")]
        public double TopP { get; set; }

        [JsonProperty("frequency_penalty")]
        public double FrequencyPenalty { get; set; }

        [JsonProperty("presence_penalty")]
        public double PresencePenalty { get; set; }
    }

    /// <summary>
    /// Represents a message in the Azure OpenAI Chat Completions API.
    /// </summary>
    private class AzureOpenAIMessage
    {
        [JsonProperty("role")]
        public string Role { get; set; } = "";

        [JsonProperty("content")]
        public string Content { get; set; } = "";
    }

    /// <summary>
    /// Represents an Azure OpenAI Chat Completions API response.
    /// </summary>
    private class AzureOpenAIResponse
    {
        [JsonProperty("id")]
        public string? Id { get; set; }

        [JsonProperty("choices")]
        public AzureOpenAIChoice[]? Choices { get; set; }

        [JsonProperty("usage")]
        public AzureOpenAIUsage? Usage { get; set; }
    }

    /// <summary>
    /// Represents a choice in the Azure OpenAI API response.
    /// </summary>
    private class AzureOpenAIChoice
    {
        [JsonProperty("index")]
        public int Index { get; set; }

        [JsonProperty("message")]
        public AzureOpenAIMessage? Message { get; set; }

        [JsonProperty("finish_reason")]
        public string? FinishReason { get; set; }
    }

    /// <summary>
    /// Represents token usage information in the Azure OpenAI API response.
    /// </summary>
    private class AzureOpenAIUsage
    {
        [JsonProperty("prompt_tokens")]
        public int PromptTokens { get; set; }

        [JsonProperty("completion_tokens")]
        public int CompletionTokens { get; set; }

        [JsonProperty("total_tokens")]
        public int TotalTokens { get; set; }
    }

    #endregion
}
