namespace AiDotNet.Agentic.Models.Connectors;

/// <summary>
/// An <see cref="IChatClient{T}"/> for the <see href="https://mistral.ai">Mistral</see> platform, whose chat
/// API is OpenAI-compatible. It reuses the OpenAI wire format and only changes the endpoint and key, adding
/// Mistral's hosted models to the connector lineup with no bespoke code.
/// </summary>
/// <typeparam name="T">The numeric type shared across the agent stack.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Same agent code as OpenAI, pointed at Mistral's servers with your Mistral key.
/// </para>
/// </remarks>
public sealed class MistralChatClient<T> : OpenAIChatClient<T>
{
    /// <summary>The default Mistral OpenAI-compatible chat endpoint.</summary>
    public const string DefaultEndpoint = "https://api.mistral.ai/v1/chat/completions";

    /// <summary>
    /// Initializes a new Mistral client.
    /// </summary>
    /// <param name="apiKey">The Mistral API key.</param>
    /// <param name="modelName">The model name (e.g., <c>mistral-large-latest</c>).</param>
    /// <param name="endpoint">The chat endpoint. <c>null</c> uses <see cref="DefaultEndpoint"/>.</param>
    /// <param name="httpClient">Optional <see cref="System.Net.Http.HttpClient"/> (for testing or custom handlers).</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="apiKey"/> or <paramref name="modelName"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="apiKey"/> or <paramref name="modelName"/> is empty/whitespace.</exception>
    public MistralChatClient(string apiKey, string modelName = "mistral-large-latest", string? endpoint = null, System.Net.Http.HttpClient? httpClient = null)
        : base(apiKey, modelName, endpoint ?? DefaultEndpoint, httpClient)
    {
    }
}
