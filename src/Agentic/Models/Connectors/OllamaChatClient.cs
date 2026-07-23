namespace AiDotNet.Agentic.Models.Connectors;

/// <summary>
/// An <see cref="IChatClient{T}"/> for a local <see href="https://ollama.com">Ollama</see> server, which
/// exposes an OpenAI-compatible chat-completions API. It reuses the entire OpenAI wire format (messages,
/// tools, streaming, usage) and only changes the endpoint — so locally-served open models (Llama, Mistral,
/// Qwen, …) drive the agent stack with no extra code.
/// </summary>
/// <typeparam name="T">The numeric type shared across the agent stack.</typeparam>
/// <remarks>
/// <para>
/// Ollama ignores the bearer token, so a placeholder key is sent. The default endpoint targets a local
/// daemon (<c>http://localhost:11434/v1/chat/completions</c>); override it to reach a remote Ollama host.
/// This complements the in-process <c>LocalEngineChatClient</c>: Ollama runs models in a separate local
/// server process, while the local engine runs them inside AiDotNet itself.
/// </para>
/// <para><b>For Beginners:</b> If you run Ollama on your machine, this points the agent at it — a free, local,
/// private model with the same code you'd use for OpenAI.
/// </para>
/// </remarks>
public sealed class OllamaChatClient<T> : OpenAIChatClient<T>
{
    /// <summary>The default Ollama OpenAI-compatible endpoint (local daemon).</summary>
    public const string DefaultEndpoint = "http://localhost:11434/v1/chat/completions";

    /// <summary>
    /// Initializes a new Ollama client.
    /// </summary>
    /// <param name="modelName">The Ollama model name (e.g., <c>llama3.1</c>, <c>qwen2.5</c>).</param>
    /// <param name="endpoint">The chat-completions endpoint. <c>null</c> uses <see cref="DefaultEndpoint"/>.</param>
    /// <param name="httpClient">Optional <see cref="System.Net.Http.HttpClient"/> (for testing or custom handlers).</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="modelName"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="modelName"/> is empty/whitespace.</exception>
    public OllamaChatClient(string modelName, string? endpoint = null, System.Net.Http.HttpClient? httpClient = null)
        : base(apiKey: "ollama", modelName: modelName, endpoint: endpoint ?? DefaultEndpoint, httpClient: httpClient)
    {
    }
}
