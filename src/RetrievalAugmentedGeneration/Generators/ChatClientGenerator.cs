using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using AiDotNet.Agentic.Models;
using AiDotNet.Interfaces;

// ChatMessage collides with AiDotNet.PromptEngineering.Templates.ChatMessage (a project-wide global using);
// the RAG generator talks to the agentic connectors, so bind ChatMessage to the agentic type (same
// disambiguation ChatClientBase uses).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.RetrievalAugmentedGeneration.Generators;

/// <summary>
/// A real external-LLM generator for RAG. Bridges the RAG <see cref="IGenerator{T}"/> contract onto the
/// AiDotNet agentic <see cref="IChatClient{T}"/> connectors (OpenAI, Azure OpenAI, Anthropic, Cohere,
/// Gemini, Mistral, Ollama), so retrieval-augmented answers are produced by a genuine chat model instead
/// of the local-LSTM <c>NeuralGenerator</c> or the <c>StubGenerator</c>. Supports token streaming via
/// <see cref="IStreamingGenerator{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// This is the linchpin that lets the previously-stubbed LLM-powered RAG components (HyDE, LLM query
/// expansion, LLM/Cohere reranking, LLM context compression, multi-query retrieval) run against a real
/// model: construct one of these over any configured <see cref="IChatClient{T}"/> and inject it.
/// </para>
/// <para><b>For Beginners:</b> give it a chat-model connection and it writes answers with that model.
/// <see cref="IGenerator{T}.Generate(string)"/> returns the whole answer; <see cref="GenerateStreamAsync"/>
/// streams it as it is produced.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for relevance scoring.</typeparam>
public class ChatClientGenerator<T> : GeneratorBase<T>, IStreamingGenerator<T>
{
    private readonly IChatClient<T> _chatClient;
    private readonly string? _systemPrompt;
    private readonly ChatOptions _options;

    /// <summary>
    /// Creates a generator over an existing chat-model connector.
    /// </summary>
    /// <param name="chatClient">The configured chat-model connector (provider, model id, API key already set).</param>
    /// <param name="systemPrompt">Optional system instruction prepended to every request.</param>
    /// <param name="temperature">Optional sampling temperature; <c>null</c> uses the provider default.</param>
    /// <param name="maxOutputTokens">Optional generation cap; defaults to <paramref name="maxGenerationTokens"/>.</param>
    /// <param name="maxContextTokens">Context window size reported to the base generator.</param>
    /// <param name="maxGenerationTokens">Default generation length reported to the base generator.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="chatClient"/> is null.</exception>
    public ChatClientGenerator(
        IChatClient<T> chatClient,
        string? systemPrompt = null,
        double? temperature = null,
        int? maxOutputTokens = null,
        int maxContextTokens = 8192,
        int maxGenerationTokens = 1024)
        : base(maxContextTokens, maxGenerationTokens)
    {
        _chatClient = chatClient ?? throw new ArgumentNullException(nameof(chatClient));
        _systemPrompt = systemPrompt;
        _options = new ChatOptions
        {
            Temperature = temperature,
            MaxOutputTokens = maxOutputTokens ?? maxGenerationTokens
        };
    }

    private IReadOnlyList<ChatMessage> BuildMessages(string prompt)
    {
        var messages = new List<ChatMessage>();
        if (!string.IsNullOrWhiteSpace(_systemPrompt))
        {
            messages.Add(new ChatMessage(ChatRole.System, _systemPrompt!));
        }

        messages.Add(new ChatMessage(ChatRole.User, prompt));
        return messages;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Synchronous bridge over the async chat client, matching the codebase's existing sync-over-async
    /// pattern (e.g. the embedding models). Prefer <see cref="GenerateStreamAsync"/> — or the async RAG
    /// pipeline (task #21) — on hot paths.
    /// </remarks>
    protected override string GenerateCore(string prompt)
    {
        var response = _chatClient
            .GetResponseAsync(BuildMessages(prompt), _options, CancellationToken.None)
            .ConfigureAwait(false).GetAwaiter().GetResult();

        return response.Text ?? string.Empty;
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<string> GenerateStreamAsync(
        string prompt,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (prompt == null)
        {
            throw new ArgumentNullException(nameof(prompt), "Prompt cannot be null.");
        }

        if (string.IsNullOrWhiteSpace(prompt))
        {
            throw new ArgumentException("Prompt cannot be empty or whitespace.", nameof(prompt));
        }

        await foreach (var update in _chatClient
            .GetStreamingResponseAsync(BuildMessages(prompt), _options, cancellationToken)
            .ConfigureAwait(false))
        {
            if (!string.IsNullOrEmpty(update.TextDelta))
            {
                yield return update.TextDelta!;
            }
        }
    }
}
