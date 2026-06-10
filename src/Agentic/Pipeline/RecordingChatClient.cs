using AiDotNet.Agentic.Models;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Pipeline;

/// <summary>
/// An <see cref="IChatClient{T}"/> decorator that calls a real inner client and records each request/response
/// into an <see cref="IChatInteractionStore"/>. Pair it with <see cref="ReplayingChatClient{T}"/> to capture a
/// run once and replay it deterministically thereafter.
/// </summary>
/// <typeparam name="T">The numeric type of the inner client.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A tape recorder around the model. It passes your request to the real model and
/// quietly saves the answer keyed by the request, so you can play it back later without spending another call.
/// </para>
/// </remarks>
public sealed class RecordingChatClient<T> : IChatClient<T>
{
    private readonly IChatClient<T> _inner;
    private readonly IChatInteractionStore _store;

    /// <summary>
    /// Initializes a new recording client.
    /// </summary>
    /// <param name="inner">The real client to call and record.</param>
    /// <param name="store">The store that receives recorded interactions.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="inner"/> or <paramref name="store"/> is <c>null</c>.</exception>
    public RecordingChatClient(IChatClient<T> inner, IChatInteractionStore store)
    {
        Guard.NotNull(inner);
        Guard.NotNull(store);
        _inner = inner;
        _store = store;
    }

    /// <inheritdoc/>
    public string ModelId => _inner.ModelId;

    /// <inheritdoc/>
    public async Task<ChatResponse> GetResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(messages);
        var response = await _inner.GetResponseAsync(messages, options, cancellationToken).ConfigureAwait(false);
        _store.Save(ChatInteractionKey.For(messages, options), response);
        return response;
    }

    /// <inheritdoc/>
    public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default) =>
        // Streaming passes through unrecorded; use GetResponseAsync to capture interactions.
        _inner.GetStreamingResponseAsync(messages, options, cancellationToken);
}
