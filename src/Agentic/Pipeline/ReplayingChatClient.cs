using System.Runtime.CompilerServices;
using AiDotNet.Agentic.Models;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Pipeline;

/// <summary>
/// An <see cref="IChatClient{T}"/> that serves responses from a recorded <see cref="IChatInteractionStore"/> —
/// deterministic replay without calling any model. On a cache miss it falls through to an optional inner
/// client (recording the new interaction) or throws, depending on configuration.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// This makes agent runs reproducible: record once against a real model, then replay the exact same
/// trajectory in tests/CI/debugging at zero cost and with no nondeterminism. With a fallback client it acts
/// as a persistent cache (replay hits are free; misses call the model and are recorded).
/// </para>
/// <para><b>For Beginners:</b> Plays back saved model answers. Ask the same thing and you get the same recorded
/// reply instantly — no model call. Great for fast, deterministic tests and for re-running a session exactly.
/// </para>
/// </remarks>
public sealed class ReplayingChatClient<T> : IChatClient<T>
{
    private readonly IChatInteractionStore _store;
    private readonly IChatClient<T>? _fallback;

    /// <summary>
    /// Initializes a new replaying client.
    /// </summary>
    /// <param name="store">The store of recorded interactions.</param>
    /// <param name="fallback">Optional client called on a cache miss (its response is recorded). <c>null</c> throws on a miss.</param>
    /// <param name="modelId">The model id to report. <c>null</c> uses the fallback's id or <c>"replay"</c>.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="store"/> is <c>null</c>.</exception>
    public ReplayingChatClient(IChatInteractionStore store, IChatClient<T>? fallback = null, string? modelId = null)
    {
        Guard.NotNull(store);
        _store = store;
        _fallback = fallback;
        ModelId = modelId is { } id && id.Trim().Length > 0 ? id : fallback?.ModelId ?? "replay";
    }

    /// <inheritdoc/>
    public string ModelId { get; }

    /// <inheritdoc/>
    public async Task<ChatResponse> GetResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(messages);
        var key = ChatInteractionKey.For(messages, options);
        if (_store.TryGet(key, out var recorded))
        {
            return recorded;
        }

        if (_fallback is null)
        {
            throw new InvalidOperationException(
                "No recorded interaction matches this request and no fallback client was configured.");
        }

        var response = await _fallback.GetResponseAsync(messages, options, cancellationToken).ConfigureAwait(false);
        _store.Save(key, response);
        return response;
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var response = await GetResponseAsync(messages, options, cancellationToken).ConfigureAwait(false);
        yield return new ChatResponseUpdate(role: ChatRole.Assistant);
        if (response.Message.Text.Length > 0)
        {
            yield return ChatResponseUpdate.ForText(response.Message.Text);
        }

        yield return ChatResponseUpdate.ForFinish(response.FinishReason, response.Usage);
    }
}
