using AiDotNet.Agentic.Models;
using Meai = Microsoft.Extensions.AI;

// Disambiguate our ChatMessage from the legacy PromptEngineering.Templates.ChatMessage imported
// project-wide via a global using. (MEAI's identically-named types are reached through the Meai alias.)
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Models.Connectors;

/// <summary>
/// Adapts a <see cref="Microsoft.Extensions.AI.IChatClient"/> (the .NET ecosystem's standard chat
/// abstraction) to AiDotNet's <see cref="IChatClient{T}"/>, so any Microsoft.Extensions.AI connector
/// (OpenAI, Azure, Ollama, etc.) can drive AiDotNet agents and reasoning.
/// </summary>
/// <typeparam name="T">The numeric type used across the AiDotNet ecosystem.</typeparam>
/// <remarks>
/// <para>
/// This bridges the two ecosystems for the common text + sampling + streaming path. Native tool calling
/// through this adapter is not yet supported (Microsoft.Extensions.AI models tools as executable
/// <c>AIFunction</c>s rather than schema-only definitions); passing tools throws
/// <see cref="NotSupportedException"/>. Use the first-party connectors
/// (<see cref="OpenAIChatClient{T}"/>, <see cref="AnthropicChatClient{T}"/>) for tool calling.
/// </para>
/// <para><b>For Beginners:</b> Microsoft.Extensions.AI is .NET's shared interface for talking to chat
/// models. This adapter lets you take any model that speaks that interface and use it anywhere AiDotNet
/// expects its own <see cref="IChatClient{T}"/> — so you inherit the whole ecosystem of providers.
/// </para>
/// </remarks>
public sealed class MeaiChatClient<T> : IChatClient<T>
{
    private readonly Meai.IChatClient _inner;

    /// <summary>
    /// Initializes a new adapter around a Microsoft.Extensions.AI chat client.
    /// </summary>
    /// <param name="inner">The underlying Microsoft.Extensions.AI client.</param>
    /// <param name="modelId">Optional model id reported by <see cref="ModelId"/> (defaults to "meai").</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="inner"/> is <c>null</c>.</exception>
    public MeaiChatClient(Meai.IChatClient inner, string? modelId = null)
    {
        Guard.NotNull(inner);
        _inner = inner;
        ModelId = modelId is null || modelId.Trim().Length == 0 ? "meai" : modelId;
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
        var meaiMessages = ToMeaiMessages(messages);
        var meaiOptions = ToMeaiOptions(options);

        var response = await _inner.GetResponseAsync(meaiMessages, meaiOptions, cancellationToken).ConfigureAwait(false);

        var assistant = ChatMessage.Assistant(response.Text ?? string.Empty);
        return new ChatResponse(assistant, MapFinishReason(response.FinishReason), MapUsage(response.Usage), response.ModelId ?? ModelId);
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        Guard.NotNull(messages);
        var meaiMessages = ToMeaiMessages(messages);
        var meaiOptions = ToMeaiOptions(options);

        ChatFinishReason? finishReason = null;
        bool roleEmitted = false;

        await foreach (var update in _inner.GetStreamingResponseAsync(meaiMessages, meaiOptions, cancellationToken).ConfigureAwait(false))
        {
            if (!roleEmitted)
            {
                roleEmitted = true;
                yield return new ChatResponseUpdate(role: ChatRole.Assistant);
            }

            var text = update.Text;
            if (!string.IsNullOrEmpty(text))
            {
                yield return ChatResponseUpdate.ForText(text);
            }

            if (update.FinishReason is { } reason)
            {
                finishReason = MapFinishReason(reason);
            }
        }

        yield return ChatResponseUpdate.ForFinish(finishReason ?? ChatFinishReason.Stop);
    }

    private static List<Meai.ChatMessage> ToMeaiMessages(IReadOnlyList<ChatMessage> messages)
    {
        var result = new List<Meai.ChatMessage>(messages.Count);
        foreach (var message in messages)
        {
            // Fail fast at the boundary rather than letting a null message
            // crash with NullReferenceException inside .Contents access below.
            Guard.NotNull(message);

            if (message.Contents.Any(c => c is ToolCallContent || c is ToolResultContent || c is ImageContent))
            {
                throw new NotSupportedException(
                    "MeaiChatClient currently supports text content only. Use the first-party connectors for tool calls or images.");
            }

            result.Add(new Meai.ChatMessage(MapRole(message.Role), message.Text));
        }

        return result;
    }

    private static Meai.ChatOptions? ToMeaiOptions(ChatOptions? options)
    {
        if (options is null)
        {
            return null;
        }

        if (options.Tools is { Count: > 0 })
        {
            throw new NotSupportedException(
                "Tool calling through MeaiChatClient is not yet supported. Use the first-party connectors for tools.");
        }

        var meai = new Meai.ChatOptions();
        if (options.Temperature is { } temperature) meai.Temperature = (float)temperature;
        if (options.MaxOutputTokens is { } maxTokens) meai.MaxOutputTokens = maxTokens;
        if (options.TopP is { } topP) meai.TopP = (float)topP;
        if (options.TopK is { } topK) meai.TopK = topK;
        if (options.Seed is { } seed) meai.Seed = seed;
        if (options.StopSequences is { Count: > 0 } stops) meai.StopSequences = stops.ToList();
        return meai;
    }

    private static Meai.ChatRole MapRole(ChatRole role) => role switch
    {
        ChatRole.System => Meai.ChatRole.System,
        ChatRole.User => Meai.ChatRole.User,
        ChatRole.Assistant => Meai.ChatRole.Assistant,
        ChatRole.Tool => Meai.ChatRole.Tool,
        _ => Meai.ChatRole.User
    };

    private static ChatFinishReason MapFinishReason(Meai.ChatFinishReason? reason)
    {
        if (reason is null)
        {
            return ChatFinishReason.Stop;
        }

        var value = reason.Value;
        if (value == Meai.ChatFinishReason.Stop) return ChatFinishReason.Stop;
        if (value == Meai.ChatFinishReason.Length) return ChatFinishReason.Length;
        if (value == Meai.ChatFinishReason.ToolCalls) return ChatFinishReason.ToolCalls;
        if (value == Meai.ChatFinishReason.ContentFilter) return ChatFinishReason.ContentFilter;
        return ChatFinishReason.Unknown;
    }

    private static ChatUsage? MapUsage(Meai.UsageDetails? usage)
    {
        if (usage is null)
        {
            return null;
        }

        var input = ClampToInt(usage.InputTokenCount ?? 0);
        var output = ClampToInt(usage.OutputTokenCount ?? 0);
        return new ChatUsage(input, output);
    }

    // Token counts are long? in MEAI; clamp (don't overflow-cast) into ChatUsage's int range.
    private static int ClampToInt(long value) =>
        value < 0 ? 0 : (value > int.MaxValue ? int.MaxValue : (int)value);
}
