using System.Runtime.CompilerServices;
using AiDotNet.Agentic.Models;
using Meai = Microsoft.Extensions.AI;
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Models.Connectors;

/// <summary>
/// Adapts an AiDotNet <see cref="IChatClient{T}"/> to a <see cref="Microsoft.Extensions.AI.IChatClient"/>, so
/// AiDotNet models (including the in-process <c>LocalEngineChatClient</c> and the first-party connectors) can
/// be consumed by any code written against the .NET ecosystem's standard chat abstraction — Semantic Kernel,
/// the MEAI middleware pipeline, <c>FunctionInvokingChatClient</c>, etc.
/// </summary>
/// <typeparam name="T">The numeric type used across the AiDotNet ecosystem.</typeparam>
/// <remarks>
/// <para>
/// This is the outbound counterpart to <see cref="MeaiChatClient{T}"/>. It maps MEAI messages/options/tools to
/// AiDotNet's types (via <see cref="MeaiInterop"/>), calls the wrapped client, and maps the response back —
/// including tool calls: an AiDotNet <see cref="ToolCallContent"/> surfaces as a MEAI
/// <c>FunctionCallContent</c>, so a MEAI host can drive AiDotNet's tool-calling loop.
/// </para>
/// <para><b>For Beginners:</b> The other adapter lets AiDotNet use the ecosystem's models. This one is the
/// reverse: it makes <em>your</em> AiDotNet model look like a standard .NET chat model, so tools and apps that
/// only know the standard interface can use it without knowing it's AiDotNet underneath.
/// </para>
/// </remarks>
public sealed class AiDotNetMeaiChatClient<T> : Meai.IChatClient
{
    private readonly IChatClient<T> _inner;

    /// <summary>
    /// Initializes a new MEAI-facing adapter around an AiDotNet chat client.
    /// </summary>
    /// <param name="inner">The AiDotNet client to expose through the MEAI interface.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="inner"/> is <c>null</c>.</exception>
    public AiDotNetMeaiChatClient(IChatClient<T> inner)
    {
        Guard.NotNull(inner);
        _inner = inner;
    }

    /// <inheritdoc/>
    public async Task<Meai.ChatResponse> GetResponseAsync(
        IEnumerable<Meai.ChatMessage> messages,
        Meai.ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(messages);
        var aiMessages = ToAiMessages(messages);
        var aiOptions = ToAiOptions(options);

        var response = await _inner.GetResponseAsync(aiMessages, aiOptions, cancellationToken).ConfigureAwait(false);

        var meaiMessage = MeaiInterop.ToMeaiMessage(response.Message);
        return new Meai.ChatResponse(meaiMessage)
        {
            FinishReason = MeaiInterop.ToMeaiFinishReason(response.FinishReason),
            Usage = MeaiInterop.ToMeaiUsage(response.Usage),
            ModelId = response.ModelId,
        };
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<Meai.ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<Meai.ChatMessage> messages,
        Meai.ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        Guard.NotNull(messages);
        var aiMessages = ToAiMessages(messages);
        var aiOptions = ToAiOptions(options);

        await foreach (var update in _inner.GetStreamingResponseAsync(aiMessages, aiOptions, cancellationToken).ConfigureAwait(false))
        {
            var meai = new Meai.ChatResponseUpdate
            {
                Role = update.Role is { } role ? MeaiInterop.ToMeaiRole(role) : null,
            };

            if (!string.IsNullOrEmpty(update.TextDelta))
            {
                meai.Contents.Add(new Meai.TextContent(update.TextDelta));
            }

            if (update.FinishReason is { } finish)
            {
                meai.FinishReason = MeaiInterop.ToMeaiFinishReason(finish);
            }

            yield return meai;
        }
    }

    /// <inheritdoc/>
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        Guard.NotNull(serviceType);
        if (serviceKey is null && serviceType.IsInstanceOfType(this))
        {
            return this;
        }

        return null;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        // This wrapper is the MEAI-facing handle a host disposes — cascade to
        // the wrapped client when it owns disposable resources (idempotent:
        // IDisposable implementations tolerate repeated Dispose calls).
        if (_inner is IDisposable disposable)
        {
            disposable.Dispose();
        }
    }

    private static List<ChatMessage> ToAiMessages(IEnumerable<Meai.ChatMessage> messages)
    {
        var result = new List<ChatMessage>();
        foreach (var message in messages)
        {
            result.Add(MeaiInterop.FromMeaiMessage(message));
        }

        return result;
    }

    private static ChatOptions? ToAiOptions(Meai.ChatOptions? options)
    {
        if (options is null)
        {
            return null;
        }

        var ai = new ChatOptions();
        if (options.Temperature is { } temperature) ai.Temperature = temperature;
        if (options.MaxOutputTokens is { } maxTokens) ai.MaxOutputTokens = maxTokens;
        if (options.TopP is { } topP) ai.TopP = topP;
        if (options.TopK is { } topK) ai.TopK = topK;
        if (options.Seed is { } seed) ai.Seed = unchecked((int)seed);
        if (options.StopSequences is { Count: > 0 } stops) ai.StopSequences = stops.ToList();

        var tools = MeaiInterop.FromMeaiTools(options.Tools);
        if (tools.Count > 0)
        {
            ai.Tools = tools;
            ai.ToolChoice = FromMeaiToolMode(options.ToolMode, out var requiredToolName);
            ai.RequiredToolName = requiredToolName;
        }

        return ai;
    }

    private static ToolChoiceMode FromMeaiToolMode(Meai.ChatToolMode? mode, out string? requiredToolName)
    {
        requiredToolName = null;
        switch (mode)
        {
            case Meai.NoneChatToolMode:
                return ToolChoiceMode.None;
            case Meai.RequiredChatToolMode required:
                requiredToolName = required.RequiredFunctionName;
                return ToolChoiceMode.Required;
            default:
                return ToolChoiceMode.Auto;
        }
    }
}
