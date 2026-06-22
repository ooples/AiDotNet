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
/// This bridges the two ecosystems for text, sampling, streaming, <em>and tool calling</em>. AiDotNet tool
/// definitions are passed to the MEAI model as schema-only function declarations (via <see cref="MeaiInterop"/>);
/// tool calls the model requests come back as AiDotNet <see cref="ToolCallContent"/> with the finish reason set
/// to <see cref="ChatFinishReason.ToolCalls"/>, and the AiDotNet agent loop executes them — MEAI never invokes
/// the tool itself. Tool results are replayed back to the model as MEAI function-result content.
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

        var assistant = BuildAssistantMessage(response);
        var finish = assistant.ToolCalls.Count > 0 && response.FinishReason is null
            ? ChatFinishReason.ToolCalls
            : MeaiInterop.FromMeaiFinishReason(response.FinishReason);
        return new ChatResponse(assistant, finish, MeaiInterop.FromMeaiUsage(response.Usage), response.ModelId ?? ModelId);
    }

    // Collects assistant text and any tool calls the MEAI model emitted into a single AiDotNet message.
    private static ChatMessage BuildAssistantMessage(Meai.ChatResponse response)
    {
        var contents = new List<AiContent>();
        foreach (var message in response.Messages)
        {
            foreach (var content in message.Contents)
            {
                if (content is Meai.FunctionCallContent call)
                {
                    contents.Add(new ToolCallContent(call.CallId, call.Name, MeaiInterop.ArgumentsToJson(call.Arguments)));
                }
            }
        }

        var text = response.Text;
        if (!string.IsNullOrEmpty(text))
        {
            contents.Insert(0, new TextContent(text));
        }

        if (contents.Count == 0)
        {
            contents.Add(new TextContent(string.Empty));
        }

        return ChatMessage.Assistant(contents);
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
                finishReason = MeaiInterop.FromMeaiFinishReason(reason);
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
            // crash with NullReferenceException inside the interop mapping
            // (preserves the null-guard review fix that landed on master before
            // this branch replaced the text-only path with the full bridge).
            Guard.NotNull(message);

            result.Add(MeaiInterop.ToMeaiMessage(message));
        }

        return result;
    }

    private static Meai.ChatOptions? ToMeaiOptions(ChatOptions? options)
    {
        if (options is null)
        {
            return null;
        }

        var meai = new Meai.ChatOptions();
        if (options.Temperature is { } temperature) meai.Temperature = (float)temperature;
        if (options.MaxOutputTokens is { } maxTokens) meai.MaxOutputTokens = maxTokens;
        if (options.TopP is { } topP) meai.TopP = (float)topP;
        if (options.TopK is { } topK) meai.TopK = topK;
        if (options.Seed is { } seed) meai.Seed = seed;
        if (options.StopSequences is { Count: > 0 } stops) meai.StopSequences = stops.ToList();

        if (options.Tools is { Count: > 0 } tools)
        {
            meai.Tools = MeaiInterop.ToMeaiTools(tools);
            meai.ToolMode = ToMeaiToolMode(options.ToolChoice, options.RequiredToolName);
        }

        return meai;
    }

    private static Meai.ChatToolMode ToMeaiToolMode(ToolChoiceMode? choice, string? requiredToolName) => choice switch
    {
        ToolChoiceMode.None => Meai.ChatToolMode.None,
        ToolChoiceMode.Required => requiredToolName is { Length: > 0 }
            ? Meai.ChatToolMode.RequireSpecific(requiredToolName)
            : Meai.ChatToolMode.RequireAny,
        _ => Meai.ChatToolMode.Auto
    };
}
