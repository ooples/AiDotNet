using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Pipeline;

namespace AiDotNetTests.UnitTests.Agentic.Pipeline;

/// <summary>A scriptable in-process chat client: no network, no provider, fully deterministic.</summary>
internal sealed class StubChatClient : IChatClient<double>
{
    private readonly Func<int, CancellationToken, Task<ChatResponse>> _handler;
    private readonly List<ChatOptions?> _observedOptions = new();
    private readonly object _gate = new();
    private int _calls;

    private StubChatClient(string modelId, Func<int, CancellationToken, Task<ChatResponse>> handler)
    {
        ModelId = modelId;
        _handler = handler;
    }

    public string ModelId { get; }

    public int Calls
    {
        get
        {
            lock (_gate)
            {
                return _calls;
            }
        }
    }

    public IReadOnlyList<ChatOptions?> ObservedOptions
    {
        get
        {
            lock (_gate)
            {
                return new List<ChatOptions?>(_observedOptions);
            }
        }
    }

    public ChatOptions? LastOptions
    {
        get
        {
            lock (_gate)
            {
                return _observedOptions.Count == 0 ? null : _observedOptions[_observedOptions.Count - 1];
            }
        }
    }

    public static StubChatClient Text(string modelId, string text, ChatUsage? usage = null) =>
        new(modelId, (_, _) => Task.FromResult(new ChatResponse(
            ChatMessage.Assistant(text), ChatFinishReason.Stop, usage, modelId)));

    public static StubChatClient TextWithoutModelId(string modelId, string text) =>
        new(modelId, (_, _) => Task.FromResult(new ChatResponse(ChatMessage.Assistant(text))));

    public static StubChatClient AlwaysThrows(string modelId, Func<Exception> factory) =>
        new(modelId, (_, _) => throw factory());

    public static StubChatClient ThrowsThenText(string modelId, int failures, Func<Exception> factory, string text) =>
        new(modelId, (attempt, _) => attempt < failures
            ? throw factory()
            : Task.FromResult(new ChatResponse(ChatMessage.Assistant(text), ChatFinishReason.Stop, null, modelId)));

    public static StubChatClient Delays(string modelId, TimeSpan delay, string text) =>
        new(modelId, async (_, token) =>
        {
            await Task.Delay(delay, token).ConfigureAwait(false);
            return new ChatResponse(ChatMessage.Assistant(text), ChatFinishReason.Stop, null, modelId);
        });

    public static StubChatClient WithToolCall(string modelId, string callId, string toolName, string argumentsJson) =>
        new(modelId, (_, _) => Task.FromResult(new ChatResponse(
            ChatMessage.Assistant(new AiContent[]
            {
                new TextContent("calling"),
                new ToolCallContent(callId, toolName, argumentsJson)
            }),
            ChatFinishReason.ToolCalls,
            new ChatUsage(11, 22),
            modelId)));

    public Task<ChatResponse> GetResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        int attempt;
        lock (_gate)
        {
            attempt = _calls;
            _calls++;
            _observedOptions.Add(options);
        }

        return _handler(attempt, cancellationToken);
    }

    public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default) =>
        throw new NotSupportedException("The stub chat client does not stream.");
}

/// <summary>Records the order in which pipeline stages ran, so layering can be asserted.</summary>
internal sealed class RecordingChatMiddleware : IChatMiddleware
{
    private readonly IList<string> _log;
    private readonly string _name;

    public RecordingChatMiddleware(IList<string> log, string name)
    {
        _log = log;
        _name = name;
    }

    public async Task<ChatResponse> InvokeAsync(
        ChatRequestContext context,
        ChatPipelineDelegate next,
        CancellationToken cancellationToken)
    {
        _log.Add(_name + "-before");
        ChatResponse response = await next(context, cancellationToken).ConfigureAwait(false);
        _log.Add(_name + "-after");
        return response;
    }
}
