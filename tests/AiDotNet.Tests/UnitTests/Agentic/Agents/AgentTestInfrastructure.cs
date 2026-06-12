using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Tools;
using Newtonsoft.Json.Linq;

namespace AiDotNetTests.UnitTests.Agentic.Agents
{
    /// <summary>
    /// A deterministic in-memory <see cref="IChatClient{T}"/> for agent tests. It returns scripted
    /// responses driven by a function of (call index, conversation so far), and records every request it
    /// received so tests can assert on what the agent actually sent the model.
    /// </summary>
    internal sealed class ScriptedChatClient<T> : IChatClient<T>
    {
        private readonly Func<int, IReadOnlyList<ChatMessage>, ChatResponse> _script;
        private int _callCount;

        public ScriptedChatClient(Func<int, IReadOnlyList<ChatMessage>, ChatResponse> script)
        {
            _script = script ?? throw new ArgumentNullException(nameof(script));
        }

        /// <summary>Builds a client that replays a fixed sequence, repeating the last entry once exhausted.</summary>
        public static ScriptedChatClient<T> Sequence(params ChatResponse[] responses)
        {
            if (responses is null || responses.Length == 0)
            {
                throw new ArgumentException("At least one response is required.", nameof(responses));
            }

            return new ScriptedChatClient<T>((index, _) =>
                responses[Math.Min(index, responses.Length - 1)]);
        }

        public string ModelId => "scripted-test-client";

        /// <summary>The conversation snapshot passed to each call, in call order.</summary>
        public List<IReadOnlyList<ChatMessage>> Requests { get; } = new();

        /// <summary>The options passed to each call, in call order.</summary>
        public List<ChatOptions?> ReceivedOptions { get; } = new();

        public int CallCount => _callCount;

        public Task<ChatResponse> GetResponseAsync(
            IReadOnlyList<ChatMessage> messages,
            ChatOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (messages is null)
            {
                throw new ArgumentNullException(nameof(messages));
            }

            var snapshot = messages.ToList();
            Requests.Add(snapshot);
            ReceivedOptions.Add(options);
            var response = _script(_callCount, snapshot);
            _callCount++;
            return Task.FromResult(response);
        }

        public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
            IReadOnlyList<ChatMessage> messages,
            ChatOptions? options = null,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var response = await GetResponseAsync(messages, options, cancellationToken).ConfigureAwait(false);
            if (response.Message.Text.Length > 0)
            {
                yield return ChatResponseUpdate.ForText(response.Message.Text);
            }

            yield return ChatResponseUpdate.ForFinish(response.FinishReason, response.Usage);
        }
    }

    /// <summary>
    /// A tool that records every invocation's arguments and returns a result computed from them, so tests
    /// can assert both that the agent ran the tool and with what arguments.
    /// </summary>
    internal sealed class RecordingTool : IAgentTool
    {
        private readonly Func<JObject, string> _implementation;

        public RecordingTool(string name, string description, Func<JObject, string> implementation)
        {
            Name = name ?? throw new ArgumentNullException(nameof(name));
            Description = description ?? throw new ArgumentNullException(nameof(description));
            _implementation = implementation ?? throw new ArgumentNullException(nameof(implementation));
        }

        public string Name { get; }

        public string Description { get; }

        public JObject ParametersSchema { get; } =
            new JObject { ["type"] = "object", ["properties"] = new JObject() };

        public List<JObject> Invocations { get; } = new();

        public AiToolDefinition ToDefinition() => new(Name, Description, ParametersSchema);

        public Task<ToolInvocationResult> InvokeAsync(JObject arguments, CancellationToken cancellationToken = default)
        {
            Invocations.Add(arguments);
            return Task.FromResult(ToolInvocationResult.Success(_implementation(arguments)));
        }
    }

    /// <summary>Factory helpers for building scripted <see cref="ChatResponse"/> values.</summary>
    internal static class ChatResponses
    {
        public static ChatResponse Text(string text, ChatUsage? usage = null) =>
            new(ChatMessage.Assistant(text), ChatFinishReason.Stop, usage);

        public static ChatResponse ToolCall(string callId, string toolName, string argumentsJson, ChatUsage? usage = null) =>
            new(
                ChatMessage.Assistant(new AiContent[] { new ToolCallContent(callId, toolName, argumentsJson) }),
                ChatFinishReason.ToolCalls,
                usage);
    }
}
