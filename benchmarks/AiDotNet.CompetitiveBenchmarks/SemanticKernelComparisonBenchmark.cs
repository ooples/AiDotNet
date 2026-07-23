using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Agents;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Tools;
using BenchmarkDotNet.Attributes;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using AiChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.CompetitiveBenchmarks;

/// <summary>
/// Head-to-head overhead comparison of a single tool-enabled chat turn: AiDotNet's <c>AgentExecutor</c> versus
/// Semantic Kernel's chat-completion + function-calling machinery, both driving an identical in-process
/// scripted model (no network) with one tool registered. Measures each framework's orchestration/dispatch cost
/// — wall-clock and allocations (<see cref="MemoryDiagnoserAttribute"/>) — not model quality or provider
/// latency.
/// </summary>
/// <remarks>
/// Run with <c>dotnet run -c Release --project benchmarks/AiDotNet.CompetitiveBenchmarks -- --filter *SemanticKernel*</c>.
/// Both sides execute the same deterministic two-step script: the model's first response requests
/// <c>add(2, 3)</c>, the framework runs the tool and feeds the result back, and the second response returns
/// the final assistant text — a complete tool-call round-trip, so the only variable is the framework
/// machinery wrapping it.
/// </remarks>
[MemoryDiagnoser]
public class SemanticKernelComparisonBenchmark
{
    private static readonly AiChatMessage[] Messages = { AiChatMessage.User("What is 2 + 3?") };

    private readonly AgentExecutor<double> _aiDotNetAgent;
    private readonly Kernel _semanticKernel;
    private readonly IChatCompletionService _semanticKernelChat;

    public SemanticKernelComparisonBenchmark()
    {
        // AiDotNet: an executor over a scripted two-step client with one tool.
        var tools = new ToolCollection().AddDelegate("add", "Adds two integers.", (int a, int b) => a + b);
        _aiDotNetAgent = new AgentExecutor<double>(new ScriptedToolCallClient("the sum is 5"), tools);

        // Semantic Kernel: a kernel over a scripted two-step chat-completion service with one plugin function.
        var builder = Kernel.CreateBuilder();
        builder.Services.AddSingleton<IChatCompletionService>(new MockChatCompletionService());
        _semanticKernel = builder.Build();
        _semanticKernel.Plugins.AddFromFunctions("tools", new[]
        {
            KernelFunctionFactory.CreateFromMethod((int a, int b) => a + b, "add", "Adds two integers."),
        });
        _semanticKernelChat = _semanticKernel.GetRequiredService<IChatCompletionService>();
    }

    [Benchmark(Baseline = true, Description = "AiDotNet AgentExecutor — 1 tool-call round-trip")]
    public async Task<string> AiDotNet_AgentTurn() => (await _aiDotNetAgent.RunAsync(Messages)).FinalText;

    [Benchmark(Description = "Semantic Kernel chat + function calling — 1 tool-call round-trip")]
    public async Task<string> SemanticKernel_AgentTurn()
    {
        // Semantic Kernel's documented manual function-invocation loop
        // (FunctionCallContent.GetFunctionCalls / InvokeAsync / ToChatMessage):
        // the comparable framework machinery to AgentExecutor's native loop.
        var history = new ChatHistory();
        history.AddUserMessage("What is 2 + 3?");

        while (true)
        {
            var responses = await _semanticKernelChat.GetChatMessageContentsAsync(history, null, _semanticKernel);
            var message = responses[0];
            var functionCalls = FunctionCallContent.GetFunctionCalls(message).ToList();
            if (functionCalls.Count == 0)
            {
                return message.ToString();
            }

            history.Add(message);
            foreach (var functionCall in functionCalls)
            {
                var result = await functionCall.InvokeAsync(_semanticKernel);
                history.Add(result.ToChatMessage());
            }
        }
    }

    // Deterministic two-step AiDotNet chat client: first call requests add(2, 3); after a tool result is
    // supplied back, returns the final answer. State-free (keyed off the conversation) so every benchmark
    // iteration replays the identical script.
    private sealed class ScriptedToolCallClient : IChatClient<double>
    {
        private readonly ChatResponse _toolCallResponse;
        private readonly ChatResponse _finalResponse;

        public ScriptedToolCallClient(string finalText)
        {
            _toolCallResponse = new ChatResponse(
                AiChatMessage.Assistant(new AiContent[] { new ToolCallContent("call-1", "add", "{\"a\":2,\"b\":3}") }),
                ChatFinishReason.ToolCalls,
                new ChatUsage(8, 4),
                "bench");
            _finalResponse = new ChatResponse(
                AiChatMessage.Assistant(finalText), ChatFinishReason.Stop, new ChatUsage(12, 4), "bench");
        }

        public string ModelId => "bench";

        public Task<ChatResponse> GetResponseAsync(IReadOnlyList<AiChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
        {
            // Tool result already present → the second (final) scripted step.
            var hasToolResult = false;
            foreach (var message in messages)
            {
                if (message.Role == ChatRole.Tool)
                {
                    hasToolResult = true;
                    break;
                }
            }

            return Task.FromResult(hasToolResult ? _finalResponse : _toolCallResponse);
        }

        public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IReadOnlyList<AiChatMessage> messages, ChatOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask.ConfigureAwait(false);
            var response = await GetResponseAsync(messages, options, cancellationToken).ConfigureAwait(false);
            yield return ChatResponseUpdate.ForText(response.Text);
            yield return ChatResponseUpdate.ForFinish(response.FinishReason, response.Usage);
        }
    }

    // Deterministic two-step Semantic Kernel chat-completion service mirroring the same script.
    private sealed class MockChatCompletionService : IChatCompletionService
    {
        public IReadOnlyDictionary<string, object?> Attributes { get; } = new Dictionary<string, object?>();

        public Task<IReadOnlyList<ChatMessageContent>> GetChatMessageContentsAsync(
            ChatHistory chatHistory,
            PromptExecutionSettings? executionSettings = null,
            Kernel? kernel = null,
            CancellationToken cancellationToken = default)
        {
            // A function result in the history means the tool leg already ran → final answer.
            foreach (var message in chatHistory)
            {
                if (message.Items.OfType<FunctionResultContent>().Any())
                {
                    return Task.FromResult<IReadOnlyList<ChatMessageContent>>(new[]
                    {
                        new ChatMessageContent(AuthorRole.Assistant, "the sum is 5"),
                    });
                }
            }

            var toolCallMessage = new ChatMessageContent(AuthorRole.Assistant, content: null);
            toolCallMessage.Items.Add(new FunctionCallContent(
                functionName: "add",
                pluginName: "tools",
                id: "call-1",
                arguments: new KernelArguments { ["a"] = 2, ["b"] = 3 }));
            return Task.FromResult<IReadOnlyList<ChatMessageContent>>(new[] { toolCallMessage });
        }

        public async IAsyncEnumerable<StreamingChatMessageContent> GetStreamingChatMessageContentsAsync(
            ChatHistory chatHistory,
            PromptExecutionSettings? executionSettings = null,
            Kernel? kernel = null,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask.ConfigureAwait(false);
            yield return new StreamingChatMessageContent(AuthorRole.Assistant, "the sum is 5");
        }
    }
}
