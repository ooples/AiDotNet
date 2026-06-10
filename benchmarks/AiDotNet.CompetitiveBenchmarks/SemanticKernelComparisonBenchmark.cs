using System.Collections.Generic;
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
/// Semantic Kernel's <c>Kernel.InvokePromptAsync</c>, both driving an identical in-process mock model (no
/// network) with one tool registered. Measures each framework's orchestration/dispatch cost — wall-clock and
/// allocations (<see cref="MemoryDiagnoserAttribute"/>) — not model quality or provider latency.
/// </summary>
/// <remarks>
/// Run with <c>dotnet run -c Release --project benchmarks/AiDotNet.CompetitiveBenchmarks -- --filter *SemanticKernel*</c>.
/// Both sides return the same fixed assistant answer and expose one "add" tool, so the only variable is the
/// framework machinery wrapping the call.
/// </remarks>
[MemoryDiagnoser]
public class SemanticKernelComparisonBenchmark
{
    private static readonly AiChatMessage[] Messages = { AiChatMessage.User("What is 2 + 3?") };

    private readonly AgentExecutor<double> _aiDotNetAgent;
    private readonly Kernel _semanticKernel;

    public SemanticKernelComparisonBenchmark()
    {
        // AiDotNet: an executor over a fixed in-process client with one tool.
        var tools = new ToolCollection().AddDelegate("add", "Adds two integers.", (int a, int b) => a + b);
        _aiDotNetAgent = new AgentExecutor<double>(new FixedChatClient("a concise answer"), tools);

        // Semantic Kernel: a kernel over a mock chat-completion service with one plugin function.
        var builder = Kernel.CreateBuilder();
        builder.Services.AddSingleton<IChatCompletionService>(new MockChatCompletionService());
        _semanticKernel = builder.Build();
        _semanticKernel.Plugins.AddFromFunctions("tools", new[]
        {
            KernelFunctionFactory.CreateFromMethod((int a, int b) => a + b, "add", "Adds two integers."),
        });
    }

    [Benchmark(Baseline = true, Description = "AiDotNet AgentExecutor — 1 tool-enabled turn")]
    public async Task<string> AiDotNet_AgentTurn() => (await _aiDotNetAgent.RunAsync(Messages)).FinalText;

    [Benchmark(Description = "Semantic Kernel InvokePromptAsync — 1 tool-enabled turn")]
    public async Task<string> SemanticKernel_AgentTurn()
    {
        var result = await _semanticKernel.InvokePromptAsync("What is 2 + 3?");
        return result.ToString();
    }

    // Deterministic in-process AiDotNet chat client returning a fixed response.
    private sealed class FixedChatClient : IChatClient<double>
    {
        private readonly ChatResponse _response;

        public FixedChatClient(string text) =>
            _response = new ChatResponse(AiChatMessage.Assistant(text), ChatFinishReason.Stop, new ChatUsage(8, 4), "bench");

        public string ModelId => "bench";

        public Task<ChatResponse> GetResponseAsync(IReadOnlyList<AiChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default) =>
            Task.FromResult(_response);

        public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IReadOnlyList<AiChatMessage> messages, ChatOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask.ConfigureAwait(false);
            yield return ChatResponseUpdate.ForText(_response.Text);
            yield return ChatResponseUpdate.ForFinish(ChatFinishReason.Stop, _response.Usage);
        }
    }

    // Deterministic in-process Semantic Kernel chat-completion service returning a fixed response.
    private sealed class MockChatCompletionService : IChatCompletionService
    {
        private static readonly ChatMessageContent Reply = new(AuthorRole.Assistant, "a concise answer");

        public IReadOnlyDictionary<string, object?> Attributes { get; } = new Dictionary<string, object?>();

        public Task<IReadOnlyList<ChatMessageContent>> GetChatMessageContentsAsync(
            ChatHistory chatHistory,
            PromptExecutionSettings? executionSettings = null,
            Kernel? kernel = null,
            CancellationToken cancellationToken = default) =>
            Task.FromResult<IReadOnlyList<ChatMessageContent>>(new[] { Reply });

        public async IAsyncEnumerable<StreamingChatMessageContent> GetStreamingChatMessageContentsAsync(
            ChatHistory chatHistory,
            PromptExecutionSettings? executionSettings = null,
            Kernel? kernel = null,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask.ConfigureAwait(false);
            yield return new StreamingChatMessageContent(AuthorRole.Assistant, "a concise answer");
        }
    }
}
