using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Agents;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Pipeline;
using BenchmarkDotNet.Attributes;
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Benchmarks;

/// <summary>
/// Micro-benchmarks for the agentic orchestration overhead, using a deterministic in-process chat client (no
/// network) so they measure AiDotNet's own dispatch cost rather than provider latency.
/// </summary>
/// <remarks>
/// <para>
/// Covered: the agent tool-calling loop and the chat middleware pipeline (telemetry + guardrail). Run with
/// <c>dotnet run -c Release -- --filter *AgenticBenchmarks*</c>. (A local-generation throughput benchmark over
/// <c>LocalEngineChatClient</c> is a natural addition once the benchmark project references the tensor
/// package directly.)
/// </para>
/// <para>
/// <b>Head-to-head vs Semantic Kernel / LangGraph:</b> a cross-framework comparison runs the equivalent
/// scenario (a single tool-calling turn against the same mock model), comparing wall-clock + allocations.
/// Those frameworks are external to this repo, so that comparison runs in a dedicated environment with SK
/// (NuGet) and LangGraph (Python, via a sidecar) installed; this harness is the AiDotNet side of it.
/// </para>
/// </remarks>
[MemoryDiagnoser]
public class AgenticBenchmarks
{
    private static readonly ChatMessage[] Messages = { ChatMessage.User("hello") };
    private static readonly FixedChatClient Client = new("a concise answer", new ChatUsage(8, 4));

    private readonly AgentExecutor<double> _agent = new(Client);

    private readonly MiddlewareChatClient<double> _middlewareClient = new(Client, new IChatMiddleware[]
    {
        new TelemetryChatMiddleware(),
        new ContentSafetyMiddleware(new DenyListContentModerator(new[] { "forbidden" })),
    });

    [Benchmark(Description = "Agent tool-calling loop (no tools, 1 model call)")]
    public async Task<string> AgentLoop() => (await _agent.RunAsync(Messages)).FinalText;

    [Benchmark(Description = "Chat middleware pipeline (telemetry + guardrail)")]
    public async Task<ChatResponse> MiddlewarePipeline() => await _middlewareClient.GetResponseAsync(Messages);

    // Deterministic in-process chat client returning a fixed response.
    private sealed class FixedChatClient : IChatClient<double>
    {
        private readonly ChatResponse _response;

        public FixedChatClient(string text, ChatUsage usage) =>
            _response = new ChatResponse(ChatMessage.Assistant(text), ChatFinishReason.Stop, usage, "bench");

        public string ModelId => "bench";

        public Task<ChatResponse> GetResponseAsync(IReadOnlyList<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default) =>
            Task.FromResult(_response);

        public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IReadOnlyList<ChatMessage> messages, ChatOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask.ConfigureAwait(false);
            yield return ChatResponseUpdate.ForText(_response.Text);
            yield return ChatResponseUpdate.ForFinish(ChatFinishReason.Stop, _response.Usage);
        }
    }
}
