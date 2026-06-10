using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Agentic.Agents;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Tools;
using AiChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.CompetitiveBenchmarks;

/// <summary>
/// Cross-framework comparison against LangGraph. BenchmarkDotNet cannot host a Python framework, so this runner
/// times the same single tool-enabled turn in both runtimes in comparable units (mean microseconds/iteration):
/// AiDotNet's <c>AgentExecutor</c> in-process, and LangGraph via the <c>langgraph_sidecar.py</c> subprocess.
/// If Python or langgraph is unavailable it prints clear setup instructions and skips the LangGraph half
/// instead of failing — the AiDotNet number is still reported.
/// </summary>
internal static class LangGraphSidecarRunner
{
    public static async Task RunAsync(string[] args)
    {
        var iterations = args.Length > 0 && int.TryParse(args[0], out var n) && n > 0 ? n : 2000;

        Console.WriteLine($"Cross-framework single-tool-turn comparison ({iterations} iterations each)\n");

        var aiDotNetMeanUs = await TimeAiDotNetAsync(iterations);
        Console.WriteLine($"  AiDotNet AgentExecutor : {aiDotNetMeanUs:F3} µs/turn");

        var langGraph = RunLangGraphSidecar(iterations);
        if (langGraph is { } meanUs)
        {
            Console.WriteLine($"  LangGraph StateGraph   : {meanUs:F3} µs/turn");
            var ratio = meanUs / aiDotNetMeanUs;
            Console.WriteLine($"\n  AiDotNet is {ratio:F2}x {(ratio >= 1 ? "faster" : "slower")} on this scenario.");
        }
    }

    private static async Task<double> TimeAiDotNetAsync(int iterations)
    {
        var tools = new ToolCollection().AddDelegate("add", "Adds two integers.", (int a, int b) => a + b);
        var agent = new AgentExecutor<double>(new FixedChatClient("a concise answer"), tools);
        var messages = new[] { AiChatMessage.User("What is 2 + 3?") };

        for (var i = 0; i < 50; i++)
        {
            await agent.RunAsync(messages);
        }

        var stopwatch = Stopwatch.StartNew();
        for (var i = 0; i < iterations; i++)
        {
            await agent.RunAsync(messages);
        }

        stopwatch.Stop();
        return stopwatch.Elapsed.TotalMilliseconds * 1000.0 / iterations;
    }

    private static double? RunLangGraphSidecar(int iterations)
    {
        var scriptPath = Path.Combine(AppContext.BaseDirectory, "langgraph_sidecar.py");
        if (!File.Exists(scriptPath))
        {
            Console.WriteLine($"  LangGraph              : skipped (sidecar script not found at {scriptPath}).");
            return null;
        }

        foreach (var python in new[] { "python3", "python" })
        {
            try
            {
                var psi = new ProcessStartInfo(python, $"\"{scriptPath}\" {iterations}")
                {
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                };

                using var process = Process.Start(psi);
                if (process is null)
                {
                    continue;
                }

                var output = process.StandardOutput.ReadToEnd();
                process.WaitForExit();

                var meanUs = ParseMeanMicroseconds(output);
                if (meanUs is null)
                {
                    Console.WriteLine(
                        "  LangGraph              : skipped (langgraph not installed in the Python environment). " +
                        "Install with: pip install langgraph");
                }

                return meanUs;
            }
            catch (Exception)
            {
                // Try the next python launcher name.
            }
        }

        Console.WriteLine(
            "  LangGraph              : skipped (no Python interpreter found). Install Python 3 and " +
            "`pip install langgraph`, then re-run with `-- --langgraph`.");
        return null;
    }

    private static double? ParseMeanMicroseconds(string json)
    {
        // Minimal extraction so the runner has no JSON dependency: find "mean_microseconds": <number>.
        const string key = "\"mean_microseconds\":";
        var index = json.IndexOf(key, StringComparison.Ordinal);
        if (index < 0)
        {
            return null;
        }

        var rest = json.Substring(index + key.Length).TrimStart();
        var end = 0;
        while (end < rest.Length && (char.IsDigit(rest[end]) || rest[end] == '.' || rest[end] == '-'))
        {
            end++;
        }

        return double.TryParse(rest.Substring(0, end), System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture, out var value)
            ? value
            : null;
    }

    // Deterministic in-process AiDotNet chat client returning a fixed response (mirrors the SK benchmark mock).
    private sealed class FixedChatClient : IChatClient<double>
    {
        private readonly ChatResponse _response;

        public FixedChatClient(string text) =>
            _response = new ChatResponse(AiChatMessage.Assistant(text), ChatFinishReason.Stop, new ChatUsage(8, 4), "bench");

        public string ModelId => "bench";

        public Task<ChatResponse> GetResponseAsync(System.Collections.Generic.IReadOnlyList<AiChatMessage> messages, ChatOptions? options = null, System.Threading.CancellationToken cancellationToken = default) =>
            Task.FromResult(_response);

        public async System.Collections.Generic.IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(System.Collections.Generic.IReadOnlyList<AiChatMessage> messages, ChatOptions? options = null, [System.Runtime.CompilerServices.EnumeratorCancellation] System.Threading.CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask.ConfigureAwait(false);
            yield return ChatResponseUpdate.ForText(_response.Text);
            yield return ChatResponseUpdate.ForFinish(ChatFinishReason.Stop, _response.Usage);
        }
    }
}
