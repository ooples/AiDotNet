using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.Agentic.Agents;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Tools;
using AiDotNet.Data.Loaders;
using AiDotNet.Interfaces;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json.Linq;
using Xunit;

using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Covers wiring ConfigureTool into the agents framework: tool accumulation, the trained model exposed as a
/// tool, schema guardrails, model-grounded answer verification, learned tool routing, and the multi-agent mesh.
/// </summary>
public class AgentToolIntegrationTests
{
    /// <summary>A chat client that replays a fixed script of responses, one per call.</summary>
    private sealed class ScriptedChatClient : IChatClient<double>
    {
        private readonly Queue<ChatResponse> _responses;
        public int Calls { get; private set; }
        public ScriptedChatClient(params ChatResponse[] responses) => _responses = new Queue<ChatResponse>(responses);
        public string ModelId => "scripted";
        public Task<ChatResponse> GetResponseAsync(
            IReadOnlyList<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
        {
            Calls++;
            var response = _responses.Count > 0
                ? _responses.Dequeue()
                : new ChatResponse(ChatMessage.Assistant("done"), ChatFinishReason.Stop);
            return Task.FromResult(response);
        }

        public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
            IReadOnlyList<ChatMessage> messages, ChatOptions? options = null,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask;
            yield break;
        }
    }

    /// <summary>An agent that replays a fixed script of results, one per run.</summary>
    private sealed class ScriptedAgent : IAgent<double>
    {
        private readonly Queue<AgentRunResult> _results;
        public int Runs { get; private set; }
        public ScriptedAgent(params AgentRunResult[] results) => _results = new Queue<AgentRunResult>(results);
        public string Name => "scripted";
        public string Description => "test";
        public Task<AgentRunResult> RunAsync(IReadOnlyList<ChatMessage> messages, CancellationToken cancellationToken = default)
        {
            Runs++;
            return Task.FromResult(_results.Dequeue());
        }
    }

    private sealed class EchoTool : AgentToolBase
    {
        public EchoTool() : base("echo", "Echoes text back.", new JObject
        {
            ["type"] = "object",
            ["properties"] = new JObject { ["text"] = new JObject { ["type"] = "string" } },
            ["required"] = new JArray("text"),
        })
        { }

        protected override Task<ToolInvocationResult> InvokeCoreAsync(JObject arguments, CancellationToken cancellationToken)
            => Task.FromResult(ToolInvocationResult.Success(arguments["text"]?.Value<string>() ?? string.Empty));
    }

    private static async Task<Models.Results.AiModelResult<double, Matrix<double>, Vector<double>>> BuildTabularAsync(
        params IAgentTool[] tools)
    {
        int rows = 60, cols = 3;
        var x = new Matrix<double>(rows, cols);
        var y = new Vector<double>(rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++) x[i, j] = Math.Sin((i + j) * 0.2) + i * 0.01;
            y[i] = 2.0 * x[i, 0] - x[i, 2];
        }

        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y));
        foreach (var tool in tools) builder.ConfigureTool(tool);
        return await builder.BuildAsync();
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureTool_Accumulates_AndModelIsExposedAsTool()
    {
        var result = await BuildTabularAsync(new EchoTool());

        Assert.Single(result.ConfiguredTools);
        Assert.Equal("echo", result.ConfiguredTools[0].Name);

        // The trained model is exposed as a callable tool whose output matches the model's own prediction.
        var modelTool = result.CreateModelPredictionTool()
            ?? throw new InvalidOperationException("a tabular model should be exposable as a tool");

        var features = new double[] { 0.3, -0.1, 0.2 };
        var toolResult = await modelTool.InvokeAsync(new JObject { ["features"] = new JArray(features.Cast<object>()) });
        Assert.False(toolResult.IsError);

        var toolPrediction = ModelPredictionTool.TryParseFirstPrediction(toolResult.Content);
        Assert.NotNull(toolPrediction);

        var m = new Matrix<double>(1, 3);
        for (int j = 0; j < 3; j++) m[0, j] = features[j];
        var direct = result.Predict(m)[0];
        Assert.Equal(direct, toolPrediction ?? double.NaN, 6);
    }

    [Fact]
    public async Task GroundedVerifier_CorrectsAnswerThatContradictsTheModel()
    {
        // predict(features) = sum(features); for [1,2,3] that is 6.
        Func<double[], double[]> predict = f => new[] { f.Sum() };

        var toolCall = ChatMessage.Assistant(new AiContent[]
        {
            new ToolCallContent("c1", "predict_model", "{\"features\":[1,2,3]}"),
        });
        var wrongTranscript = new[] { ChatMessage.User("predict"), toolCall, ChatMessage.Assistant("The prediction is 999") };
        var run1 = AgentRunResult.Finished("The prediction is 999", wrongTranscript, 2, null);
        var run2 = AgentRunResult.Finished("The prediction is 6", new[] { ChatMessage.Assistant("The prediction is 6") }, 1, null);

        var inner = new ScriptedAgent(run1, run2);
        var verifier = new GroundedVerifierAgent<double>(inner, predict, "predict_model", maxRefinements: 1);

        var result = await verifier.RunAsync(new[] { ChatMessage.User("predict") });

        Assert.Equal(2, inner.Runs); // the wrong answer forced exactly one refinement.
        Assert.Contains("6", result.FinalText);
        Assert.DoesNotContain("999", result.FinalText);
    }

    [Fact]
    public async Task GroundedVerifier_LeavesGroundedAnswerUntouched()
    {
        Func<double[], double[]> predict = f => new[] { f.Sum() };
        var toolCall = ChatMessage.Assistant(new AiContent[] { new ToolCallContent("c1", "predict_model", "{\"features\":[1,2,3]}") });
        var goodTranscript = new[] { ChatMessage.User("predict"), toolCall, ChatMessage.Assistant("The prediction is 6.0") };
        var run1 = AgentRunResult.Finished("The prediction is 6.0", goodTranscript, 2, null);

        var inner = new ScriptedAgent(run1);
        var verifier = new GroundedVerifierAgent<double>(inner, predict, "predict_model");

        var result = await verifier.RunAsync(new[] { ChatMessage.User("predict") });

        Assert.Equal(1, inner.Runs); // no refinement needed.
        Assert.Contains("6", result.FinalText);
    }

    [Fact]
    public async Task GuardedTool_RejectsInvalidArguments_AndPassesValidOnes()
    {
        var guarded = new GuardedAgentTool(new EchoTool());

        var missing = await guarded.InvokeAsync(new JObject());
        Assert.True(missing.IsError);
        Assert.Contains("required", missing.Content, StringComparison.OrdinalIgnoreCase);

        var wrongType = await guarded.InvokeAsync(new JObject { ["text"] = 42 });
        Assert.True(wrongType.IsError);

        var valid = await guarded.InvokeAsync(new JObject { ["text"] = "hello" });
        Assert.False(valid.IsError);
        Assert.Equal("hello", valid.Content);
    }

    [Fact]
    public void LearnedRouter_PrefersToolsThatWorkedOnSimilarRequests()
    {
        var router = new LearnedToolRouter();
        const string query = "predict the sales number for next quarter";
        for (int i = 0; i < 3; i++)
        {
            router.Record(query, "predict_model", 1.0);
            router.Record(query, "web_search", 0.0);
        }

        var ranked = router.RankTools("predict the sales number for the next quarter", new[] { "web_search", "predict_model" });
        Assert.Equal("predict_model", ranked[0]);

        var hint = router.BuildHint(query, new[] { "web_search", "predict_model" });
        Assert.NotNull(hint);
        Assert.Contains("predict_model", hint);

        // Reward table round-trips through export/import.
        var other = new LearnedToolRouter();
        other.Import(router.Export());
        Assert.Equal(1.0, other.ScoreFor(query, "predict_model"), 6);
        Assert.Equal(0.0, other.ScoreFor(query, "web_search"), 6);
    }

    [Fact(Timeout = 120000)]
    public async Task RunAgentAsync_CallsModelTool_AndRouterRecordsOutcome()
    {
        var result = await BuildTabularAsync();

        var client = new ScriptedChatClient(
            new ChatResponse(
                ChatMessage.Assistant(new AiContent[] { new ToolCallContent("c1", "predict_model", "{\"features\":[0.1,0.2,0.3]}") }),
                ChatFinishReason.ToolCalls),
            new ChatResponse(ChatMessage.Assistant("The prediction is available above."), ChatFinishReason.Stop));

        var router = new LearnedToolRouter();
        var run = await result.RunAgentAsync(client, "what does the model predict for these features?",
            guardTools: true, groundWithModel: false, router: router);

        Assert.True(run.Completed);
        Assert.True(client.Calls >= 2); // one call to request the tool, one to finish.
        // The router recorded that predict_model was used for this request.
        Assert.Equal(1.0, router.ScoreFor("what does the model predict for these features?", "predict_model"), 6);
    }

    [Fact(Timeout = 120000)]
    public async Task Supervisor_WithModelAnalyst_RunsToCompletion()
    {
        var result = await BuildTabularAsync();
        var client = new ScriptedChatClient(
            new ChatResponse(ChatMessage.Assistant("The team has the answer."), ChatFinishReason.Stop));

        var analyst = result.CreateModelAnalystAgent(client)
            ?? throw new InvalidOperationException("a tabular model should produce a model-analyst agent");

        var supervisor = result.CreateSupervisor(client, new[] { analyst });
        var run = await supervisor.RunAsync("coordinate the team");

        Assert.True(run.Completed);
        Assert.Equal("The team has the answer.", run.FinalText);
    }
}
