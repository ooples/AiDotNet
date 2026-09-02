using System.Linq;
using System.Net.Http;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Models.Connectors;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using Newtonsoft.Json.Linq;
using Xunit;

using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

/// <summary>
/// Covers reasoning-model request adaptation. No network call is made anywhere: the connector's request body is
/// built directly through a probe subclass, and the decorator is exercised against the in-memory fake client.
/// </summary>
public sealed class ReasoningModelHandlingTests
{
    private static readonly HttpClient SharedClient = new();

    private static IReadOnlyList<ChatMessage> Conversation() => new List<ChatMessage>
    {
        ChatMessage.System("You are an expert programmer."),
        ChatMessage.User("Improve this program.")
    };

    private static ChatOptions FullOptions() => new()
    {
        Temperature = 0.7,
        TopP = 0.9,
        MaxOutputTokens = 4096,
        Seed = 1234
    };

    [Theory]
    [InlineData("o1")]
    [InlineData("o1-mini")]
    [InlineData("o1-preview-2024-09-12")]
    [InlineData("o3")]
    [InlineData("o3-mini")]
    [InlineData("O3-PRO")]
    [InlineData("o4-mini")]
    [InlineData("gpt-5")]
    [InlineData("gpt-5-nano")]
    [InlineData("gpt-oss-20b")]
    [InlineData("gpt-oss-120b")]
    public void BuiltInRegistryClaimsEveryUpstreamReasoningPrefix(string modelId)
    {
        Assert.True(ReasoningModelProfileRegistry.Default.IsReasoningModel(modelId));
    }

    [Theory]
    [InlineData("gpt-4o")]
    [InlineData("gpt-4o-mini")]
    [InlineData("gpt-4.1")]
    [InlineData("claude-3-5-sonnet")]
    [InlineData("llama-3.1-70b")]
    [InlineData("")]
    public void BuiltInRegistryLeavesOrdinaryModelsAlone(string modelId)
    {
        Assert.False(ReasoningModelProfileRegistry.Default.IsReasoningModel(modelId));
        Assert.Null(ReasoningModelProfileRegistry.Default.Find(modelId));
    }

    [Fact]
    public void OrdinaryModelKeepsTheBaseConnectorRequestBody()
    {
        var client = new ProbeClient("gpt-4o");

        JObject payload = client.Build(Conversation(), FullOptions());

        Assert.Equal(0.7, (double?)payload["temperature"]);
        Assert.Equal(0.9, (double?)payload["top_p"]);
        Assert.Equal(4096, (int?)payload["max_tokens"]);
        Assert.Null(payload["max_completion_tokens"]);
        Assert.Null(payload["reasoning_effort"]);
        Assert.False(client.IsReasoningModel);
        Assert.Empty(client.GetDiagnosticSummary());
    }

    [Fact]
    public void ReasoningModelDropsSamplingControlsAndRenamesTheAnswerCap()
    {
        var client = new ProbeClient("o3-mini");

        JObject payload = client.Build(Conversation(), FullOptions());

        Assert.Null(payload["temperature"]);
        Assert.Null(payload["top_p"]);
        Assert.Null(payload["max_tokens"]);
        Assert.Equal(4096, (int?)payload["max_completion_tokens"]);

        // Upstream keeps the seed on the reasoning path; only temperature and top_p are removed.
        Assert.Equal(1234, (int?)payload["seed"]);
        Assert.True(client.IsReasoningModel);
    }

    [Fact]
    public void ReasoningEffortIsOmittedUnlessConfiguredAndSentAsALowerCaseString()
    {
        var silent = new ProbeClient("gpt-5");
        Assert.Null(silent.Build(Conversation(), FullOptions())["reasoning_effort"]);

        var loud = new ProbeClient("gpt-5", new ReasoningModelOptions { ReasoningEffort = ReasoningEffortLevel.High });
        Assert.Equal("high", (string?)loud.Build(Conversation(), FullOptions())["reasoning_effort"]);
    }

    [Fact]
    public void EveryAdjustmentIsReportedRatherThanSilentlyApplied()
    {
        var sink = new CollectingReasoningModelDiagnosticSink();
        var client = new ProbeClient("o3", new ReasoningModelOptions
        {
            DiagnosticSink = sink,
            ReasoningEffort = ReasoningEffortLevel.Low
        });

        client.Build(Conversation(), FullOptions());

        IReadOnlyList<ReasoningModelDiagnostic> reported = sink.GetRecent();
        Assert.Contains(reported, d =>
            d.Parameter == ReasoningRequestParameter.Temperature &&
            d.Adjustment == ReasoningParameterAdjustment.Dropped);
        Assert.Contains(reported, d =>
            d.Parameter == ReasoningRequestParameter.TopP &&
            d.Adjustment == ReasoningParameterAdjustment.Dropped);
        Assert.Contains(reported, d =>
            d.Parameter == ReasoningRequestParameter.MaxOutputTokens &&
            d.Adjustment == ReasoningParameterAdjustment.Substituted);
        Assert.Contains(reported, d =>
            d.Parameter == ReasoningRequestParameter.ReasoningEffort &&
            d.Adjustment == ReasoningParameterAdjustment.Added);

        foreach (ReasoningModelDiagnostic diagnostic in reported)
        {
            Assert.Equal("o3", diagnostic.ModelId);
            Assert.Equal("openai-o-series", diagnostic.ProfileName);
        }
    }

    [Fact]
    public void AdjustmentsAreRetainedEvenWhenNoSinkIsConfigured()
    {
        var client = new ProbeClient("o3");

        client.Build(Conversation(), FullOptions());
        client.Build(Conversation(), FullOptions());

        IReadOnlyDictionary<string, ReasoningModelDiagnosticSummary> summary = client.GetDiagnosticSummary();
        List<ReasoningModelDiagnosticSummary> dropped = summary.Values
            .Where(s => s.Diagnostic.Parameter == ReasoningRequestParameter.Temperature)
            .ToList();
        Assert.Single(dropped);
        Assert.Equal(2L, dropped[0].Occurrences);
    }

    [Fact]
    public void StrictHandlingFailsInsteadOfChangingTheRequest()
    {
        var client = new ProbeClient("o3", new ReasoningModelOptions
        {
            UnsupportedParameterHandling = UnsupportedChatParameterHandling.Throw
        });

        InvalidOperationException error = Assert.Throws<InvalidOperationException>(
            () => client.Build(Conversation(), FullOptions()));
        Assert.Contains("Temperature", error.Message, StringComparison.Ordinal);
        Assert.Contains("o3", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void DisablingAdaptationRestoresTheUnmodifiedRequest()
    {
        var client = new ProbeClient("o3", new ReasoningModelOptions { Enabled = false });

        JObject payload = client.Build(Conversation(), FullOptions());

        Assert.Equal(0.7, (double?)payload["temperature"]);
        Assert.Equal(4096, (int?)payload["max_tokens"]);
        Assert.Null(payload["max_completion_tokens"]);
    }

    [Fact]
    public void SystemMessagesKeepTheSystemRoleByDefault()
    {
        var client = new ProbeClient("o3");

        JObject payload = client.Build(Conversation(), FullOptions());

        JArray messages = Assert.IsType<JArray>(payload["messages"]);
        Assert.Equal("system", (string?)messages[0]["role"]);
    }

    [Fact]
    public void SystemMessagesCanBeReRoledForAProviderThatDemandsIt()
    {
        var registry = new ReasoningModelProfileRegistry(new[]
        {
            new ReasoningModelProfile("developer-role", new[] { "o3" }, systemMessageRole: "developer")
        });

        var client = new ProbeClient("o3", new ReasoningModelOptions
        {
            Profiles = registry,
            RewriteSystemMessageRole = true
        });

        JObject payload = client.Build(Conversation(), FullOptions());

        JArray messages = Assert.IsType<JArray>(payload["messages"]);
        Assert.Equal("developer", (string?)messages[0]["role"]);
        Assert.Equal("user", (string?)messages[1]["role"]);
    }

    [Fact]
    public async Task DecoratorStripsRejectedOptionsWithoutMutatingTheCallersInstance()
    {
        var inner = new NamedFakeChatClient("o3-mini", "answer");
        var decorated = new ReasoningModelChatClient<double>(inner);
        ChatOptions caller = FullOptions();

        await decorated.GetResponseAsync(Conversation(), caller);

        ChatOptions sent = Assert.IsType<ChatOptions>(inner.LastOptions);
        Assert.Null(sent.Temperature);
        Assert.Null(sent.TopP);
        Assert.Equal(4096, sent.MaxOutputTokens);
        Assert.Equal(1234, sent.Seed);

        // The caller's own options object must come back untouched.
        Assert.Equal(0.7, caller.Temperature);
        Assert.Equal(0.9, caller.TopP);
    }

    [Fact]
    public async Task DecoratorPassesOrdinaryModelsStraightThrough()
    {
        var inner = new NamedFakeChatClient("gpt-4o", "answer");
        var decorated = new ReasoningModelChatClient<double>(inner);
        ChatOptions caller = FullOptions();

        await decorated.GetResponseAsync(Conversation(), caller);

        Assert.Same(caller, inner.LastOptions);
        Assert.Empty(decorated.GetDiagnosticSummary());
    }

    [Fact]
    public void EffortLevelsRoundTripThroughTheirWireValues()
    {
        Assert.Null(ReasoningEffortLevel.Unspecified.ToWireValue());
        Assert.Equal("minimal", ReasoningEffortLevel.Minimal.ToWireValue());
        Assert.Equal("low", ReasoningEffortLevel.Low.ToWireValue());
        Assert.Equal("medium", ReasoningEffortLevel.Medium.ToWireValue());
        Assert.Equal("high", ReasoningEffortLevel.High.ToWireValue());

        Assert.Equal(ReasoningEffortLevel.High, ReasoningEffortLevelExtensions.ParseReasoningEffort("HIGH"));
        Assert.Equal(ReasoningEffortLevel.Unspecified, ReasoningEffortLevelExtensions.ParseReasoningEffort("nonsense"));
        Assert.Equal(ReasoningEffortLevel.Unspecified, ReasoningEffortLevelExtensions.ParseReasoningEffort(null));
    }

    [Fact]
    public void TheAzureConnectorAppliesTheSameRulesToItsDeploymentName()
    {
        var client = new AzureProbeClient("o3-mini");

        JObject payload = client.Build(Conversation(), FullOptions());

        Assert.Null(payload["temperature"]);
        Assert.Null(payload["max_tokens"]);
        Assert.Equal(4096, (int?)payload["max_completion_tokens"]);
        Assert.True(client.IsReasoningModel);
    }

    [Fact]
    public void AnAzureDeploymentNamedAfterNothingIsLeftAlone()
    {
        var client = new AzureProbeClient("reasoning-prod");

        JObject payload = client.Build(Conversation(), FullOptions());

        Assert.Equal(0.7, (double?)payload["temperature"]);
        Assert.Equal(4096, (int?)payload["max_tokens"]);
        Assert.False(client.IsReasoningModel);
    }

    /// <summary>Exposes the Azure connector's request body without performing a network call.</summary>
    private sealed class AzureProbeClient : ReasoningAzureOpenAIChatClient<double>
    {
        public AzureProbeClient(string deploymentName, ReasoningModelOptions? options = null)
            : base(
                "test-key-not-a-real-credential",
                deploymentName,
                "https://example-resource.openai.azure.com",
                "2024-10-21",
                SharedClient,
                options)
        {
        }

        public JObject Build(IReadOnlyList<ChatMessage> messages, ChatOptions options) =>
            BuildRequest(messages, options, stream: false);
    }

    /// <summary>Exposes the connector's request body so the wire format can be asserted without a network call.</summary>
    private sealed class ProbeClient : ReasoningOpenAIChatClient<double>
    {
        public ProbeClient(string modelName, ReasoningModelOptions? options = null)
            : base("test-key-not-a-real-credential", modelName, null, SharedClient, options)
        {
        }

        public JObject Build(IReadOnlyList<ChatMessage> messages, ChatOptions options) =>
            BuildRequest(messages, options, stream: false);
    }

    /// <summary>A fake client that reports a chosen model id and remembers the options it was given.</summary>
    private sealed class NamedFakeChatClient : IChatClient<double>
    {
        private readonly string _response;

        public NamedFakeChatClient(string modelId, string response)
        {
            ModelId = modelId;
            _response = response;
        }

        public string ModelId { get; }

        public ChatOptions? LastOptions { get; private set; }

        public Task<ChatResponse> GetResponseAsync(
            IReadOnlyList<ChatMessage> messages,
            ChatOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            LastOptions = options;
            return Task.FromResult(new ChatResponse(ChatMessage.Assistant(_response)));
        }

        public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
            IReadOnlyList<ChatMessage> messages,
            ChatOptions? options = null,
            CancellationToken cancellationToken = default) =>
            throw new NotSupportedException("The fake chat client does not stream.");
    }
}
