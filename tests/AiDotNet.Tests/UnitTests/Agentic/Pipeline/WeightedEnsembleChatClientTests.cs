using System;
using System.Collections.Generic;
using System.Net;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Models.Connectors;
using AiDotNet.Agentic.Pipeline;
using AiDotNet.Configuration;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Pipeline;

public sealed class WeightedEnsembleChatClientTests
{
    private static readonly ChatMessage[] Prompt = { ChatMessage.User("improve this") };

    [Fact]
    public async Task SelectionIsIdenticalForTwoClientsThatShareASeed()
    {
        // OpenEvolve seeds Python's process-global generator from the first model's
        // config, so anything else in the process can move the sequence.
        IReadOnlyList<string> first = await SelectionSequenceAsync(seed: 20260901UL, calls: 60);
        IReadOnlyList<string> second = await SelectionSequenceAsync(seed: 20260901UL, calls: 60);
        Assert.Equal(first, second);
    }

    [Fact]
    public async Task DifferentSeedsProduceDifferentSelectionSequences()
    {
        IReadOnlyList<string> first = await SelectionSequenceAsync(seed: 1UL, calls: 60);
        IReadOnlyList<string> second = await SelectionSequenceAsync(seed: 2UL, calls: 60);
        Assert.NotEqual(first, second);
    }

    [Fact]
    public async Task SelectionFrequenciesTrackTheConfiguredWeights()
    {
        var heavy = StubChatClient.Text("heavy", "h");
        var light = StubChatClient.Text("light", "l");
        var ensemble = new WeightedEnsembleChatClient<double>(
            new[]
            {
                new ChatClientEnsembleMember<double>(heavy, 3.0),
                new ChatClientEnsembleMember<double>(light, 1.0)
            },
            new WeightedEnsembleChatClientOptions { Seed = 99UL });

        const int Calls = 4000;
        for (int index = 0; index < Calls; index++) await ensemble.GetResponseAsync(Prompt);

        IReadOnlyDictionary<string, long> counts = ensemble.GetSelectionCounts();
        double heavyShare = counts["heavy"] / (double)Calls;

        Assert.Equal(Calls, counts["heavy"] + counts["light"]);
        Assert.InRange(heavyShare, 0.72, 0.78);
        Assert.Equal(Calls, ensemble.TotalCalls);
    }

    [Fact]
    public async Task SingleMemberEnsembleAlwaysUsesThatMember()
    {
        var only = StubChatClient.Text("only", "answer");
        var ensemble = new WeightedEnsembleChatClient<double>(
            new[] { new ChatClientEnsembleMember<double>(only) });

        for (int index = 0; index < 10; index++) await ensemble.GetResponseAsync(Prompt);
        Assert.Equal(10, only.Calls);
    }

    [Fact]
    public async Task AFailingMemberHandsItsTurnToTheNextHeaviestMember()
    {
        var broken = StubChatClient.AlwaysThrows("broken", () => new InvalidOperationException("boom"));
        var healthy = StubChatClient.Text("healthy", "recovered");
        var ensemble = new WeightedEnsembleChatClient<double>(
            new[]
            {
                new ChatClientEnsembleMember<double>(broken, 1000.0),
                new ChatClientEnsembleMember<double>(healthy, 0.001)
            },
            new WeightedEnsembleChatClientOptions { Seed = 5UL });

        ChatResponse response = await ensemble.GetResponseAsync(Prompt);

        Assert.Equal("recovered", response.Text);
        Assert.True(broken.Calls > 0);
        Assert.Equal(1, healthy.Calls);
    }

    [Fact]
    public async Task FallbackCanBeTurnedOff()
    {
        var broken = StubChatClient.AlwaysThrows("broken", () => new InvalidOperationException("boom"));
        var healthy = StubChatClient.Text("healthy", "recovered");
        var ensemble = new WeightedEnsembleChatClient<double>(
            new[]
            {
                new ChatClientEnsembleMember<double>(broken, 1000.0),
                new ChatClientEnsembleMember<double>(healthy, 0.001)
            },
            new WeightedEnsembleChatClientOptions { Seed = 5UL, FallbackOnError = false });

        await Assert.ThrowsAsync<InvalidOperationException>(() => ensemble.GetResponseAsync(Prompt));
        Assert.Equal(0, healthy.Calls);
    }

    [Fact]
    public async Task WhenEveryMemberFailsTheErrorNamesThemWithoutQuotingProviderText()
    {
        var first = StubChatClient.AlwaysThrows("alpha", () => new InvalidOperationException("key sk-SECRETVALUE rejected"));
        var second = StubChatClient.AlwaysThrows(
            "beta", () => new HttpResponseException(HttpStatusCode.InternalServerError, "endpoint https://internal/x failed"));

        var ensemble = new WeightedEnsembleChatClient<double>(
            new[]
            {
                new ChatClientEnsembleMember<double>(first),
                new ChatClientEnsembleMember<double>(second)
            });

        InvalidOperationException error =
            await Assert.ThrowsAsync<InvalidOperationException>(() => ensemble.GetResponseAsync(Prompt));

        Assert.Contains("alpha", error.Message, StringComparison.Ordinal);
        Assert.Contains("beta", error.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("SECRETVALUE", error.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("internal", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TheAnsweringMemberIsRecordedOnAResponseThatCarriesNoModelId()
    {
        var member = StubChatClient.TextWithoutModelId("raw", "answer");
        var ensemble = new WeightedEnsembleChatClient<double>(
            new[] { new ChatClientEnsembleMember<double>(member, 1.0, null, "named-member") });

        ChatResponse response = await ensemble.GetResponseAsync(Prompt);
        Assert.Equal("named-member", response.ModelId);
    }

    [Fact]
    public async Task AMemberSuppliedModelIdIsLeftAlone()
    {
        var member = StubChatClient.Text("provider-model", "answer");
        var ensemble = new WeightedEnsembleChatClient<double>(
            new[] { new ChatClientEnsembleMember<double>(member, 1.0, null, "named-member") });

        ChatResponse response = await ensemble.GetResponseAsync(Prompt);
        Assert.Equal("provider-model", response.ModelId);
    }

    [Fact]
    public async Task UsageIsAggregatedAcrossEveryCall()
    {
        var member = StubChatClient.Text("m", "a", new ChatUsage(10, 4));
        var ensemble = new WeightedEnsembleChatClient<double>(
            new[] { new ChatClientEnsembleMember<double>(member) });

        for (int index = 0; index < 3; index++) await ensemble.GetResponseAsync(Prompt);

        ChatUsage usage = ensemble.GetUsage();
        Assert.Equal(30, usage.InputTokens);
        Assert.Equal(12, usage.OutputTokens);
        Assert.Equal(42, usage.TotalTokens);
    }

    [Fact]
    public async Task SettingsLayerAsCallThenMemberThenEnsembleDefault()
    {
        var member = StubChatClient.Text("m", "a");
        var ensemble = new WeightedEnsembleChatClient<double>(
            new[]
            {
                new ChatClientEnsembleMember<double>(
                    member, 1.0, new ChatOptions { Temperature = 0.2, MaxOutputTokens = 256 })
            },
            new WeightedEnsembleChatClientOptions
            {
                DefaultChatOptions = new ChatOptions { Temperature = 0.9, TopP = 0.5, MaxOutputTokens = 4096 }
            });

        await ensemble.GetResponseAsync(Prompt, new ChatOptions { MaxOutputTokens = 32 });

        ChatOptions? observed = member.LastOptions;
        Assert.NotNull(observed);
        Assert.Equal<int?>(32, observed?.MaxOutputTokens);
        Assert.Equal<double?>(0.2, observed?.Temperature);
        Assert.Equal<double?>(0.5, observed?.TopP);
    }

    [Fact]
    public async Task EveryMemberCanBeAskedAtOnceAndOneFailureDoesNotLoseTheOthers()
    {
        var good = StubChatClient.Text("good", "yes");
        var broken = StubChatClient.AlwaysThrows("broken", () => new InvalidOperationException("boom"));
        var alsoGood = StubChatClient.Text("also", "maybe");

        var ensemble = new WeightedEnsembleChatClient<double>(
            new[]
            {
                new ChatClientEnsembleMember<double>(good),
                new ChatClientEnsembleMember<double>(broken),
                new ChatClientEnsembleMember<double>(alsoGood)
            },
            new WeightedEnsembleChatClientOptions { MaxParallelism = 2 });

        IReadOnlyList<ChatResponse?> responses = await ensemble.GetAllResponsesAsync(Prompt);

        Assert.Equal(3, responses.Count);
        Assert.Equal("yes", responses[0]?.Text);
        Assert.Null(responses[1]);
        Assert.Equal("maybe", responses[2]?.Text);
    }

    [Fact]
    public void AnEmptyEnsembleIsRejected()
    {
        Assert.Throws<ArgumentException>(() =>
            new WeightedEnsembleChatClient<double>(Array.Empty<ChatClientEnsembleMember<double>>()));
    }

    [Fact]
    public void DuplicateMemberNamesAreRejected()
    {
        // Two members sharing a name would merge in the statistics and hide which
        // one actually answered.
        Assert.Throws<ArgumentException>(() => new WeightedEnsembleChatClient<double>(
            new[]
            {
                new ChatClientEnsembleMember<double>(StubChatClient.Text("a", "x"), 1.0, null, "same"),
                new ChatClientEnsembleMember<double>(StubChatClient.Text("b", "y"), 1.0, null, "same")
            }));
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-1.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void NonPositiveOrNonFiniteWeightsAreRejected(double weight)
    {
        // Upstream normalizes by the sum and raises ZeroDivisionError at call time
        // when every weight is zero.
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new ChatClientEnsembleMember<double>(StubChatClient.Text("a", "x"), weight));
    }

    [Fact]
    public void TheReportedModelIdDefaultsToEnsembleAndCanBeOverridden()
    {
        var members = new[] { new ChatClientEnsembleMember<double>(StubChatClient.Text("a", "x")) };
        Assert.Equal(WeightedEnsembleChatClient<double>.DefaultModelId,
            new WeightedEnsembleChatClient<double>(members).ModelId);
        Assert.Equal("panel",
            new WeightedEnsembleChatClient<double>(members, new WeightedEnsembleChatClientOptions { ModelId = "panel" }).ModelId);
    }

    [Fact]
    public void OptionsAreCopiedSoLaterMutationCannotChangeARunningClient()
    {
        var options = new WeightedEnsembleChatClientOptions { Seed = 1UL, FallbackOnError = true };
        var ensemble = new WeightedEnsembleChatClient<double>(
            new[] { new ChatClientEnsembleMember<double>(StubChatClient.Text("a", "x")) }, options);

        options.FallbackOnError = false;
        Assert.True(ensemble.GetOptions().FallbackOnError);
    }

    private static async Task<IReadOnlyList<string>> SelectionSequenceAsync(ulong seed, int calls)
    {
        var alpha = StubChatClient.TextWithoutModelId("alpha", "a");
        var beta = StubChatClient.TextWithoutModelId("beta", "b");
        var ensemble = new WeightedEnsembleChatClient<double>(
            new[]
            {
                new ChatClientEnsembleMember<double>(alpha),
                new ChatClientEnsembleMember<double>(beta)
            },
            new WeightedEnsembleChatClientOptions { Seed = seed });

        var sequence = new List<string>(calls);
        for (int index = 0; index < calls; index++)
        {
            ChatResponse response = await ensemble.GetResponseAsync(Prompt);
            sequence.Add(response.ModelId ?? "?");
        }

        return sequence;
    }
}
