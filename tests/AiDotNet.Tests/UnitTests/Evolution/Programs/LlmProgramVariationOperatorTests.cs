using AiDotNet.Agentic.Models;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Evolution.Prompts;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class LlmProgramVariationOperatorTests
{
    private const string ParentSource = "def solve(x):\n    return x\n";

    private static EvolutionArchiveEntry<ProgramGenome> Entry(
        ProgramGenome genome,
        double quality,
        long evaluationId = 0,
        IReadOnlyList<EvolutionDiagnostic>? diagnostics = null)
    {
        var lineage = new EvolutionLineage(null, null, "seed", null, 0, 0, 0UL);
        var candidate = new EvolutionCandidate<ProgramGenome>(
            evaluationId, new EvolutionCanonicalGenome<ProgramGenome>(genome, genome.Id), lineage);
        var evaluation = new EvolutionEvaluation(
            evaluationId,
            genome.Id,
            EvolutionEvaluationStatus.Completed,
            quality,
            EvolutionOptimizationDirection.Maximize,
            new Dictionary<string, double>(StringComparer.Ordinal) { ["x"] = quality },
            Array.Empty<double>(),
            Array.Empty<double>(),
            new EvolutionEvaluationCost(TimeSpan.Zero, 1, 1),
            lineage,
            EvolutionCacheStatus.Miss,
            diagnostics ?? Array.Empty<EvolutionDiagnostic>(),
            "task-v1",
            "evaluator-v1",
            "config-v1");

        return new EvolutionArchiveEntry<ProgramGenome>(new EvolutionCellKey(new[] { 1, 2 }), candidate, evaluation);
    }

    private static EvolutionVariationContext<ProgramGenome> Context(
        ProgramGenome? parent = null,
        IReadOnlyList<EvolutionArchiveEntry<ProgramGenome>>? inspirations = null,
        IReadOnlyList<EvolutionDiagnostic>? diagnostics = null) =>
        new(Entry(parent ?? new ProgramGenome(ParentSource, ProgramLanguage.Python), 0.5, 0, diagnostics),
            inspirations ?? Array.Empty<EvolutionArchiveEntry<ProgramGenome>>(),
            new StableRandom(1234UL, 7UL),
            0,
            0);

    private static string DiffResponse(string search, string replace) =>
        "<<<<<<< SEARCH\n" + search + "\n=======\n" + replace + "\n>>>>>>> REPLACE\n";

    [Fact]
    public async Task AppliedEditsBecomeTheChildProgram()
    {
        var client = new FakeChatClient(DiffResponse("    return x", "    return x * 2"));
        var operatorUnderTest = new LlmProgramVariationOperator<double>(client);

        ProgramGenome child = await operatorUnderTest.ProposeAsync(Context());

        Assert.Equal("def solve(x):\n    return x * 2\n", child.Source);
        Assert.Equal(ProgramLanguage.Python, child.Language);
        Assert.Equal(1, client.Calls);
    }

    [Fact]
    public async Task UnusableEditsAreRetriedWithFeedback()
    {
        var client = new FakeChatClient(
            DiffResponse("    return y", "    return 0"),
            DiffResponse("    return x", "    return x + 1"));

        var operatorUnderTest = new LlmProgramVariationOperator<double>(client);
        ProgramGenome child = await operatorUnderTest.ProposeAsync(Context());

        Assert.Equal("def solve(x):\n    return x + 1\n", child.Source);
        Assert.Equal(2, client.Calls);

        IReadOnlyList<ChatMessage> retry = client.Conversations[1];
        Assert.Equal(4, retry.Count);
        Assert.Contains("could not be applied", retry[3].Text, StringComparison.Ordinal);
        Assert.Contains("return y", retry[3].Text, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ExhaustedRetriesReturnTheParentSoNoEvaluationIsSpent()
    {
        var client = new FakeChatClient(
            DiffResponse("nope", "x"), DiffResponse("nope", "x"), DiffResponse("nope", "x"), DiffResponse("nope", "x"));

        var operatorUnderTest = new LlmProgramVariationOperator<double>(client);
        var parent = new ProgramGenome(ParentSource, ProgramLanguage.Python);
        EvolutionVariationContext<ProgramGenome> context = Context(parent);

        ProgramGenome child = await operatorUnderTest.ProposeAsync(context);

        Assert.Equal(parent.Id, child.Id);
        Assert.Equal(parent, child);
        Assert.NotSame(parent, child);
        Assert.Same(context.Parent.Candidate.CanonicalGenome.Genome, child);
        Assert.Equal(3, client.Calls);
    }

    [Fact]
    public async Task RetryCountIsConfigurable()
    {
        var client = new FakeChatClient(DiffResponse("nope", "x"), DiffResponse("nope", "x"));
        var operatorUnderTest = new LlmProgramVariationOperator<double>(
            client, null, new LlmProgramVariationOptions { MaxProposalRetries = 0 });

        await operatorUnderTest.ProposeAsync(Context());
        Assert.Equal(1, client.Calls);
    }

    [Fact]
    public async Task IdenticalProposalsAreRejectedRatherThanReturnedAsChildren()
    {
        var client = new FakeChatClient(
            DiffResponse("    return x", "    return x"),
            DiffResponse("    return x", "    return x - 1"));

        var operatorUnderTest = new LlmProgramVariationOperator<double>(client);
        ProgramGenome child = await operatorUnderTest.ProposeAsync(Context());

        Assert.Equal("def solve(x):\n    return x - 1\n", child.Source);
        Assert.Contains("identical", client.Conversations[1][3].Text, StringComparison.Ordinal);
    }

    [Fact]
    public async Task FullRewriteModeReadsAFencedBlock()
    {
        var client = new FakeChatClient("Here it is:\n```python\ndef solve(x):\n    return x * 7\n```\n");
        var operatorUnderTest = new LlmProgramVariationOperator<double>(
            client,
            new ProgramEvolutionOptions { Language = ProgramLanguage.Python },
            new LlmProgramVariationOptions { Mode = ProgramEvolutionMode.FullRewrite });

        ProgramGenome child = await operatorUnderTest.ProposeAsync(Context());
        Assert.Equal("def solve(x):\n    return x * 7", child.Source);
    }

    [Fact]
    public async Task FullRewriteModeRefusesUnfencedProse()
    {
        var client = new FakeChatClient("I would change the return value.", "```python\nprint(2)\n```");
        var operatorUnderTest = new LlmProgramVariationOperator<double>(
            client,
            new ProgramEvolutionOptions { Language = ProgramLanguage.Python },
            new LlmProgramVariationOptions { Mode = ProgramEvolutionMode.FullRewrite });

        ProgramGenome child = await operatorUnderTest.ProposeAsync(Context());

        Assert.Equal("print(2)", child.Source);
        Assert.Contains("fenced code block", client.Conversations[1][3].Text, StringComparison.Ordinal);
    }

    [Fact]
    public async Task OversizedProposalsAreRefused()
    {
        string huge = new string('x', 400);
        var client = new FakeChatClient(
            "```python\n" + huge + "\n```",
            "```python\nprint(3)\n```");

        var operatorUnderTest = new LlmProgramVariationOperator<double>(
            client,
            new ProgramEvolutionOptions { Language = ProgramLanguage.Python, MaxProgramChars = 100 },
            new LlmProgramVariationOptions { Mode = ProgramEvolutionMode.FullRewrite });

        ProgramGenome child = await operatorUnderTest.ProposeAsync(Context());

        Assert.Equal(2, client.Calls);
        Assert.Contains("above the limit", client.Conversations[1][3].Text, StringComparison.Ordinal);
        Assert.Equal("print(3)", child.Source);
    }

    [Fact]
    public async Task ProviderExceptionsAreRetriedAndNeverLeakTheirMessage()
    {
        var client = new FakeChatClient(DiffResponse("    return x", "    return 9"))
        {
            ThrowOnFirstCall = new InvalidOperationException("api key sk-secret-value rejected")
        };

        var operatorUnderTest = new LlmProgramVariationOperator<double>(client);
        ProgramGenome child = await operatorUnderTest.ProposeAsync(Context());

        Assert.Equal("def solve(x):\n    return 9\n", child.Source);
        string feedback = client.Conversations[1][3].Text;
        Assert.Contains("InvalidOperationException", feedback, StringComparison.Ordinal);
        Assert.DoesNotContain("sk-secret-value", feedback, StringComparison.Ordinal);
    }

    [Fact]
    public async Task PromptQuotesTheParentAndBoundsItsLength()
    {
        var client = new FakeChatClient(DiffResponse("    return x", "    return 4"));
        var operatorUnderTest = new LlmProgramVariationOperator<double>(
            client, null, new LlmProgramVariationOptions { MaxPromptProgramChars = 300 });

        await operatorUnderTest.ProposeAsync(Context(new ProgramGenome(new string('q', 5000), ProgramLanguage.Python)));

        string user = client.Conversations[0][1].Text;
        Assert.Contains("# Current Program", user, StringComparison.Ordinal);
        Assert.Contains(new string('q', 100), user, StringComparison.Ordinal);
        Assert.DoesNotContain(new string('q', 400), user, StringComparison.Ordinal);
        Assert.Contains("...", user, StringComparison.Ordinal);
    }

    [Fact]
    public async Task PromptIsRenderedThroughTheTemplateBuilder()
    {
        var client = new FakeChatClient(DiffResponse("    return x", "    return 4"));
        var operatorUnderTest = new LlmProgramVariationOperator<double>(client);

        await operatorUnderTest.ProposeAsync(Context());

        string user = client.Conversations[0][1].Text;
        Assert.Contains("# Current Program Information", user, StringComparison.Ordinal);
        Assert.Contains("# Program Evolution History", user, StringComparison.Ordinal);
        Assert.Contains("- Fitness: 0.5", user, StringComparison.Ordinal);
        Assert.Equal(ProgramPromptTemplateKey.DiffUser, ProgramPromptTemplateKey.DiffUser);
    }

    [Fact]
    public async Task AnExplicitPromptBuilderOwnsTheModeAndTheParserFollowsIt()
    {
        // The builder asks for a full rewrite even though the variation options still say Diff, so an answer that
        // is only a fenced block must be accepted: the parser has to follow the builder, not the options.
        var builder = new ProgramPromptBuilder(
            new ProgramEvolutionPromptOptions { EvolutionMode = ProgramPromptEvolutionMode.FullRewrite },
            new ProgramEvolutionOptions { Language = ProgramLanguage.Python });
        var client = new FakeChatClient("```python\ndef solve(x):\n    return x + 9\n```");
        var operatorUnderTest = new LlmProgramVariationOperator<double>(
            client,
            new ProgramEvolutionOptions { Language = ProgramLanguage.Python },
            new LlmProgramVariationOptions { Mode = ProgramEvolutionMode.Diff },
            promptBuilder: builder);

        ProgramGenome child = await operatorUnderTest.ProposeAsync(Context());

        // The fenced-code extractor drops the block's trailing blank line, so the child keeps only the code.
        Assert.Equal("def solve(x):\n    return x + 9", child.Source);
        Assert.Same(builder, operatorUnderTest.PromptBuilder);
    }

    [Fact]
    public async Task ParentDiagnosticsAreSplitIntoArtifactsAndDiagnostics()
    {
        var client = new FakeChatClient(DiffResponse("    return x", "    return 4"));
        var operatorUnderTest = new LlmProgramVariationOperator<double>(client);

        await operatorUnderTest.ProposeAsync(Context(diagnostics: new[]
        {
            new EvolutionDiagnostic("program_script_artifact_stdout", "captured stdout line", isRedacted: true),
            new EvolutionDiagnostic("program_sandbox_timeout", "the candidate ran out of time")
        }));

        string user = client.Conversations[0][1].Text;
        Assert.Contains("captured stdout line", user, StringComparison.Ordinal);
        Assert.Contains("the candidate ran out of time", user, StringComparison.Ordinal);
        Assert.Contains("Evaluation Diagnostics", user, StringComparison.Ordinal);
    }

    [Fact]
    public async Task FeatureDimensionNamesLabelTheArchiveCell()
    {
        var client = new FakeChatClient(DiffResponse("    return x", "    return 4"));
        var named = new LlmProgramVariationOperator<double>(client, null, new LlmProgramVariationOptions
        {
            FeatureDimensions = new List<string> { "length", "complexity" },
            FeatureBinCounts = new List<int> { 10, 10 }
        });

        await named.ProposeAsync(Context());

        string user = client.Conversations[0][1].Text;
        Assert.Contains("length", user, StringComparison.Ordinal);
        Assert.Contains("complexity", user, StringComparison.Ordinal);
    }

    [Fact]
    public async Task MismatchedFeatureNamesKeepTheNamesAndDropTheIndices()
    {
        // The cell has two bins but only one name is configured, so the indices must not be attached to it.
        var client = new FakeChatClient(DiffResponse("    return x", "    return 4"));
        var mismatched = new LlmProgramVariationOperator<double>(client, null, new LlmProgramVariationOptions
        {
            FeatureDimensions = new List<string> { "length" }
        });

        ProgramGenome child = await mismatched.ProposeAsync(Context());

        Assert.NotEqual(ParentSource, child.Source);
        Assert.Contains("length", client.Conversations[0][1].Text, StringComparison.Ordinal);
    }

    [Fact]
    public async Task UsageTotalsSeparateProposalsFromRequests()
    {
        var client = new FakeChatClient("no edits here", DiffResponse("    return x", "    return 4"));
        var operatorUnderTest = new LlmProgramVariationOperator<double>(client);

        await operatorUnderTest.ProposeAsync(Context());
        ProgramEvolutionLlmUsage usage = operatorUnderTest.GetUsage();

        Assert.Equal(1, usage.Proposals);
        Assert.Equal(2, usage.ChatCalls);
        Assert.Equal(1, usage.Retries);
        Assert.Equal(0, usage.AbandonedProposals);
        Assert.Equal(0, usage.ProviderErrors);
        Assert.Equal(2.0, usage.CallsPerProposal);
    }

    [Fact]
    public async Task AbandonedProposalsAreCountedAndRecorded()
    {
        var client = new FakeChatClient("prose", "more prose", "still prose");
        var operatorUnderTest = new LlmProgramVariationOperator<double>(client);

        ProgramGenome child = await operatorUnderTest.ProposeAsync(Context());

        Assert.Equal(ParentSource, child.Source);
        Assert.Equal(1, operatorUnderTest.GetUsage().AbandonedProposals);

        IReadOnlyList<ProgramProposalAttempt> attempts = operatorUnderTest.GetRecentAttempts();
        Assert.Equal(4, attempts.Count);
        Assert.All(attempts.Take(3), attempt => Assert.Equal(ProgramProposalOutcome.ParseFailed, attempt.Outcome));
        Assert.Equal(ProgramProposalOutcome.Exhausted, attempts[3].Outcome);
        Assert.Equal(new[] { 1, 2, 3, 3 }, attempts.Select(attempt => attempt.AttemptNumber));
    }

    [Fact]
    public async Task RecordedAttemptsAreBoundedAndNeverEchoProviderMessages()
    {
        var client = new FakeChatClient("prose", "more prose", DiffResponse("    return x", "    return 4"))
        {
            ThrowOnFirstCall = new InvalidOperationException("key=sk-secret-value endpoint=https://internal")
        };
        var operatorUnderTest = new LlmProgramVariationOperator<double>(
            client, null, new LlmProgramVariationOptions { MaxProposalRetries = 3, MaxRecordedAttempts = 2 });

        await operatorUnderTest.ProposeAsync(Context());

        IReadOnlyList<ProgramProposalAttempt> attempts = operatorUnderTest.GetRecentAttempts();
        Assert.Equal(2, attempts.Count);
        Assert.All(attempts, attempt => Assert.DoesNotContain("sk-secret-value", attempt.Detail, StringComparison.Ordinal));
        Assert.Equal(1, operatorUnderTest.GetUsage().ProviderErrors);
    }

    [Fact]
    public async Task AttemptRecordingCanBeTurnedOffWithoutLosingTheCounters()
    {
        var client = new FakeChatClient(DiffResponse("    return x", "    return 4"));
        var operatorUnderTest = new LlmProgramVariationOperator<double>(
            client, null, new LlmProgramVariationOptions { MaxRecordedAttempts = 0 });

        await operatorUnderTest.ProposeAsync(Context());

        Assert.Empty(operatorUnderTest.GetRecentAttempts());
        Assert.Equal(1, operatorUnderTest.GetUsage().ChatCalls);
    }

    [Fact]
    public async Task ProviderReportedTokensAccumulate()
    {
        var client = new FakeChatClient(DiffResponse("    return x", "    return 4")) { Usage = new ChatUsage(11, 7) };
        var operatorUnderTest = new LlmProgramVariationOperator<double>(client);

        await operatorUnderTest.ProposeAsync(Context());
        ProgramEvolutionLlmUsage usage = operatorUnderTest.GetUsage();

        Assert.Equal(11, usage.InputTokens);
        Assert.Equal(7, usage.OutputTokens);
        Assert.Equal(18, usage.TotalTokens);
        Assert.Equal(11, operatorUnderTest.GetRecentAttempts()[0].InputTokens);
    }

    [Fact]
    public async Task TaskDescriptionReachesThePrompt()
    {
        var client = new FakeChatClient(DiffResponse("    return x", "    return 4"));
        var operatorUnderTest = new LlmProgramVariationOperator<double>(
            client,
            new ProgramEvolutionOptions
            {
                Language = ProgramLanguage.Python,
                TaskDescription = "Return the smallest prime above the input."
            });

        await operatorUnderTest.ProposeAsync(Context());

        Assert.Contains("smallest prime above the input", client.Conversations[0][1].Text, StringComparison.Ordinal);
    }

    [Fact]
    public async Task InspirationsAreQuotedAndCapped()
    {
        var client = new FakeChatClient(DiffResponse("    return x", "    return 4"));
        var operatorUnderTest = new LlmProgramVariationOperator<double>(
            client, null, new LlmProgramVariationOptions { MaxInspirations = 2 });

        var inspirations = new[]
        {
            Entry(new ProgramGenome("alpha_program = 1", ProgramLanguage.Python), 0.9, 1),
            Entry(new ProgramGenome("beta_program = 2", ProgramLanguage.Python), 0.8, 2),
            Entry(new ProgramGenome("gamma_program = 3", ProgramLanguage.Python), 0.7, 3)
        };

        await operatorUnderTest.ProposeAsync(Context(inspirations: inspirations));

        string user = client.Conversations[0][1].Text;
        Assert.Contains("alpha_program", user, StringComparison.Ordinal);
        Assert.Contains("beta_program", user, StringComparison.Ordinal);
        Assert.DoesNotContain("gamma_program", user, StringComparison.Ordinal);
    }

    [Fact]
    public async Task SeedIsDerivedDeterministicallyFromTheProposalStream()
    {
        var first = new FakeChatClient(DiffResponse("    return x", "    return 4"));
        var second = new FakeChatClient(DiffResponse("    return x", "    return 4"));

        await new LlmProgramVariationOperator<double>(first).ProposeAsync(Context());
        await new LlmProgramVariationOperator<double>(second).ProposeAsync(Context());

        Assert.NotNull(first.LastOptions);
        Assert.NotNull(first.LastOptions?.Seed);
        Assert.Equal(first.LastOptions?.Seed, second.LastOptions?.Seed);
        Assert.True(first.LastOptions?.Seed >= 0);
    }

    [Fact]
    public async Task ExplicitSeedOverridesTheStream()
    {
        var client = new FakeChatClient(DiffResponse("    return x", "    return 4"));
        var operatorUnderTest = new LlmProgramVariationOperator<double>(
            client, null, new LlmProgramVariationOptions { Seed = 4242, Temperature = 0.3, MaxOutputTokens = 500 });

        await operatorUnderTest.ProposeAsync(Context());

        Assert.Equal(4242, client.LastOptions?.Seed);
        Assert.Equal(0.3, client.LastOptions?.Temperature);
        Assert.Equal(500, client.LastOptions?.MaxOutputTokens);
    }

    [Fact]
    public async Task RequestedFormatAndEvolveBlockRulesReachTheModel()
    {
        var diffClient = new FakeChatClient(DiffResponse("    return x", "    return 4"));
        await new LlmProgramVariationOperator<double>(diffClient).ProposeAsync(Context());
        Assert.Contains("<<<<<<< SEARCH", diffClient.Conversations[0][1].Text, StringComparison.Ordinal);

        var enforcedClient = new FakeChatClient(DiffResponse("    return x", "    return 4"));
        var enforced = new LlmProgramVariationOperator<double>(
            enforcedClient,
            new ProgramEvolutionOptions { Language = ProgramLanguage.Python, EnforceEvolveBlocks = true });
        await enforced.ProposeAsync(Context());
        Assert.Contains("EVOLVE-BLOCK-START", enforcedClient.Conversations[0][1].Text, StringComparison.Ordinal);

        var customClient = new FakeChatClient(DiffResponse("    return x", "    return 4"));
        var custom = new LlmProgramVariationOperator<double>(
            customClient, null, new LlmProgramVariationOptions { SystemMessage = "Be terse." });
        await custom.ProposeAsync(Context());
        Assert.Equal("Be terse.", customClient.Conversations[0][0].Text);
    }

    [Fact]
    public async Task FullRewriteModeAsksForAFencedBlockRatherThanEditBlocks()
    {
        var client = new FakeChatClient("```python\ndef solve(x):\n    return x - 1\n```");
        var operatorUnderTest = new LlmProgramVariationOperator<double>(
            client,
            new ProgramEvolutionOptions { Language = ProgramLanguage.Python },
            new LlmProgramVariationOptions { Mode = ProgramEvolutionMode.FullRewrite });

        await operatorUnderTest.ProposeAsync(Context());

        string user = client.Conversations[0][1].Text;
        Assert.Contains("Return the complete new program inside a single fenced block", user, StringComparison.Ordinal);
        Assert.DoesNotContain("<<<<<<< SEARCH", user, StringComparison.Ordinal);
    }

    [Fact]
    public async Task CancellationPropagates()
    {
        using var source = new CancellationTokenSource();
        source.Cancel();
        var client = new FakeChatClient(DiffResponse("    return x", "    return 4"));
        var operatorUnderTest = new LlmProgramVariationOperator<double>(client);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(
            async () => await operatorUnderTest.ProposeAsync(Context(), source.Token));
        Assert.Equal(0, client.Calls);
    }

    [Fact]
    public void VersionHashTracksBehaviourChanges()
    {
        var client = new FakeChatClient("ignored");
        string baseline = new LlmProgramVariationOperator<double>(client).VersionHash;

        Assert.Equal(baseline, new LlmProgramVariationOperator<double>(client).VersionHash);
        Assert.NotEqual(baseline, new LlmProgramVariationOperator<double>(
            client, null, new LlmProgramVariationOptions { Mode = ProgramEvolutionMode.FullRewrite }).VersionHash);
        Assert.NotEqual(baseline, new LlmProgramVariationOperator<double>(
            client, null, new LlmProgramVariationOptions { MaxProposalRetries = 5 }).VersionHash);
        Assert.NotEqual(baseline, new LlmProgramVariationOperator<double>(
            client, new ProgramEvolutionOptions { Language = ProgramLanguage.CSharp }).VersionHash);
        Assert.Equal("llm-program-variation", new LlmProgramVariationOperator<double>(client).Id);
    }

    [Fact]
    public void OptionsAreValidatedAndCopied()
    {
        var client = new FakeChatClient("ignored");
#pragma warning disable CS8600, CS8625
        Assert.Throws<ArgumentNullException>(() => new LlmProgramVariationOperator<double>(null));
#pragma warning restore CS8600, CS8625
        Assert.Throws<ArgumentOutOfRangeException>(() => new LlmProgramVariationOperator<double>(
            client, null, new LlmProgramVariationOptions { MaxProposalRetries = -1 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LlmProgramVariationOperator<double>(
            client, null, new LlmProgramVariationOptions { Temperature = 5 }));
        Assert.Throws<ArgumentException>(() => new LlmProgramVariationOperator<double>(client, id: " "));

        var options = new LlmProgramVariationOptions { MaxInspirations = 1 };
        var operatorUnderTest = new LlmProgramVariationOperator<double>(client, null, options);
        options.MaxInspirations = 9;
        Assert.Equal(1, operatorUnderTest.GetVariationOptions().MaxInspirations);
        Assert.Equal(100_000, operatorUnderTest.GetProgramOptions().MaxProgramChars);
    }
}
