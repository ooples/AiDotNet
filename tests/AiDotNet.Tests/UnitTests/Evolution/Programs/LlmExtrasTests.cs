using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Models.Connectors;
using AiDotNet.Agentic.Pipeline;
using AiDotNet.Configuration;
using Newtonsoft.Json;
using Xunit;

using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

/// <summary>
/// Covers the language-model capabilities an evolution run needs beyond a single HTTP call: a person answering
/// prompts by hand, a model reached by running a command with a spending cap, and drawing several answers from one
/// prompt before rewriting it.
/// </summary>
public sealed class LlmExtrasTests
{
    [Fact]
    public async Task TheManualClientWritesTheQuestionAndWaitsForAnAnswerFile()
    {
        using var directory = new TemporaryDirectory();
        var client = new ManualChatClient<double>(directory.Path,
            new ManualChatClientOptions { PollInterval = TimeSpan.FromMilliseconds(20) });

        Task<ChatResponse> pending = client.GetResponseAsync(
            new[] { new ChatMessage(ChatRole.User, "Improve this program.") });

        string task = await WaitForFileAsync(directory.Path, ManualChatClient<double>.TaskExtension);
        string document = File.ReadAllText(task);
        Assert.Contains("Improve this program.", document, StringComparison.Ordinal);
        Assert.Contains("User", document, StringComparison.Ordinal);

        File.WriteAllText(
            AnswerPathFor(task),
            JsonConvert.SerializeObject(new { text = "here is the improved program" }));

        ChatResponse response = await pending;
        Assert.Equal("here is the improved program", response.Text);
        Assert.Equal("manual", response.ModelId);
        Assert.Equal(1, client.Requests);

        // Both files are removed once the answer is read, so the directory shows only what is still outstanding.
        Assert.Empty(Directory.GetFiles(directory.Path));
    }

    [Fact]
    public async Task AnAnswerWrittenAsPlainTextIsAcceptedToo()
    {
        using var directory = new TemporaryDirectory();
        var client = new ManualChatClient<double>(directory.Path,
            new ManualChatClientOptions { PollInterval = TimeSpan.FromMilliseconds(20) });

        Task<ChatResponse> pending = client.GetResponseAsync(new[] { new ChatMessage(ChatRole.User, "hello") });
        string task = await WaitForFileAsync(directory.Path, ManualChatClient<double>.TaskExtension);
        File.WriteAllText(
            AnswerPathFor(task),
            "just some text\n");

        ChatResponse response = await pending;
        Assert.Equal("just some text\n", response.Text);
    }

    [Fact]
    public async Task WaitingIsBoundedWhenATimeoutIsSetAndCancellableWhenItIsNot()
    {
        using var directory = new TemporaryDirectory();
        var impatient = new ManualChatClient<double>(directory.Path, new ManualChatClientOptions
        {
            PollInterval = TimeSpan.FromMilliseconds(20),
            Timeout = TimeSpan.FromMilliseconds(120)
        });

        await Assert.ThrowsAsync<TimeoutException>(() =>
            impatient.GetResponseAsync(new[] { new ChatMessage(ChatRole.User, "hello") }));

        var patient = new ManualChatClient<double>(directory.Path,
            new ManualChatClientOptions { PollInterval = TimeSpan.FromMilliseconds(20) });
        using var cancellation = new CancellationTokenSource(TimeSpan.FromMilliseconds(120));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            patient.GetResponseAsync(new[] { new ChatMessage(ChatRole.User, "hello") }, null, cancellation.Token));
    }

    [Fact]
    public void StaleAnswersFromAnEarlierRunAreClearedRatherThanServed()
    {
        using var directory = new TemporaryDirectory();
        File.WriteAllText(Path.Combine(directory.Path, "000001-old" + ManualChatClient<double>.AnswerExtension), "old");
        File.WriteAllText(Path.Combine(directory.Path, "000001-old" + ManualChatClient<double>.TaskExtension), "old");
        File.WriteAllText(Path.Combine(directory.Path, "notes.txt"), "keep me");

        _ = new ManualChatClient<double>(directory.Path);

        // An answer left behind would be served instantly as the reply to a prompt it never saw. Unrelated files are
        // left alone, because the queue directory is the caller's and may hold other things.
        Assert.Equal(new[] { "notes.txt" },
            Directory.GetFiles(directory.Path).Select(Path.GetFileName).ToArray());
    }

    [Fact]
    public void TheManualClientRefusesSettingsItCannotHonour()
    {
        using var directory = new TemporaryDirectory();
        Assert.Throws<ArgumentException>(() => new ManualChatClient<double>("   "));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ManualChatClient<double>(directory.Path,
            new ManualChatClientOptions { PollInterval = TimeSpan.Zero }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ManualChatClient<double>(directory.Path,
            new ManualChatClientOptions { Timeout = TimeSpan.FromSeconds(-1) }));
        Assert.Throws<ArgumentException>(() => new ManualChatClient<double>(directory.Path,
            new ManualChatClientOptions { ModelId = " " }));
    }

    [Fact]
    public void TheSpendingCapReachesTheCommandAndTheCommandLineIsBounded()
    {
        var capped = new ProcessChatClientOptions
        {
            FileName = "my-model",
            Arguments = { "--json" },
            MaxBudgetUsd = 0.25
        };

        Assert.Equal(new[] { "--json", "--max-budget-usd", "0.25" }, capped.BuildArguments().ToArray());

        // Formatted with the invariant culture, so a machine with a comma decimal separator sends the same text.
        var fractional = new ProcessChatClientOptions { FileName = "my-model", MaxBudgetUsd = 1.5 };
        Assert.Equal("1.5", fractional.BuildArguments()[1]);

        var uncapped = new ProcessChatClientOptions { FileName = "my-model", Arguments = { "--json" } };
        Assert.Equal(new[] { "--json" }, uncapped.BuildArguments().ToArray());

        var overlong = new ProcessChatClientOptions { FileName = "my-model" };
        overlong.Arguments.Add(new string('x', ProcessChatClientOptions.MaxArgumentLength + 1));
        ArgumentException failure = Assert.Throws<ArgumentException>(() => overlong.BuildArguments());
        Assert.Contains("sent on standard input", failure.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheProcessClientRefusesSettingsThatCannotProduceARunnableCommand()
    {
        Assert.Throws<ArgumentNullException>(() => new ProcessChatClient<double>(null!));
        Assert.Throws<ArgumentException>(() =>
            new ProcessChatClient<double>(new ProcessChatClientOptions { FileName = "  " }));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ProcessChatClient<double>(new ProcessChatClientOptions { FileName = "m", MaxBudgetUsd = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ProcessChatClient<double>(new ProcessChatClientOptions { FileName = "m", Timeout = TimeSpan.Zero }));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ProcessChatClient<double>(new ProcessChatClientOptions { FileName = "m", MaxOutputChars = 0 }));

        var client = new ProcessChatClient<double>(new ProcessChatClientOptions
        {
            FileName = "my-model",
            MaxBudgetUsd = 2,
            ModelId = "local-agent"
        });
        Assert.Equal("local-agent", client.ModelId);
        Assert.Equal(0, client.Calls);
        Assert.Equal(new[] { "--max-budget-usd", "2" }, client.Arguments.ToArray());
    }

    [Fact]
    public async Task AMissingCommandFailsWithAMessageRatherThanACrash()
    {
        var client = new ProcessChatClient<double>(new ProcessChatClientOptions
        {
            FileName = "aidotnet-no-such-command-" + Guid.NewGuid().ToString("N"),
            Timeout = TimeSpan.FromSeconds(5)
        });

        InvalidOperationException failure = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            client.GetResponseAsync(new[] { new ChatMessage(ChatRole.User, "hello") }));
        Assert.Contains("could not be started", failure.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EveryJudgeOfAPanelCanBeAskedAndTheSettingIsCarriedThroughACopy()
    {
        var options = new LlmFeedbackOptions { JudgeWithEveryEnsembleMember = true };
        Assert.True(options.Clone().JudgeWithEveryEnsembleMember);
        Assert.False(new LlmFeedbackOptions().JudgeWithEveryEnsembleMember);

        // The ensemble exposes its members and their weights, which is what weighted averaging needs.
        var panel = new WeightedEnsembleChatClient<double>(new[]
        {
            new ChatClientEnsembleMember<double>(new FakeChatClient("{}"), 0.75, name: "first"),
            new ChatClientEnsembleMember<double>(new FakeChatClient("{}"), 0.25, name: "second")
        });
        Assert.Equal(2, panel.Members.Count);
        Assert.Equal(0.75, panel.Members[0].Weight);
    }

    [Fact]
    public void DrawingSeveralAnswersFromOnePromptIsConfiguredAndBounded()
    {
        var options = new LlmProgramVariationOptions { SamplesPerAttempt = 3 };
        options.Validate();
        Assert.Equal(3, options.Clone().SamplesPerAttempt);
        Assert.Equal(1, new LlmProgramVariationOptions().SamplesPerAttempt);

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LlmProgramVariationOptions { SamplesPerAttempt = 0 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LlmProgramVariationOptions { SamplesPerAttempt = 17 }.Validate());
    }

    [Fact]
    public async Task EveryPanelMemberScoresAndEachCriterionIsTheirWeightedMean()
    {
        var panel = new WeightedEnsembleChatClient<double>(new[]
        {
            new ChatClientEnsembleMember<double>(new FakeChatClient(JudgeAnswer(1.0)), 0.75, name: "generous"),
            new ChatClientEnsembleMember<double>(new FakeChatClient(JudgeAnswer(0.2)), 0.25, name: "harsh")
        });

        var judge = new AiDotNet.Evolution.Programs.LlmJudgeProgramFitnessEvaluator<double>(
            panel, Measured(0.5), null, new LlmFeedbackOptions { JudgeWithEveryEnsembleMember = true });

        AiDotNet.Evolution.EvolutionTaskResult result = await judge.EvaluateAsync(Candidate(), JudgeContext());

        // 0.75 * 1.0 + 0.25 * 0.2, normalised by the weights that answered.
        Assert.Equal(0.8, result.Descriptors["llm_average"], 10);
        Assert.Equal(0.8, result.Descriptors["llm_correctness"], 10);
    }

    [Fact]
    public async Task AMemberThatCannotAnswerIsLeftOutRatherThanCountedAsZero()
    {
        var panel = new WeightedEnsembleChatClient<double>(new[]
        {
            new ChatClientEnsembleMember<double>(new FakeChatClient(JudgeAnswer(0.6)), 0.5, name: "answering"),
            new ChatClientEnsembleMember<double>(new FakeChatClient("not json at all"), 0.5, name: "confused")
        });

        var judge = new AiDotNet.Evolution.Programs.LlmJudgeProgramFitnessEvaluator<double>(
            panel, Measured(0.5), null, new LlmFeedbackOptions { JudgeWithEveryEnsembleMember = true });

        AiDotNet.Evolution.EvolutionTaskResult result = await judge.EvaluateAsync(Candidate(), JudgeContext());

        // Counting the unusable answer as zero would halve the score; leaving it out costs precision instead.
        Assert.Equal(0.6, result.Descriptors["llm_average"], 10);
    }

    [Fact]
    public async Task APanelWhereNobodyAnsweredIsReportedRatherThanScoredZero()
    {
        var panel = new WeightedEnsembleChatClient<double>(new[]
        {
            new ChatClientEnsembleMember<double>(new FakeChatClient("prose"), 0.5, name: "first"),
            new ChatClientEnsembleMember<double>(new FakeChatClient("more prose"), 0.5, name: "second")
        });

        var judge = new AiDotNet.Evolution.Programs.LlmJudgeProgramFitnessEvaluator<double>(
            panel, Measured(0.5), null, new LlmFeedbackOptions { JudgeWithEveryEnsembleMember = true });

        AiDotNet.Evolution.EvolutionTaskResult result = await judge.EvaluateAsync(Candidate(), JudgeContext());

        Assert.Equal(0.5, result.Quality.GetValueOrDefault(), 10);
        Assert.DoesNotContain("llm_average", result.Descriptors.Keys);
        Assert.Contains(result.Diagnostics, diagnostic => diagnostic.Code == "llm_judge_unusable");
    }

    [Fact]
    public async Task AnUnusableAnswerIsRedrawnFromTheSamePromptBeforeTheModelIsGivenFeedback()
    {
        // One attempt, two samples: the second draw uses the identical conversation, with no feedback added.
        var client = new FakeChatClient(
            Diff("    return y", "    return 0"),
            Diff("    return x", "    return x + 1"));

        var operatorUnderTest = new AiDotNet.Evolution.Programs.LlmProgramVariationOperator<double>(
            client, null, new LlmProgramVariationOptions { SamplesPerAttempt = 2, MaxProposalRetries = 0 });

        AiDotNet.Evolution.Programs.ProgramGenome child =
            await operatorUnderTest.ProposeAsync(VariationContext());

        Assert.Equal("def solve(x):\n    return x + 1\n", child.Source);
        Assert.Equal(2, client.Calls);
        Assert.Equal(client.Conversations[0].Count, client.Conversations[1].Count);
        Assert.Equal(client.Conversations[0][^1].Text, client.Conversations[1][^1].Text);
    }

    [Fact]
    public async Task FeedbackStillArrivesOnceEverySampleOfAnAttemptHasFailed()
    {
        var client = new FakeChatClient(
            Diff("    return y", "    return 0"),
            Diff("    return z", "    return 0"),
            Diff("    return x", "    return x + 1"));

        var operatorUnderTest = new AiDotNet.Evolution.Programs.LlmProgramVariationOperator<double>(
            client, null, new LlmProgramVariationOptions { SamplesPerAttempt = 2, MaxProposalRetries = 1 });

        AiDotNet.Evolution.Programs.ProgramGenome child =
            await operatorUnderTest.ProposeAsync(VariationContext());

        Assert.Equal("def solve(x):\n    return x + 1\n", child.Source);
        Assert.Equal(3, client.Calls);
        Assert.True(client.Conversations[2].Count > client.Conversations[1].Count);
        Assert.Contains("could not be applied", client.Conversations[2][^1].Text, StringComparison.Ordinal);
    }

    private static AiDotNet.Evolution.EvolutionVariationContext<AiDotNet.Evolution.Programs.ProgramGenome>
        VariationContext()
    {
        var genome = new AiDotNet.Evolution.Programs.ProgramGenome(
            "def solve(x):\n    return x\n", AiDotNet.ProgramSynthesis.Enums.ProgramLanguage.Python);
        var lineage = new AiDotNet.Evolution.EvolutionLineage(null, null, "seed", null, 0, 0, 0UL);
        var candidate = new AiDotNet.Evolution.EvolutionCandidate<AiDotNet.Evolution.Programs.ProgramGenome>(
            0, new AiDotNet.Evolution.EvolutionCanonicalGenome<AiDotNet.Evolution.Programs.ProgramGenome>(genome, genome.Id),
            lineage);
        var evaluation = new AiDotNet.Evolution.EvolutionEvaluation(
            0, genome.Id, AiDotNet.Enums.EvolutionEvaluationStatus.Completed, 0.5,
            AiDotNet.Enums.EvolutionOptimizationDirection.Maximize,
            new Dictionary<string, double>(StringComparer.Ordinal) { ["x"] = 0.5 },
            Array.Empty<double>(), Array.Empty<double>(),
            new AiDotNet.Evolution.EvolutionEvaluationCost(TimeSpan.Zero, 1, 1), lineage,
            AiDotNet.Enums.EvolutionCacheStatus.Miss, Array.Empty<AiDotNet.Evolution.EvolutionDiagnostic>(),
            "task-v1", "evaluator-v1", "config-v1");
        var entry = new AiDotNet.Evolution.EvolutionArchiveEntry<AiDotNet.Evolution.Programs.ProgramGenome>(
            new AiDotNet.Evolution.EvolutionCellKey(new[] { 1 }), candidate, evaluation);
        return new AiDotNet.Evolution.EvolutionVariationContext<AiDotNet.Evolution.Programs.ProgramGenome>(
            entry, Array.Empty<AiDotNet.Evolution.EvolutionArchiveEntry<AiDotNet.Evolution.Programs.ProgramGenome>>(),
            new AiDotNet.Evolution.StableRandom(1234UL, 7UL), 0, 0);
    }

    private static string Diff(string search, string replace) =>
        "<<<<<<< SEARCH\n" + search + "\n=======\n" + replace + "\n>>>>>>> REPLACE\n";

    private static AiDotNet.Evolution.EvolutionEvaluationContext JudgeContext() => new(1, 99UL, 7UL, 1);

    private static AiDotNet.Evolution.Programs.ProgramGenome Candidate() =>
        new("def solve(x):\n    return x * 2\n", AiDotNet.ProgramSynthesis.Enums.ProgramLanguage.Python);

    private static AiDotNet.Interfaces.IProgramFitnessEvaluator Measured(double quality) =>
        new AiDotNet.Evolution.Programs.DelegateProgramFitnessEvaluator((_, _, _) =>
            new ValueTask<AiDotNet.Evolution.EvolutionTaskResult>(
                new AiDotNet.Evolution.EvolutionTaskResult(
                    AiDotNet.Enums.EvolutionEvaluationStatus.Completed,
                    quality,
                    AiDotNet.Enums.EvolutionOptimizationDirection.Maximize,
                    new Dictionary<string, double>(StringComparer.Ordinal) { ["passRate"] = quality },
                    costUnits: 3)));

    /// <summary>A judge answer where every criterion carries the same score, so a weighted mean is easy to read.</summary>
    private static string JudgeAnswer(double score)
    {
        string value = score.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
        return "{\"correctness\": " + value + ", \"efficiency\": " + value + ", \"readability\": " + value +
               ", \"reasoning\": \"looks fine\"}";
    }

    /// <summary>Names the answer file that goes with a task file.</summary>
    /// <remarks>Written without the culture-aware Replace overload, which the oldest target framework lacks.</remarks>
    private static string AnswerPathFor(string taskPath) =>
        taskPath.Substring(0, taskPath.Length - ManualChatClient<double>.TaskExtension.Length) +
        ManualChatClient<double>.AnswerExtension;

    private static async Task<string> WaitForFileAsync(string directory, string extension)
    {
        for (int attempt = 0; attempt < 200; attempt++)
        {
            string[] files = Directory.GetFiles(directory, "*" + extension);
            if (files.Length > 0) return files[0];
            await Task.Delay(20);
        }

        throw new TimeoutException("No file ending '" + extension + "' appeared in '" + directory + "'.");
    }
}
