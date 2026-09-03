using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

/// <summary>
/// Covers the operator's checkpointed memory. The recorded attempts are the only thing the operator carries between
/// proposals, and they change the next prompt, so leaving them out of a checkpoint made a resumed run ask a
/// different question than an uninterrupted one while every other part of the engine resumed exactly.
/// </summary>
public sealed class LlmProgramVariationStateTests
{
    private const string ParentSource = "def solve(x):\n    return x\n";

    [Fact]
    public void TheOperatorAdvertisesItselfAsCheckpointable()
    {
        // The engine only captures and restores state for operators that opt in, so the interface is the whole
        // mechanism: dropping it silently reverts the behaviour this file pins.
        var operatorUnderTest = new LlmProgramVariationOperator<double>(new FakeChatClient("x"));
        Assert.IsAssignableFrom<ICheckpointableVariationOperator<ProgramGenome>>(operatorUnderTest);
    }

    [Fact]
    public async Task ARestoredOperatorTellsTheModelWhatTheRunAlreadyTried()
    {
        // Interrupted run: the first answer is unusable, so it is recorded and the retry succeeds.
        var interrupted = new LlmProgramVariationOperator<double>(
            new FakeChatClient("I would change the return value.", DiffResponse("    return x", "    return x + 1")));
        await interrupted.ProposeAsync(Context());

        ProgramProposalAttempt rejected = Assert.Single(
            interrupted.GetRecentAttempts(), attempt => !attempt.IsAccepted);
        Assert.False(string.IsNullOrWhiteSpace(rejected.Detail));

        string captured = interrupted.CaptureState();

        // Resumed run: a fresh operator, the same parent, restored memory.
        var resumedClient = new FakeChatClient(DiffResponse("    return x", "    return x + 2"));
        var resumed = new LlmProgramVariationOperator<double>(resumedClient);
        resumed.RestoreState(captured);
        await resumed.ProposeAsync(Context());

        Assert.Contains(rejected.Detail, PromptOf(resumedClient), StringComparison.Ordinal);

        // Control: without the restore the same parent produces a prompt that has forgotten the dead end.
        var freshClient = new FakeChatClient(DiffResponse("    return x", "    return x + 2"));
        await new LlmProgramVariationOperator<double>(freshClient).ProposeAsync(Context());

        Assert.DoesNotContain(rejected.Detail, PromptOf(freshClient), StringComparison.Ordinal);
    }

    [Fact]
    public async Task CapturedStateSurvivesARoundTripUnchanged()
    {
        var original = new LlmProgramVariationOperator<double>(
            new FakeChatClient("prose, not a diff", DiffResponse("    return x", "    return x + 1")));
        await original.ProposeAsync(Context());

        string captured = original.CaptureState();
        var restored = new LlmProgramVariationOperator<double>(new FakeChatClient("x"));
        restored.RestoreState(captured);

        // Field-by-field, because the engine folds this text into the run's state hash: a capture that is merely
        // similar after a round trip would make a resumed run's identity differ from the run it continues.
        Assert.Equal(captured, restored.CaptureState());
        Assert.Equal(original.GetRecentAttempts().Count, restored.GetRecentAttempts().Count);
        for (int i = 0; i < original.GetRecentAttempts().Count; i++)
        {
            ProgramProposalAttempt before = original.GetRecentAttempts()[i];
            ProgramProposalAttempt after = restored.GetRecentAttempts()[i];
            Assert.Equal(before.ParentGenomeId, after.ParentGenomeId);
            Assert.Equal(before.AttemptNumber, after.AttemptNumber);
            Assert.Equal(before.Outcome, after.Outcome);
            Assert.Equal(before.Detail, after.Detail);
            Assert.Equal(before.InputTokens, after.InputTokens);
            Assert.Equal(before.OutputTokens, after.OutputTokens);
        }
    }

    [Fact]
    public void AnOperatorThatHasProposedNothingStillCapturesSomethingRestorable()
    {
        // "Nothing recorded yet" has to be distinguishable from "not captured", or a checkpoint taken before the
        // first proposal cannot be told apart from a missing one.
        var operatorUnderTest = new LlmProgramVariationOperator<double>(new FakeChatClient("x"));
        string captured = operatorUnderTest.CaptureState();

        Assert.False(string.IsNullOrEmpty(captured));

        var restored = new LlmProgramVariationOperator<double>(new FakeChatClient("x"));
        restored.RestoreState(captured);
        Assert.Empty(restored.GetRecentAttempts());
    }

    [Theory]
    [InlineData("not json at all")]
    [InlineData("{\"SchemaVersion\":0,\"Attempts\":[]}")]
    [InlineData("{\"SchemaVersion\":99,\"Attempts\":[]}")]
    [InlineData("{\"SchemaVersion\":1,\"Attempts\":[{\"ParentGenomeId\":\"\",\"AttemptNumber\":1,\"Outcome\":0}]}")]
    [InlineData("{\"SchemaVersion\":1,\"Attempts\":[{\"ParentGenomeId\":\"p\",\"AttemptNumber\":0,\"Outcome\":0}]}")]
    [InlineData("{\"SchemaVersion\":1,\"Attempts\":[{\"ParentGenomeId\":\"p\",\"AttemptNumber\":1,\"Outcome\":9999}]}")]
    [InlineData("{\"SchemaVersion\":1,\"Attempts\":[{\"ParentGenomeId\":\"p\",\"AttemptNumber\":1,\"Outcome\":0,\"InputTokens\":-1}]}")]
    public void AMalformedPayloadIsRefusedRatherThanPartlyApplied(string payload)
    {
        var operatorUnderTest = new LlmProgramVariationOperator<double>(new FakeChatClient("x"));
        Assert.Throws<InvalidDataException>(() => operatorUnderTest.RestoreState(payload));
    }

    [Fact]
    public void APayloadHoldingMoreAttemptsThanConfiguredIsRefused()
    {
        // The window is a memory bound. Honouring a checkpoint that exceeds it would let a crafted or corrupted
        // file reintroduce the unbounded growth the bound exists to prevent.
        var operatorUnderTest = new LlmProgramVariationOperator<double>(
            new FakeChatClient("x"), null, new LlmProgramVariationOptions { MaxRecordedAttempts = 1 });

        string payload = "{\"SchemaVersion\":1,\"Attempts\":[" +
            "{\"ParentGenomeId\":\"p\",\"AttemptNumber\":1,\"Outcome\":0,\"Detail\":\"a\"}," +
            "{\"ParentGenomeId\":\"p\",\"AttemptNumber\":2,\"Outcome\":0,\"Detail\":\"b\"}]}";

        Assert.Throws<InvalidDataException>(() => operatorUnderTest.RestoreState(payload));
    }

    [Fact]
    public async Task RestoringReplacesWhateverTheOperatorHadRatherThanAppending()
    {
        var operatorUnderTest = new LlmProgramVariationOperator<double>(
            new FakeChatClient("prose", DiffResponse("    return x", "    return x + 1")));
        await operatorUnderTest.ProposeAsync(Context());
        Assert.NotEmpty(operatorUnderTest.GetRecentAttempts());

        operatorUnderTest.RestoreState("{\"SchemaVersion\":1,\"Attempts\":[]}");
        Assert.Empty(operatorUnderTest.GetRecentAttempts());
    }

    private static string PromptOf(FakeChatClient client) =>
        string.Join("\n", client.Conversations[0].Select(message => message.Text));

    private static string DiffResponse(string search, string replace) =>
        "<<<<<<< SEARCH\n" + search + "\n=======\n" + replace + "\n>>>>>>> REPLACE\n";

    private static EvolutionVariationContext<ProgramGenome> Context()
    {
        var genome = new ProgramGenome(ParentSource, ProgramLanguage.Python);
        var lineage = new EvolutionLineage(null, null, "seed", null, 0, 0, 0UL);
        var candidate = new EvolutionCandidate<ProgramGenome>(
            0, new EvolutionCanonicalGenome<ProgramGenome>(genome, genome.Id), lineage);
        var evaluation = new EvolutionEvaluation(
            0,
            genome.Id,
            EvolutionEvaluationStatus.Completed,
            0.5,
            EvolutionOptimizationDirection.Maximize,
            new Dictionary<string, double>(StringComparer.Ordinal) { ["x"] = 0.5 },
            Array.Empty<double>(),
            Array.Empty<double>(),
            new EvolutionEvaluationCost(TimeSpan.Zero, 1, 1),
            lineage,
            EvolutionCacheStatus.Miss,
            Array.Empty<EvolutionDiagnostic>(),
            "task-v1",
            "evaluator-v1",
            "config-v1");

        return new EvolutionVariationContext<ProgramGenome>(
            new EvolutionArchiveEntry<ProgramGenome>(new EvolutionCellKey(new[] { 1, 2 }), candidate, evaluation),
            Array.Empty<EvolutionArchiveEntry<ProgramGenome>>(),
            new StableRandom(1234UL, 7UL),
            0,
            0);
    }
}
