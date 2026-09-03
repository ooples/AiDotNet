using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

/// <summary>
/// Covers changes-description mode, where a reply edits two documents at once. A SEARCH/REPLACE block says what text
/// to find and never where, so the block has to be routed before anything is applied; applying a description edit to
/// the program is a change nobody asked for that still reports success.
/// </summary>
public sealed class ProgramChangesDescriptionRoutingTests
{
    private const string Source = "def solve(x):\n    return x + 1\n";
    private const string Description = "Adds one to the input.\nStill needs a faster path.\n";

    [Fact]
    public void EachBlockGoesToTheDocumentItsSearchTextOccursIn()
    {
        ProgramDiffTargetSplit split = ProgramDiff.SplitByTarget(
            new[]
            {
                Block("    return x + 1", "    return x + 2", ordinal: 0),
                Block("Adds one to the input.", "Adds two to the input.", ordinal: 1)
            },
            Source,
            Description);

        Assert.True(split.IsSuccess);
        Assert.Single(split.ProgramBlocks);
        Assert.Single(split.DescriptionBlocks);
        Assert.Equal(0, split.ProgramBlocks[0].Ordinal);
        Assert.Equal(1, split.DescriptionBlocks[0].Ordinal);
    }

    [Fact]
    public void ABlockThatCouldMeanEitherDocumentIsRefusedRatherThanGuessed()
    {
        // A description that quotes the line it changed is the ordinary way this happens: the same line is now in
        // both documents, so an edit naming that line could mean either.
        ProgramDiffTargetSplit split = ProgramDiff.SplitByTarget(
            new[] { Block("    return x + 1", "    return x + 2") },
            Source,
            "Changed the body to:\n    return x + 1\n");

        Assert.False(split.IsSuccess);
        Assert.Empty(split.ProgramBlocks);
        Assert.Empty(split.DescriptionBlocks);
        ProgramDiffFailure failure = Assert.Single(split.Failures);
        Assert.Equal(ProgramDiffFailureReason.AmbiguousTarget, failure.Reason);
        Assert.Contains("only one of them", failure.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ABlockThatMatchesNeitherDocumentIsLeftForTheProgramToReportPrecisely()
    {
        // Routing does not invent a second not-found message; Apply already reports one with the block number and an
        // excerpt, against the document the model was most likely editing.
        ProgramDiffTargetSplit split = ProgramDiff.SplitByTarget(
            new[] { Block("nothing like this exists", "replacement") }, Source, Description);

        Assert.True(split.IsSuccess);
        Assert.Single(split.ProgramBlocks);

        ProgramDiffApplyResult applied = ProgramDiff.Apply(Source, split.ProgramBlocks);
        Assert.False(applied.IsSuccess);
        Assert.Equal(ProgramDiffFailureReason.SearchTextNotFound, applied.Failures[0].Reason);
    }

    [Fact]
    public void RoutingRefusesArgumentsItCannotWorkWith()
    {
        Assert.Throws<ArgumentNullException>(() => ProgramDiff.SplitByTarget(null!, Source, Description));
        Assert.Throws<ArgumentNullException>(() => ProgramDiff.SplitByTarget(Array.Empty<ProgramDiffBlock>(), null!, Description));
        Assert.Throws<ArgumentNullException>(() => ProgramDiff.SplitByTarget(Array.Empty<ProgramDiffBlock>(), Source, null!));
        Assert.Throws<ArgumentNullException>(() =>
            ProgramDiff.SplitByTarget(new ProgramDiffBlock[] { null! }, Source, Description));

        ProgramDiffTargetSplit empty = ProgramDiff.SplitByTarget(Array.Empty<ProgramDiffBlock>(), Source, Description);
        Assert.True(empty.IsSuccess);
        Assert.Empty(empty.ProgramBlocks);
        Assert.Empty(empty.DescriptionBlocks);
    }

    [Fact]
    public void WhitespaceToleranceAppliesToBothDocumentsOrToNeither()
    {
        var lenient = new ProgramEvolutionOptions();
        lenient.Diff.FuzzyWhitespace = true;
        var strict = new ProgramEvolutionOptions();
        strict.Diff.FuzzyWhitespace = false;

        ProgramDiffBlock[] blocks = { Block("Adds   one to the input.", "Adds two.") };

        Assert.Single(ProgramDiff.SplitByTarget(blocks, Source, Description, lenient).DescriptionBlocks);
        Assert.Single(ProgramDiff.SplitByTarget(blocks, Source, Description, strict).ProgramBlocks);
    }

    [Fact]
    public void TheStartingDescriptionIsCarriedAndCopiedLikeEveryOtherSetting()
    {
        var options = new ProgramEvolutionPromptOptions
        {
            ProgramsAsChangesDescription = true,
            InitialChangesDescription = "Initial version."
        };

        ProgramEvolutionPromptOptions copy = options.Clone();

        Assert.True(copy.ProgramsAsChangesDescription);
        Assert.Equal("Initial version.", copy.InitialChangesDescription);
        Assert.Null(new ProgramEvolutionPromptOptions().InitialChangesDescription);
    }

    [Fact]
    public void EvolveBlockEnforcementIsDroppedForProseAndKeptForCode()
    {
        var enforcing = new ProgramEvolutionOptions { EnforceEvolveBlocks = true };
        ProgramEvolutionOptions relaxed = enforcing.WithoutEvolveBlockEnforcement();

        Assert.True(enforcing.EnforceEvolveBlocks);
        Assert.False(relaxed.EnforceEvolveBlocks);
        Assert.NotSame(enforcing, relaxed);

        // Nothing is copied when there is nothing to relax, so the common path allocates nothing.
        var permissive = new ProgramEvolutionOptions();
        Assert.Same(permissive, permissive.WithoutEvolveBlockEnforcement());
    }

    [Fact]
    public async Task AReplyThatEditsBothDocumentsProducesAChildCarryingTheNewDescription()
    {
        var client = new FakeChatClient(
            Diff("    return x", "    return x * 2") + Diff("Returns the input.", "Doubles the input."));

        ProgramGenome child = await Operator(client).ProposeAsync(Context());

        Assert.Equal("def solve(x):\n    return x * 2\n", child.Source);
        Assert.Equal("Doubles the input.\n", child.Description);
        Assert.Equal(1, client.Calls);
    }

    [Fact]
    public async Task AReplyThatLeavesTheDescriptionAloneIsRejectedAndRetriedWithTheReason()
    {
        var client = new FakeChatClient(
            Diff("    return x", "    return x * 2"),
            Diff("    return x", "    return x + 1") + Diff("Returns the input.", "Adds one to the input."));

        ProgramGenome child = await Operator(client).ProposeAsync(Context());

        Assert.Equal(2, client.Calls);
        Assert.Equal("def solve(x):\n    return x + 1\n", child.Source);
        Assert.Equal("Adds one to the input.\n", child.Description);

        IReadOnlyList<AiDotNet.Agentic.Models.ChatMessage> retry = client.Conversations[1];
        Assert.Contains("changes description was not updated", retry[retry.Count - 1].Text, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AnAmbiguousEditIsRefusedAndTheModelIsToldWhy()
    {
        // The parent's description quotes the line the edit names, so the edit could mean either document.
        var quoting = new ProgramGenome("def solve(x):\n    return x\n", ProgramLanguage.Python,
            "Body is currently:\n    return x\n");
        var client = new FakeChatClient(
            Diff("    return x", "    return x * 2"),
            Diff("def solve(x):", "def solve(value):") + Diff("Body is currently:", "Body now doubles:"));

        ProgramGenome child = await Operator(client).ProposeAsync(Context(quoting));

        Assert.Equal(2, client.Calls);
        Assert.StartsWith("def solve(value):", child.Source, StringComparison.Ordinal);

        IReadOnlyList<AiDotNet.Agentic.Models.ChatMessage> retry = client.Conversations[1];
        Assert.Contains("only one of them", retry[retry.Count - 1].Text, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ThePromptShowsTheSameDescriptionTheEditsAreAppliedTo()
    {
        // The prompt used to render one description while the edits were routed against another, so from the second
        // generation on every description edit missed and the mode failed for a reason nothing reported. It also
        // substituted the description for the program, leaving the model editing source it was never shown.
        var parent = new ProgramGenome("def solve(x):\n    return x\n", ProgramLanguage.Python,
            "Second generation: returns the input unchanged.\n");
        var client = new FakeChatClient(
            Diff("    return x", "    return x * 2") +
            Diff("Second generation: returns the input unchanged.", "Third generation: doubles the input."));

        ProgramGenome child = await Operator(client).ProposeAsync(Context(parent));

        Assert.Equal("def solve(x):\n    return x * 2\n", child.Source);
        Assert.Equal("Third generation: doubles the input.\n", child.Description);

        string prompt = string.Join("\n", client.Conversations[0].Select(message => message.Text));
        Assert.Contains("Second generation: returns the input unchanged.", prompt, StringComparison.Ordinal);
        Assert.Contains("def solve(x):", prompt, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AReplyThatEditsOnlyTheDescriptionIsToldTheProgramDidNotChange()
    {
        // A description-only edit is still no new candidate, so it is rejected - but for the true reason. Applying
        // an empty program-block list used to refuse it with "no edit blocks were supplied", which the model could
        // see was false about its own answer and which said nothing about the routing that rejected it.
        var client = new FakeChatClient(
            Diff("Returns the input.", "Returns the input, documented."),
            Diff("    return x", "    return x + 1") + Diff("Returns the input.", "Adds one."));

        ProgramGenome child = await Operator(client).ProposeAsync(Context());

        Assert.Equal(2, client.Calls);
        Assert.Equal("def solve(x):\n    return x + 1\n", child.Source);

        string retry = client.Conversations[1][^1].Text;
        Assert.Contains("identical to the current one", retry, StringComparison.Ordinal);
        Assert.DoesNotContain("No edit blocks were supplied", retry, StringComparison.Ordinal);
    }

    [Fact]
    public void MaintainingADescriptionThroughAFullRewriteIsRefused()
    {
        // A full rewrite produces no edit blocks to route, so the combination could only fail once per proposal in a
        // way that reads like the model ignoring its instructions.
        var options = new ProgramEvolutionOptions();
        options.Prompt.ProgramsAsChangesDescription = true;
        options.Variation.Mode = ProgramEvolutionMode.FullRewrite;

        ArgumentException failure = Assert.Throws<ArgumentException>(() => options.Validate());
        Assert.Contains("full rewrite does not produce", failure.Message, StringComparison.Ordinal);
    }

    private static LlmProgramVariationOperator<double> Operator(FakeChatClient client)
    {
        var options = new ProgramEvolutionOptions();
        options.Prompt.ProgramsAsChangesDescription = true;
        options.Prompt.EvolutionMode = ProgramPromptEvolutionMode.Diff;
        return new LlmProgramVariationOperator<double>(client, options);
    }

    private static EvolutionVariationContext<ProgramGenome> Context(ProgramGenome? parent = null)
    {
        ProgramGenome genome = parent ?? new ProgramGenome(
            "def solve(x):\n    return x\n", ProgramLanguage.Python, "Returns the input.\n");
        var lineage = new EvolutionLineage(null, null, "seed", null, 0, 0, 0UL);
        var candidate = new EvolutionCandidate<ProgramGenome>(
            0, new EvolutionCanonicalGenome<ProgramGenome>(genome, genome.Id), lineage);
        var evaluation = new EvolutionEvaluation(
            0, genome.Id, EvolutionEvaluationStatus.Completed, 0.5, EvolutionOptimizationDirection.Maximize,
            new Dictionary<string, double>(StringComparer.Ordinal) { ["x"] = 0.5 },
            Array.Empty<double>(), Array.Empty<double>(), new EvolutionEvaluationCost(TimeSpan.Zero, 1, 1),
            lineage, EvolutionCacheStatus.Miss, Array.Empty<EvolutionDiagnostic>(),
            "task-v1", "evaluator-v1", "config-v1");
        var entry = new EvolutionArchiveEntry<ProgramGenome>(new EvolutionCellKey(new[] { 1 }), candidate, evaluation);
        return new EvolutionVariationContext<ProgramGenome>(
            entry, Array.Empty<EvolutionArchiveEntry<ProgramGenome>>(), new StableRandom(1234UL, 7UL), 0, 0);
    }

    private static string Diff(string search, string replace) =>
        "<<<<<<< SEARCH\n" + search + "\n=======\n" + replace + "\n>>>>>>> REPLACE\n";

    private static ProgramDiffBlock Block(string search, string replace, int ordinal = 0) =>
        new(search, replace, ordinal);
}
