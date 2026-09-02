using System;
using System.Collections.Generic;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Evolution.Prompts;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class ProgramPromptBuilderTests
{
    private const string ParentSource = "def solve(n):\n    return sum(range(n))\n";

    [Fact]
    public void ThePromptIsReproducibleForTheSameOptionsInputsAndRandomStream()
    {
        // The property a benchmark needs and OpenEvolve cannot offer: its diverse
        // sampling and template variations come from Python's unseeded global RNG.
        ProgramEvolutionPromptOptions options = OptionsWithVariations();
        var first = new ProgramPromptBuilder(options);
        var second = new ProgramPromptBuilder(options);

        ProgramPromptResult a = first.Build(RichContext(), new StableRandom(4242UL));
        ProgramPromptResult b = second.Build(RichContext(), new StableRandom(4242UL));

        Assert.Equal(a.UserText, b.UserText);
        Assert.Equal(a.SystemText, b.SystemText);
        Assert.Equal(a.VariationChoices, b.VariationChoices);
        Assert.Equal(first.VersionHash, second.VersionHash);
    }

    [Fact]
    public void ThePromptIsReproducibleWhenTheStreamIsDerivedFromTheParent()
    {
        var builder = new ProgramPromptBuilder();
        Assert.Equal(builder.Build(RichContext()).UserText, builder.Build(RichContext()).UserText);
    }

    [Fact]
    public void TheVariationSequenceIsAFunctionOfTheSeedAlone()
    {
        ProgramEvolutionPromptOptions options = OptionsWithVariations();
        var builder = new ProgramPromptBuilder(options);

        var firstRun = new List<string>();
        var stream = new StableRandom(7UL);
        for (int index = 0; index < 8; index++) firstRun.Add(builder.Build(RichContext(), stream).VariationChoices["tone"]);

        var secondRun = new List<string>();
        var replay = new StableRandom(7UL);
        for (int index = 0; index < 8; index++) secondRun.Add(builder.Build(RichContext(), replay).VariationChoices["tone"]);

        Assert.Equal(firstRun, secondRun);
        Assert.Contains(firstRun, choice => !string.Equals(choice, firstRun[0], StringComparison.Ordinal));
    }

    [Fact]
    public void ChosenVariationTextReachesTheRenderedPrompt()
    {
        ProgramEvolutionPromptOptions options = OptionsWithVariations();
        var builder = new ProgramPromptBuilder(options);
        ProgramPromptResult result = builder.Build(RichContext(), new StableRandom(11UL));

        Assert.Contains(result.VariationChoices["tone"], result.UserText, StringComparison.Ordinal);
    }

    [Fact]
    public void VariationsAreNotDrawnWhenStochasticityIsOff()
    {
        ProgramEvolutionPromptOptions options = OptionsWithVariations();
        options.UseTemplateStochasticity = false;
        var builder = new ProgramPromptBuilder(options);

        ProgramPromptResult result = builder.Build(RichContext(), new StableRandom(11UL));
        Assert.Empty(result.VariationChoices);
    }

    [Fact]
    public void AnUnresolvablePlaceholderIsRejectedWhenTheBuilderIsConstructed()
    {
        var options = new ProgramEvolutionPromptOptions();
        options.TemplateOverrides[ProgramPromptTemplateKey.DiffUser] =
            "Improve {current_program} for {mystery_value}.";

        ArgumentException error = Assert.Throws<ArgumentException>(() => new ProgramPromptBuilder(options));
        Assert.Contains("mystery_value", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ACustomVariableMakesAnOtherwiseUnresolvablePlaceholderValid()
    {
        var options = new ProgramEvolutionPromptOptions();
        options.TemplateOverrides[ProgramPromptTemplateKey.DiffUser] =
            "Improve {current_program} for {mystery_value}.";
        options.CustomVariables["mystery_value"] = "throughput";

        var builder = new ProgramPromptBuilder(options);
        Assert.Contains("throughput", builder.Build(SimpleContext()).UserText, StringComparison.Ordinal);
    }

    [Fact]
    public void ACustomVariableThatShadowsASuppliedNameIsRejected()
    {
        var options = new ProgramEvolutionPromptOptions();
        options.CustomVariables["current_program"] = "hijacked";
        Assert.Throws<ArgumentException>(() => new ProgramPromptBuilder(options));
    }

    [Fact]
    public void ASystemMessageNamingNoTemplateIsRejectedUnlessItIsMarkedLiteral()
    {
        var options = new ProgramEvolutionPromptOptions { SystemMessage = "Be extremely terse." };
        Assert.Throws<ArgumentException>(() => new ProgramPromptBuilder(options));

        options.SystemMessageMode = ProgramPromptSystemMessageMode.Literal;
        var builder = new ProgramPromptBuilder(options);
        Assert.Equal("Be extremely terse.", builder.Build(SimpleContext()).SystemText);
    }

    [Fact]
    public void ASystemMessageNamingATemplateUsesThatTemplatesText()
    {
        var options = new ProgramEvolutionPromptOptions { SystemMessage = "evaluator_system_message" };
        var builder = new ProgramPromptBuilder(options);
        ProgramPromptResult result = builder.Build(SimpleContext());

        Assert.Equal(
            builder.Templates.GetTemplate(ProgramPromptTemplateKey.EvaluatorSystemMessage).Text,
            result.SystemText);
    }

    [Fact]
    public void ExtraSystemTextIsAppendedToTheSystemMessage()
    {
        var options = new ProgramEvolutionPromptOptions { ExtraSystemText = "Never use recursion." };
        ProgramPromptResult result = new ProgramPromptBuilder(options).Build(SimpleContext());
        Assert.EndsWith("Never use recursion.", result.SystemText, StringComparison.Ordinal);
    }

    [Fact]
    public void TheConfiguredEvaluatorSystemMessageIsHonoured()
    {
        // Upstream accepts prompt.evaluator_system_message and then never reads it,
        // because its controller hard-codes its own.
        var options = new ProgramEvolutionPromptOptions { EvaluatorSystemMessage = "You are a strict reviewer." };
        var builder = new ProgramPromptBuilder(options);

        IReadOnlyList<AiDotNet.Agentic.Models.ChatMessage> messages = builder.BuildEvaluationMessages(
            new ProgramGenome(ParentSource), new[] { "Readability", "Efficiency" });

        Assert.Equal("You are a strict reviewer.", messages[0].Text);
        Assert.Contains("Readability", messages[1].Text, StringComparison.Ordinal);
        Assert.Contains("\"readability\"", messages[1].Text, StringComparison.Ordinal);
    }

    [Fact]
    public void AutoModeRewritesShortProgramsAndPatchesLongOnes()
    {
        var options = new ProgramEvolutionPromptOptions
        {
            EvolutionMode = ProgramPromptEvolutionMode.AutoBySize,
            AutoFullRewriteBelowChars = 100
        };
        var builder = new ProgramPromptBuilder(options);

        ProgramPromptResult small = builder.Build(SimpleContext());
        Assert.Equal(ProgramPromptEvolutionMode.FullRewrite, small.Mode);
        Assert.Equal(ProgramPromptTemplateKey.FullRewriteUser, small.UserTemplateKey);

        var large = new ProgramPromptContext(new ProgramGenome(new string('x', 500)));
        ProgramPromptResult big = builder.Build(large);
        Assert.Equal(ProgramPromptEvolutionMode.Diff, big.Mode);
        Assert.Equal(ProgramPromptTemplateKey.DiffUser, big.UserTemplateKey);
    }

    [Fact]
    public void TheDiffPromptQuotesTheConfiguredMarkersRatherThanHardCodedOnes()
    {
        var programOptions = new ProgramEvolutionOptions();
        programOptions.Diff.SearchMarker = "<<< FIND";
        programOptions.Diff.DividerMarker = "--- WITH";
        programOptions.Diff.ReplaceMarker = ">>> DONE";

        var builder = new ProgramPromptBuilder(new ProgramEvolutionPromptOptions(), programOptions);
        string user = builder.Build(SimpleContext()).UserText;

        Assert.Contains("<<< FIND", user, StringComparison.Ordinal);
        Assert.Contains("--- WITH", user, StringComparison.Ordinal);
        Assert.Contains(">>> DONE", user, StringComparison.Ordinal);
    }

    [Fact]
    public void EvolveBlockInstructionsAppearOnlyWhenBlocksAreEnforced()
    {
        var free = new ProgramPromptBuilder(new ProgramEvolutionPromptOptions(), new ProgramEvolutionOptions());
        Assert.DoesNotContain("Change only the lines between", free.Build(SimpleContext()).UserText, StringComparison.Ordinal);

        var enforced = new ProgramPromptBuilder(
            new ProgramEvolutionPromptOptions(),
            new ProgramEvolutionOptions { EnforceEvolveBlocks = true, Language = ProgramLanguage.Python });
        Assert.Contains("Change only the lines between", enforced.Build(SimpleContext()).UserText, StringComparison.Ordinal);
    }

    [Fact]
    public void FeatureCoordinatesShowTheBinAndTheGridSize()
    {
        // Upstream renders the value alone, which tells the model a number but not
        // where in the archive the grid still has room.
        var builder = new ProgramPromptBuilder();
        string user = builder.Build(RichContext()).UserText;

        Assert.Contains("length=0.42 [bin 4/10]", user, StringComparison.Ordinal);
        Assert.Contains("complexity=0.7 [bin 8/10]", user, StringComparison.Ordinal);
    }

    [Fact]
    public void FeatureCoordinatesCanBeTurnedOff()
    {
        var builder = new ProgramPromptBuilder(new ProgramEvolutionPromptOptions { IncludeFeatureCoordinates = false });
        Assert.DoesNotContain("[bin ", builder.Build(RichContext()).UserText, StringComparison.Ordinal);
    }

    [Fact]
    public void ArtifactsAreRedactedAndTruncatedToTheByteBudget()
    {
        var noise = new System.Text.StringBuilder();
        for (int index = 0; index < 400; index++) noise.Append("line of run output\n");

        var context = SimpleContext();
        context.Artifacts = new List<ProgramPromptArtifact>
        {
            new("stderr", "api_key=sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD\n" + noise)
        };

        var options = new ProgramEvolutionPromptOptions { MaxArtifactBytes = 64 };
        string user = new ProgramPromptBuilder(options).Build(context).UserText;

        Assert.Contains("### stderr", user, StringComparison.Ordinal);
        Assert.DoesNotContain("sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD", user, StringComparison.Ordinal);
        Assert.Contains("truncated at 64 bytes", user, StringComparison.Ordinal);
        Assert.DoesNotContain("line of run output\nline of run output\nline of run output\n", user, StringComparison.Ordinal);
    }

    [Fact]
    public void ArtifactsAreOmittedWhenTheyAreTurnedOff()
    {
        var context = SimpleContext();
        context.Artifacts = new List<ProgramPromptArtifact> { new("stdout", "hello") };

        var options = new ProgramEvolutionPromptOptions { IncludeArtifacts = false };
        Assert.DoesNotContain("### stdout", new ProgramPromptBuilder(options).Build(context).UserText, StringComparison.Ordinal);
    }

    [Fact]
    public void OnlyTheConfiguredNumberOfArtifactsIsQuoted()
    {
        var context = SimpleContext();
        context.Artifacts = new List<ProgramPromptArtifact>
        {
            new("one", "a"), new("two", "b"), new("three", "c")
        };

        string user = new ProgramPromptBuilder(new ProgramEvolutionPromptOptions { MaxArtifactCount = 2 })
            .Build(context).UserText;

        Assert.Contains("### one", user, StringComparison.Ordinal);
        Assert.Contains("### two", user, StringComparison.Ordinal);
        Assert.DoesNotContain("### three", user, StringComparison.Ordinal);
    }

    [Fact]
    public void DiagnosticMessagesAreRedactedAndBounded()
    {
        var context = SimpleContext();
        context.Diagnostics = new List<EvolutionDiagnostic>
        {
            new("evaluation_failed", "connect failed with token=abcdefghijklmnop"),
            new("second", "b"),
            new("third", "c")
        };

        string user = new ProgramPromptBuilder(new ProgramEvolutionPromptOptions { MaxDiagnostics = 2 })
            .Build(context).UserText;

        Assert.Contains("evaluation_failed", user, StringComparison.Ordinal);
        Assert.DoesNotContain("abcdefghijklmnop", user, StringComparison.Ordinal);
        Assert.DoesNotContain("third", user, StringComparison.Ordinal);
        Assert.Contains("and 1 more.", user, StringComparison.Ordinal);
    }

    [Fact]
    public void InspirationsAreDeduplicatedByProgramContentNotByArchiveIdentity()
    {
        // Two archive entries can hold byte-identical programs; upstream dedups by
        // program id only, so the same code is quoted twice.
        var shared = new ProgramGenome("def f():\n    return 1\n");
        var context = SimpleContext();
        context.TopPrograms = new List<ProgramPromptExample>
        {
            new(shared, ProgramPromptExampleKind.TopProgram, 0.9)
        };
        context.Inspirations = new List<ProgramPromptExample>
        {
            new(new ProgramGenome("def f():  \n    return 1\n"), ProgramPromptExampleKind.Diverse, 0.5),
            new(new ProgramGenome("def g():\n    return 2\n"), ProgramPromptExampleKind.Diverse, 0.4)
        };

        string user = new ProgramPromptBuilder().Build(context).UserText;
        Assert.Contains("### Inspiration 1", user, StringComparison.Ordinal);
        Assert.DoesNotContain("### Inspiration 2", user, StringComparison.Ordinal);
        Assert.Contains("def g():", user, StringComparison.Ordinal);
    }

    [Fact]
    public void AnInspirationsChangeDescriptionReachesThePrompt()
    {
        // The equivalent upstream fragment is formatted with a {changes} argument
        // its text does not contain, so the description never appears.
        string user = new ProgramPromptBuilder().Build(RichContext()).UserText;
        Assert.Contains("Modification: switched to a generator", user, StringComparison.Ordinal);
    }

    [Fact]
    public void ExampleSizeIsLabelledFromTheConfiguredLineThresholds()
    {
        var context = SimpleContext();
        context.Inspirations = new List<ProgramPromptExample>
        {
            new(new ProgramGenome("a = 1\n"), ProgramPromptExampleKind.Diverse, 0.5)
        };

        string concise = new ProgramPromptBuilder(
                new ProgramEvolutionPromptOptions { ConciseImplementationMaxLines = 4 })
            .Build(context).UserText;
        Assert.Contains("Concise implementation", concise, StringComparison.Ordinal);

        string comprehensive = new ProgramPromptBuilder(
                new ProgramEvolutionPromptOptions
                {
                    ConciseImplementationMaxLines = null,
                    ComprehensiveImplementationMinLines = 1
                })
            .Build(context).UserText;
        Assert.Contains("Comprehensive implementation", comprehensive, StringComparison.Ordinal);
    }

    [Fact]
    public void TopProgramsRespectTheConfiguredCount()
    {
        var context = SimpleContext();
        context.TopPrograms = new List<ProgramPromptExample>
        {
            new(new ProgramGenome("a = 1\n"), ProgramPromptExampleKind.TopProgram, 0.9),
            new(new ProgramGenome("a = 2\n"), ProgramPromptExampleKind.TopProgram, 0.8),
            new(new ProgramGenome("a = 3\n"), ProgramPromptExampleKind.TopProgram, 0.7)
        };

        var options = new ProgramEvolutionPromptOptions { NumTopPrograms = 2, NumDiversePrograms = 0 };
        string user = new ProgramPromptBuilder(options).Build(context).UserText;

        Assert.Contains("### Program 1", user, StringComparison.Ordinal);
        Assert.Contains("### Program 2", user, StringComparison.Ordinal);
        Assert.DoesNotContain("### Program 3", user, StringComparison.Ordinal);
    }

    [Fact]
    public void DiverseProgramsAreSampledFromTheRemainderAndAreReproducible()
    {
        var context = SimpleContext();
        var top = new List<ProgramPromptExample>();
        for (int index = 0; index < 8; index++)
        {
            top.Add(new ProgramPromptExample(
                new ProgramGenome("value = " + index.ToString(System.Globalization.CultureInfo.InvariantCulture) + "\n"),
                ProgramPromptExampleKind.TopProgram,
                1.0 - (index * 0.1)));
        }

        context.TopPrograms = top;
        var options = new ProgramEvolutionPromptOptions { NumTopPrograms = 2, NumDiversePrograms = 2 };
        var builder = new ProgramPromptBuilder(options);

        string first = builder.Build(context, new StableRandom(99UL)).UserText;
        string second = builder.Build(context, new StableRandom(99UL)).UserText;

        Assert.Equal(first, second);
        Assert.Contains("### Program D1", first, StringComparison.Ordinal);
        Assert.Contains("### Program D2", first, StringComparison.Ordinal);
    }

    [Fact]
    public void PreviousAttemptsAreSummarizedNewestFirstWithAnOutcome()
    {
        var context = SimpleContext();
        context.PreviousAttempts = new List<ProgramPromptAttempt>
        {
            new(1, "first idea",
                new Dictionary<string, double> { ["score"] = 0.1 },
                new Dictionary<string, double> { ["score"] = 0.2 }),
            new(2, "second idea",
                new Dictionary<string, double> { ["score"] = 0.5 },
                new Dictionary<string, double> { ["score"] = 0.1 })
        };

        string user = new ProgramPromptBuilder().Build(context).UserText;
        int second = user.IndexOf("### Attempt 2", StringComparison.Ordinal);
        int first = user.IndexOf("### Attempt 1", StringComparison.Ordinal);

        Assert.True(second >= 0 && first > second, "The most recent attempt must be listed first.");
        Assert.Contains("Every measurement improved", user, StringComparison.Ordinal);
        Assert.Contains("Every measurement regressed", user, StringComparison.Ordinal);
    }

    [Fact]
    public void ImprovementAreasReportTheDirectionFitnessMoved()
    {
        var improved = SimpleContext();
        improved.ParentQuality = 0.7;
        improved.PreviousQuality = 0.5;
        Assert.Contains("Fitness improved", new ProgramPromptBuilder().Build(improved).UserText, StringComparison.Ordinal);

        var declined = SimpleContext();
        declined.ParentQuality = 0.4;
        declined.PreviousQuality = 0.5;
        Assert.Contains("Fitness declined", new ProgramPromptBuilder().Build(declined).UserText, StringComparison.Ordinal);

        var stable = SimpleContext();
        stable.ParentQuality = 0.5;
        stable.PreviousQuality = 0.5;
        Assert.Contains("Fitness unchanged", new ProgramPromptBuilder().Build(stable).UserText, StringComparison.Ordinal);
    }

    [Fact]
    public void ImprovementDirectionFollowsTheOptimizationDirection()
    {
        var context = SimpleContext();
        context.Direction = EvolutionOptimizationDirection.Minimize;
        context.ParentQuality = 0.4;
        context.PreviousQuality = 0.5;
        Assert.Contains("Fitness improved", new ProgramPromptBuilder().Build(context).UserText, StringComparison.Ordinal);
    }

    [Fact]
    public void CoverageHintsAppearOnlyWhenTheyAreEnabled()
    {
        var context = SimpleContext();
        context.EmptyNeighborCells = new List<string> { "length bin 5", "complexity bin 2" };

        Assert.DoesNotContain("still empty", new ProgramPromptBuilder().Build(context).UserText, StringComparison.Ordinal);

        var options = new ProgramEvolutionPromptOptions { IncludeCoverageHints = true };
        string user = new ProgramPromptBuilder(options).Build(context).UserText;
        Assert.Contains("length bin 5; complexity bin 2", user, StringComparison.Ordinal);
    }

    [Fact]
    public void TheProgramSnippetIsBoundedToItsCeiling()
    {
        var context = new ProgramPromptContext(new ProgramGenome(new string('q', 5_000)));
        var options = new ProgramEvolutionPromptOptions { MaxProgramSnippetChars = 256 };
        string user = new ProgramPromptBuilder(options).Build(context).UserText;

        Assert.DoesNotContain(new string('q', 1_000), user, StringComparison.Ordinal);
        Assert.Contains("prompt truncated", user, StringComparison.Ordinal);
    }

    [Fact]
    public void AnOverBudgetPromptDropsOptionalSectionsAndReportsTruncation()
    {
        var context = RichContext();
        context.Artifacts = new List<ProgramPromptArtifact> { new("stdout", new string('o', 4_000)) };

        var options = new ProgramEvolutionPromptOptions { MaxPromptChars = 900, MaxProgramSnippetChars = 256 };
        ProgramPromptResult result = new ProgramPromptBuilder(options).Build(context);

        Assert.True(result.WasTruncated);
        Assert.True(result.UserText.Length <= options.MaxPromptChars);
        Assert.Contains("def solve", result.UserText, StringComparison.Ordinal);
    }

    [Fact]
    public void TheTaskDescriptionReachesThePromptWhenSupplied()
    {
        var options = new ProgramEvolutionPromptOptions { TaskDescription = "Return the nth triangular number." };
        string user = new ProgramPromptBuilder(options).Build(SimpleContext()).UserText;
        Assert.Contains("Return the nth triangular number.", user, StringComparison.Ordinal);
    }

    [Fact]
    public void MessagesAreASystemMessageFollowedByTheUserMessage()
    {
        ProgramPromptResult result = new ProgramPromptBuilder().Build(SimpleContext());
        Assert.Equal(2, result.Messages.Count);
        Assert.Equal(AiDotNet.Agentic.Models.ChatRole.System, result.Messages[0].Role);
        Assert.Equal(AiDotNet.Agentic.Models.ChatRole.User, result.Messages[1].Role);
        Assert.Equal(result.UserText, result.Messages[1].Text);
    }

    [Fact]
    public void ChangesDescriptionModeQuotesSummariesAndRequiresTheBlockToBeUpdated()
    {
        var context = SimpleContext();
        context.ChangesDescription = "1. use a closed form";

        var options = new ProgramEvolutionPromptOptions { ProgramsAsChangesDescription = true };
        ProgramPromptResult result = new ProgramPromptBuilder(options).Build(context);

        Assert.Contains("Changes Description", result.UserText, StringComparison.Ordinal);
        Assert.Contains("1. use a closed form", result.UserText, StringComparison.Ordinal);
        Assert.Contains("Changes Description", result.SystemText, StringComparison.Ordinal);
    }

    [Fact]
    public void EditingATemplateChangesTheBuilderVersionHash()
    {
        // Resume must notice that a run's wording changed part-way through.
        var baseline = new ProgramPromptBuilder();
        var options = new ProgramEvolutionPromptOptions();
        options.TemplateOverrides[ProgramPromptTemplateKey.SystemMessage] = "Different instructions.";

        Assert.NotEqual(baseline.VersionHash, new ProgramPromptBuilder(options).VersionHash);
    }

    [Fact]
    public void ChangingASizeLimitChangesTheBuilderVersionHash()
    {
        var baseline = new ProgramPromptBuilder();
        var changed = new ProgramPromptBuilder(new ProgramEvolutionPromptOptions { MaxArtifactBytes = 1_024 });
        Assert.NotEqual(baseline.VersionHash, changed.VersionHash);
    }

    [Fact]
    public void OptionsAreCopiedSoLaterMutationCannotChangeARunningBuilder()
    {
        var options = new ProgramEvolutionPromptOptions { NumTopPrograms = 1 };
        var builder = new ProgramPromptBuilder(options);
        options.NumTopPrograms = 9;

        Assert.Equal(1, builder.GetPromptOptions().NumTopPrograms);
    }

    [Fact]
    public void ANonFiniteScoreIsRejected()
    {
        var context = SimpleContext();
        context.ParentQuality = double.NaN;
        Assert.Throws<ArgumentException>(() => new ProgramPromptBuilder().Build(context));
    }

    [Fact]
    public void MisalignedFeatureListsAreRejected()
    {
        var context = SimpleContext();
        context.FeatureDimensions = new List<string> { "length", "complexity" };
        context.FeatureBins = new List<int> { 1 };

        Assert.Throws<ArgumentException>(() => new ProgramPromptBuilder().Build(context));
    }

    private static ProgramEvolutionPromptOptions OptionsWithVariations()
    {
        var options = new ProgramEvolutionPromptOptions();
        options.TemplateVariations["tone"] = new List<string> { "Be bold.", "Be careful.", "Be surgical." };
        options.TemplateOverrides[ProgramPromptTemplateKey.DiffUser] =
            AiDotNet.Evolution.Prompts.ProgramPromptTemplateSet.CreateDefault()
                .GetTemplate(ProgramPromptTemplateKey.DiffUser).Text + "\n{tone}";
        return options;
    }

    private static ProgramPromptContext SimpleContext() => new(new ProgramGenome(ParentSource, ProgramLanguage.Python));

    private static ProgramPromptContext RichContext()
    {
        var context = new ProgramPromptContext(new ProgramGenome(ParentSource, ProgramLanguage.Python))
        {
            ParentQuality = 0.625,
            PreviousQuality = 0.5,
            ParentMetrics = new Dictionary<string, double> { ["accuracy"] = 0.625, ["runtime"] = 12.5 },
            ParentDescriptors = new Dictionary<string, double> { ["length"] = 0.42, ["complexity"] = 0.7 },
            FeatureDimensions = new List<string> { "length", "complexity" },
            FeatureBins = new List<int> { 3, 7 },
            FeatureBinCounts = new List<int> { 10, 10 },
            TopPrograms = new List<ProgramPromptExample>
            {
                new(new ProgramGenome("def solve(n):\n    return n * (n - 1) // 2\n"),
                    ProgramPromptExampleKind.TopProgram,
                    0.81,
                    new Dictionary<string, double> { ["length"] = 0.2 }),
                new(new ProgramGenome("def solve(n):\n    total = 0\n    return total\n"),
                    ProgramPromptExampleKind.TopProgram,
                    0.44,
                    new Dictionary<string, double> { ["length"] = 0.6 })
            },
            Inspirations = new List<ProgramPromptExample>
            {
                new(new ProgramGenome("def solve(n):\n    return sum(i for i in range(n))\n"),
                    ProgramPromptExampleKind.Migrant,
                    0.55,
                    new Dictionary<string, double> { ["complexity"] = 0.95 },
                    "switched to a generator")
            },
            PreviousAttempts = new List<ProgramPromptAttempt>
            {
                new(1, "vectorized the loop",
                    new Dictionary<string, double> { ["accuracy"] = 0.5 },
                    new Dictionary<string, double> { ["accuracy"] = 0.4 })
            }
        };

        return context;
    }
}
