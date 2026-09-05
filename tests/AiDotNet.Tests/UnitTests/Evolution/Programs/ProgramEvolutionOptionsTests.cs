using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution.Programs;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class ProgramEvolutionOptionsTests
{
    [Fact]
    public void DefaultsAreValidAndTieTheSubsystemsTogether()
    {
        var options = new ProgramEvolutionOptions();
        options.Validate();

        Assert.Equal(ProgramLanguage.Generic, options.Language);
        Assert.Equal(100_000, options.MaxProgramChars);
        Assert.Empty(options.SeedPrograms);
        Assert.Empty(options.TestCases);
        Assert.Empty(options.Descriptors);
        Assert.Null(options.TaskDescription);
        Assert.Null(options.EvaluatorScript);
        Assert.Equal(ProgramLanguage.Python, options.EvaluatorScriptLanguage);
        Assert.Equal(ProgramEvolutionOptions.DefaultIncludeEliteSourceCount, options.IncludeEliteSourceCount);
        Assert.NotNull(options.Prompt);
        Assert.NotNull(options.Variation);
        Assert.NotNull(options.Sandbox);
        Assert.NotNull(options.Script);
        Assert.NotNull(options.Engine);
        Assert.NotNull(options.Diff);
    }

    [Fact]
    public void NestedSubsystemsAreCreatedOnceAndReused()
    {
        var options = new ProgramEvolutionOptions();

        Assert.Same(options.Prompt, options.Prompt);
        Assert.Same(options.Variation, options.Variation);
        Assert.Same(options.Sandbox, options.Sandbox);
        Assert.Same(options.Engine, options.Engine);
        Assert.Same(options.Diff, options.Diff);
    }

    [Fact]
    public void EvaluatorScriptForwardsToTheScriptOptions()
    {
        var options = new ProgramEvolutionOptions
        {
            EvaluatorScript = "def evaluate(program):\n    return {}\n",
            EvaluatorScriptLanguage = ProgramLanguage.JavaScript
        };

        Assert.Equal(options.Script.EvaluatorScript, options.EvaluatorScript);
        Assert.Equal(ProgramLanguage.JavaScript, options.Script.EvaluatorScriptLanguage);
    }

    [Fact]
    public void CloneIsDeepAcrossEverySubsystem()
    {
        var options = new ProgramEvolutionOptions
        {
            Language = ProgramLanguage.CSharp,
            TaskDescription = "original",
            MaxProgramChars = 4_000,
            IncludeEliteSourceCount = 3,
            MaxEliteSourceChars = 200
        };
        options.SeedPrograms.Add("class Solver { }");
        options.TestCases.Add(new ProgramInputOutputExample { Input = "1", ExpectedOutput = "2" });
        options.Descriptors.Add(new ProgramLengthDescriptor());
        options.Variation.MaxProposalRetries = 5;
        options.Prompt.NumTopPrograms = 1;
        options.Engine.Seed = 9;
        options.Sandbox.Limits.TimeLimitSeconds = 11;

        ProgramEvolutionOptions copy = options.Clone();
        options.SeedPrograms.Add("class Other { }");
        options.TestCases[0].Input = "mutated";
        options.Variation.MaxProposalRetries = 1;
        options.Prompt.NumTopPrograms = 7;
        options.Engine.Seed = 1;
        options.Sandbox.Limits.TimeLimitSeconds = 1;
        options.TaskDescription = "mutated";

        Assert.Single(copy.SeedPrograms);
        Assert.Equal("1", copy.TestCases[0].Input);
        Assert.Equal(5, copy.Variation.MaxProposalRetries);
        Assert.Equal(1, copy.Prompt.NumTopPrograms);
        Assert.Equal(9UL, copy.Engine.Seed);
        Assert.Equal(11, copy.Sandbox.Limits.TimeLimitSeconds);
        Assert.Equal("original", copy.TaskDescription);
        Assert.Equal(3, copy.IncludeEliteSourceCount);
        Assert.Equal(ProgramLanguage.CSharp, copy.Language);
        Assert.Single(copy.Descriptors);
    }

    [Fact]
    public void CloningABareInstanceLeavesTheSubsystemsUncreated()
    {
        // The diff path constructs a bare instance per call, so cloning one must not allocate five option graphs.
        var options = new ProgramEvolutionOptions();
        ProgramEvolutionOptions copy = options.Clone();

        Assert.Equal(ProgramLanguage.Generic, copy.Language);
        Assert.Equal(ProgramDiffOptions.DefaultSearchMarker, copy.Diff.SearchMarker);
        Assert.Equal(2, copy.Variation.MaxProposalRetries);
    }

    [Fact]
    public void SeedGenomesAreBuiltInOrderAndCarryTheLanguage()
    {
        var options = new ProgramEvolutionOptions { Language = ProgramLanguage.Python };
        options.SeedPrograms.Add("a = 1\n");
        options.SeedPrograms.Add("b = 2\n");

        IReadOnlyList<ProgramGenome> genomes = options.CreateSeedGenomes();

        Assert.Equal(2, genomes.Count);
        Assert.Equal("a = 1\n", genomes[0].Source);
        Assert.Equal("b = 2\n", genomes[1].Source);
        Assert.All(genomes, genome => Assert.Equal(ProgramLanguage.Python, genome.Language));
    }

    [Fact]
    public void BlankAndOversizedSeedProgramsAreRejectedAtValidationTime()
    {
        var blank = new ProgramEvolutionOptions();
        blank.SeedPrograms.Add("   ");
        Assert.Throws<ArgumentException>(() => blank.Validate());

        var oversized = new ProgramEvolutionOptions { MaxProgramChars = 10 };
        oversized.SeedPrograms.Add(new string('x', 50));
        Assert.Throws<ArgumentException>(() => oversized.Validate());
    }

    [Fact]
    public void DuplicateDescriptorNamesAreRejectedAtValidationTime()
    {
        var options = new ProgramEvolutionOptions();
        options.Descriptors.Add(new ProgramLengthDescriptor());
        options.Descriptors.Add(new ProgramLengthDescriptor());

        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    public void InvalidNestedSubsystemsFailValidationOfTheAggregate()
    {
        var badVariation = new ProgramEvolutionOptions();
        badVariation.Variation.MaxProposalRetries = -1;
        Assert.Throws<ArgumentOutOfRangeException>(() => badVariation.Validate());

        var badPrompt = new ProgramEvolutionOptions();
        badPrompt.Prompt.NumTopPrograms = -1;
        Assert.Throws<ArgumentOutOfRangeException>(() => badPrompt.Validate());

        var badSandbox = new ProgramEvolutionOptions();
        badSandbox.Sandbox.Mode = ProgramSandboxMode.InProcessUnsafe;
        Assert.Throws<ArgumentException>(() => badSandbox.Validate());
    }

    [Fact]
    public void EliteRetentionBoundsAreRangeChecked()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new ProgramEvolutionOptions { IncludeEliteSourceCount = -1 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new ProgramEvolutionOptions { MaxEliteSourceChars = 0 }.Validate());
    }

    [Fact]
    public void DescriptorSetIsEmptyUntilDescriptorsAreConfigured()
    {
        var options = new ProgramEvolutionOptions();
        Assert.Equal(0, options.CreateDescriptorSet().Count);

        options.Descriptors.Add(new ProgramLengthDescriptor());
        options.Descriptors.Add(new ProgramTokenComplexityDescriptor());
        ProgramDescriptorSet set = options.CreateDescriptorSet();

        Assert.Equal(2, set.Count);
        Assert.Contains("length", set.Names);
        Assert.Contains("tokenComplexity", set.Names);
    }
}
