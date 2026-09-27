using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class ProgramEvolutionTaskTests
{
    private static readonly EvolutionEvaluationContext Context = new(0, 1234UL, 7UL, 1);

    private static EvolutionCandidate<ProgramGenome> Candidate(ProgramGenome genome) =>
        new(0,
            new EvolutionCanonicalGenome<ProgramGenome>(genome, genome.Id),
            new EvolutionLineage(null, null, "seed", null, 0, 0, 0UL));

    private static ProgramEvolutionTask Task(
        IReadOnlyDictionary<string, double>? evaluatorDescriptors = null,
        ProgramDescriptorSet? descriptors = null,
        ProgramEvolutionOptions? options = null)
    {
        var evaluator = new DelegateProgramFitnessEvaluator(
            (genome, context, cancellationToken) => new ValueTask<EvolutionTaskResult>(
                EvolutionTaskResult.Completed(
                    1.0,
                    evaluatorDescriptors ?? new Dictionary<string, double>(StringComparer.Ordinal))),
            "fake",
            "fake-v1");

        return new ProgramEvolutionTask(evaluator, descriptors, options);
    }

    [Fact]
    public async System.Threading.Tasks.Task CanonicalIdentityIsTheNormalizedSourceAndLanguageHash()
    {
        ProgramEvolutionTask task = Task();
        var genome = new ProgramGenome("print(1)  \r\n", ProgramLanguage.Python);

        EvolutionCanonicalGenome<ProgramGenome> canonical = await task.CanonicalizeAsync(genome);

        Assert.Equal(genome.Id, canonical.Id);
        Assert.Equal(ProgramGenome.ComputeId("print(1)", ProgramLanguage.Python), canonical.Id);
        Assert.NotEqual(ProgramGenome.ComputeId("print(1)"), canonical.Id);
        Assert.NotSame(genome, canonical.Genome);
        Assert.Equal(genome, canonical.Genome);
        Assert.Equal(genome.Source, canonical.Genome.Source);
        Assert.Equal(genome.Description, canonical.Genome.Description);
    }

    [Fact]
    public async System.Threading.Tasks.Task EvaluatorVersionHashIsForwarded()
    {
        ProgramEvolutionTask task = Task();
        Assert.Equal("fake-v1", task.EvaluatorVersionHash);
        Assert.Equal("program-evolution", task.Id);
        EvolutionTaskResult result = await task.EvaluateAsync(Candidate(new ProgramGenome("x")), Context);
        Assert.Equal(EvolutionEvaluationStatus.Completed, result.Status);
    }

    [Fact]
    public async System.Threading.Tasks.Task OversizedCandidatesAreRejectedWithoutBeingEvaluated()
    {
        var options = new ProgramEvolutionOptions { MaxProgramChars = 10 };
        ProgramEvolutionTask task = Task(options: options);

        EvolutionTaskResult result = await task.EvaluateAsync(
            Candidate(new ProgramGenome(new string('x', 50))), Context);

        Assert.Equal(EvolutionEvaluationStatus.Rejected, result.Status);
        Assert.Equal("program_too_long", result.Diagnostics[0].Code);
        Assert.Null(result.Quality);
    }

    [Fact]
    public async System.Threading.Tasks.Task DamagedEvolveBlocksAreRejectedWhenEnforced()
    {
        var options = new ProgramEvolutionOptions
        {
            Language = ProgramLanguage.Python,
            EnforceEvolveBlocks = true
        };

        ProgramEvolutionTask task = Task(options: options);

        EvolutionTaskResult missing = await task.EvaluateAsync(Candidate(new ProgramGenome("print(1)")), Context);
        Assert.Equal(EvolutionEvaluationStatus.Rejected, missing.Status);
        Assert.Equal("program_evolve_block_invalid", missing.Diagnostics[0].Code);

        EvolutionTaskResult unterminated = await task.EvaluateAsync(
            Candidate(new ProgramGenome("# EVOLVE-BLOCK-START\nprint(1)\n")), Context);
        Assert.Equal(EvolutionEvaluationStatus.Rejected, unterminated.Status);

        EvolutionTaskResult wellFormed = await task.EvaluateAsync(
            Candidate(new ProgramGenome("# EVOLVE-BLOCK-START\nprint(1)\n# EVOLVE-BLOCK-END\n")), Context);
        Assert.Equal(EvolutionEvaluationStatus.Completed, wellFormed.Status);
    }

    [Fact]
    public async System.Threading.Tasks.Task EvolveBlocksAreNotEnforcedByDefault()
    {
        EvolutionTaskResult result = await Task().EvaluateAsync(Candidate(new ProgramGenome("print(1)")), Context);
        Assert.Equal(EvolutionEvaluationStatus.Completed, result.Status);
    }

    [Fact]
    public async System.Threading.Tasks.Task DescriptorsAreMergedIntoCompletedResults()
    {
        var descriptors = new ProgramDescriptorSet(
            new ProgramLengthDescriptor(), new ProgramTokenComplexityDescriptor());

        EvolutionTaskResult result = await Task(descriptors: descriptors)
            .EvaluateAsync(Candidate(new ProgramGenome("x = 1")), Context);

        Assert.Equal(EvolutionEvaluationStatus.Completed, result.Status);
        Assert.Equal(5.0, result.Descriptors["length"]);
        Assert.Equal(3.0, result.Descriptors["tokenComplexity"]);
        Assert.Equal(1.0, result.Quality);
    }

    [Fact]
    public async System.Threading.Tasks.Task EvaluatorDescriptorsOverrideBuiltInsOfTheSameName()
    {
        var descriptors = new ProgramDescriptorSet(new ProgramLengthDescriptor());
        var evaluatorDescriptors = new Dictionary<string, double>(StringComparer.Ordinal) { ["length"] = 99.0 };

        EvolutionTaskResult result = await Task(evaluatorDescriptors, descriptors)
            .EvaluateAsync(Candidate(new ProgramGenome("x = 1")), Context);

        Assert.Equal(99.0, result.Descriptors["length"]);
    }

    [Fact]
    public async System.Threading.Tasks.Task FailedResultsPassThroughUnchanged()
    {
        var evaluator = new DelegateProgramFitnessEvaluator(
            (genome, context, cancellationToken) => new ValueTask<EvolutionTaskResult>(
                EvolutionTaskResult.Failed("boom", "synthetic")),
            "failing",
            "failing-v1");

        var task = new ProgramEvolutionTask(evaluator, new ProgramDescriptorSet(new ProgramLengthDescriptor()));
        EvolutionTaskResult result = await task.EvaluateAsync(Candidate(new ProgramGenome("x = 1")), Context);

        Assert.Equal(EvolutionEvaluationStatus.Failed, result.Status);
        Assert.Empty(result.Descriptors);
    }

    [Fact]
    public void VersionHashTracksOptionsAndDescriptors()
    {
        string baseline = Task().VersionHash;
        Assert.Equal(baseline, Task().VersionHash);

        string differentLanguage = Task(options: new ProgramEvolutionOptions { Language = ProgramLanguage.CSharp }).VersionHash;
        string differentBound = Task(options: new ProgramEvolutionOptions { MaxProgramChars = 500 }).VersionHash;
        string enforced = Task(options: new ProgramEvolutionOptions { EnforceEvolveBlocks = true }).VersionHash;
        string withDescriptors = Task(descriptors: new ProgramDescriptorSet(new ProgramLengthDescriptor())).VersionHash;

        Assert.NotEqual(baseline, differentLanguage);
        Assert.NotEqual(baseline, differentBound);
        Assert.NotEqual(baseline, enforced);
        Assert.NotEqual(baseline, withDescriptors);
    }

    [Fact]
    public void OptionsAreCopiedDefensively()
    {
        var options = new ProgramEvolutionOptions { MaxProgramChars = 4096 };
        ProgramEvolutionTask task = Task(options: options);

        options.MaxProgramChars = 1;
        Assert.Equal(4096, task.GetOptions().MaxProgramChars);
    }

    [Fact]
    public void InvalidArgumentsAreRejected()
    {
#pragma warning disable CS8600, CS8625
        Assert.Throws<ArgumentNullException>(() => new ProgramEvolutionTask(null));
#pragma warning restore CS8600, CS8625
        var evaluator = new DelegateProgramFitnessEvaluator(_ => 1.0);
        Assert.Throws<ArgumentException>(() => new ProgramEvolutionTask(evaluator, id: " "));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ProgramEvolutionTask(evaluator, options: new ProgramEvolutionOptions { MaxProgramChars = -1 }));
    }
}
