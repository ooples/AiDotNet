using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class ProgramEvolutionResultTests
{
    [Fact]
    public void EmptyRunReportsNoBestProgramAndNoElites()
    {
        EvolutionRunResult<ProgramGenome> run = Run(EvolutionOptimizationDirection.Maximize);

        ProgramEvolutionResult summary = ProgramEvolutionResult.Create(run);

        Assert.False(summary.HasBestProgram);
        Assert.Null(summary.BestProgram);
        Assert.Null(summary.BestQuality);
        Assert.Empty(summary.Elites);
        Assert.Empty(summary.BestDescriptors);
        Assert.Equal(0, summary.ArchiveCount);
        Assert.Same(ProgramEvolutionLlmUsage.Empty, summary.LlmUsage);
    }

    [Fact]
    public void ElitesAreOrderedBestFirstWhenMaximizing()
    {
        EvolutionRunResult<ProgramGenome> run = Run(
            EvolutionOptimizationDirection.Maximize, ("a = 1\n", 0.2), ("b = 2\n", 0.9), ("c = 3\n", 0.5));

        ProgramEvolutionResult summary = ProgramEvolutionResult.Create(run);

        Assert.Equal(new double?[] { 0.9, 0.5, 0.2 }, summary.Elites.Select(elite => elite.Quality));
        Assert.Equal("b = 2\n", summary.Elites[0].Source);
        Assert.Equal(0.9, summary.BestQuality);
        Assert.Equal(3, summary.ArchiveCount);
    }

    [Fact]
    public void ElitesAreOrderedBestFirstWhenMinimizing()
    {
        EvolutionRunResult<ProgramGenome> run = Run(
            EvolutionOptimizationDirection.Minimize, ("a = 1\n", 0.2), ("b = 2\n", 0.9), ("c = 3\n", 0.5));

        ProgramEvolutionResult summary = ProgramEvolutionResult.Create(run);

        Assert.Equal(new double?[] { 0.2, 0.5, 0.9 }, summary.Elites.Select(elite => elite.Quality));
        Assert.Equal(EvolutionOptimizationDirection.Minimize, summary.Direction);
    }

    [Fact]
    public void RetentionCountCapsTheEliteListButNotTheArchiveCount()
    {
        EvolutionRunResult<ProgramGenome> run = Run(
            EvolutionOptimizationDirection.Maximize, ("a = 1\n", 0.2), ("b = 2\n", 0.9), ("c = 3\n", 0.5));

        ProgramEvolutionResult summary = ProgramEvolutionResult.Create(run, includeEliteSourceCount: 1);

        Assert.Single(summary.Elites);
        Assert.Equal(3, summary.ArchiveCount);
        Assert.Equal(0.9, summary.Elites[0].Quality);
    }

    [Fact]
    public void SourceIsBoundedAndTruncationIsFlagged()
    {
        EvolutionRunResult<ProgramGenome> run = Run(
            EvolutionOptimizationDirection.Maximize, (new string('z', 500), 0.4));

        ProgramEvolutionResult summary = ProgramEvolutionResult.Create(run, maxEliteSourceChars: 64);

        Assert.Equal(64, summary.Elites[0].Source.Length);
        Assert.True(summary.Elites[0].IsSourceTruncated);
        Assert.EndsWith("...", summary.Elites[0].Source, StringComparison.Ordinal);

        // The best program itself is not bounded; only the elite copies are.
        Assert.Equal(500, summary.BestProgram?.Source.Length);
    }

    [Fact]
    public void UntruncatedSourceIsNotFlagged()
    {
        EvolutionRunResult<ProgramGenome> run = Run(EvolutionOptimizationDirection.Maximize, ("a = 1\n", 0.4));

        ProgramEvolutionResult summary = ProgramEvolutionResult.Create(run, maxEliteSourceChars: 4_000);

        Assert.False(summary.Elites[0].IsSourceTruncated);
        Assert.Equal("a = 1\n", summary.Elites[0].Source);
    }

    [Fact]
    public void ElitesCarryTheirCoordinatesAndCell()
    {
        EvolutionRunResult<ProgramGenome> run = Run(EvolutionOptimizationDirection.Maximize, ("a = 1\n", 0.4));

        ProgramEvolutionResult summary = ProgramEvolutionResult.Create(run);

        Assert.Equal(0.4, summary.Elites[0].Descriptors["score"]);
        Assert.Single(summary.Elites[0].Cell);
        Assert.Equal(1, summary.Elites[0].Cell[0]);
        Assert.Equal(ProgramLanguage.Python, summary.Elites[0].Language);
        Assert.Equal(0, summary.Elites[0].Island);
    }

    [Fact]
    public void UsageAndCheckpointPathAreCarriedThrough()
    {
        EvolutionRunResult<ProgramGenome> run = Run(EvolutionOptimizationDirection.Maximize, ("a = 1\n", 0.4));
        var usage = new ProgramEvolutionLlmUsage(proposals: 4, chatCalls: 7, retries: 3, inputTokens: 90, outputTokens: 10);

        ProgramEvolutionResult summary = ProgramEvolutionResult.Create(run, usage, "C:\\runs\\checkpoint.json");

        Assert.Equal(7, summary.LlmUsage.ChatCalls);
        Assert.Equal(100, summary.LlmUsage.TotalTokens);
        Assert.Equal(1.75, summary.LlmUsage.CallsPerProposal);
        Assert.Equal("C:\\runs\\checkpoint.json", summary.CheckpointPath);
    }

    [Fact]
    public void RetentionArgumentsAreRangeChecked()
    {
        EvolutionRunResult<ProgramGenome> run = Run(EvolutionOptimizationDirection.Maximize);

        Assert.Throws<ArgumentOutOfRangeException>(
            () => ProgramEvolutionResult.Create(run, includeEliteSourceCount: -1));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => ProgramEvolutionResult.Create(run, maxEliteSourceChars: 0));
    }

    [Fact]
    public void UsageTotalsAddAndRejectNegatives()
    {
        ProgramEvolutionLlmUsage combined = new ProgramEvolutionLlmUsage(proposals: 1, chatCalls: 2)
            .Add(new ProgramEvolutionLlmUsage(proposals: 3, chatCalls: 4, providerErrors: 1));

        Assert.Equal(4, combined.Proposals);
        Assert.Equal(6, combined.ChatCalls);
        Assert.Equal(1, combined.ProviderErrors);
        Assert.Equal(0, ProgramEvolutionLlmUsage.Empty.CallsPerProposal);
        Assert.Equal(0, ProgramEvolutionLlmUsage.Empty.AbandonRate);
        Assert.Throws<ArgumentOutOfRangeException>(() => new ProgramEvolutionLlmUsage(proposals: -1));
    }

    [Fact]
    public void SummaryStringNeverEchoesProgramText()
    {
        EvolutionRunResult<ProgramGenome> run = Run(
            EvolutionOptimizationDirection.Maximize, ("secret_token = 'abc'\n", 0.4));

        ProgramEvolutionResult summary = ProgramEvolutionResult.Create(run);

        Assert.DoesNotContain("secret_token", summary.ToString(), StringComparison.Ordinal);
        Assert.DoesNotContain("secret_token", summary.Elites[0].ToString(), StringComparison.Ordinal);
    }

    private static EvolutionRunResult<ProgramGenome> Run(
        EvolutionOptimizationDirection direction,
        params (string Source, double Quality)[] entries)
    {
        var archive = new MapElitesArchive<ProgramGenome>(
            new[] { new EvolutionDescriptorDefinition("score", 0, 1, 4, EvolutionOutOfRangePolicy.Clamp) },
            direction);

        long evaluationId = 0;
        foreach ((string source, double quality) in entries)
        {
            var genome = new ProgramGenome(source, ProgramLanguage.Python);
            var lineage = new EvolutionLineage(null, null, "seed", null, 0, 0, 0UL);
            var candidate = new EvolutionCandidate<ProgramGenome>(
                evaluationId, new EvolutionCanonicalGenome<ProgramGenome>(genome, genome.Id), lineage);
            var evaluation = new EvolutionEvaluation(
                evaluationId,
                genome.Id,
                EvolutionEvaluationStatus.Completed,
                quality,
                direction,
                new Dictionary<string, double>(StringComparer.Ordinal) { ["score"] = quality },
                Array.Empty<double>(),
                Array.Empty<double>(),
                new EvolutionEvaluationCost(TimeSpan.Zero, 1, 1),
                lineage,
                EvolutionCacheStatus.Miss,
                Array.Empty<EvolutionDiagnostic>(),
                "task-v1",
                "evaluator-v1",
                "config-v1");

            archive.TryAdd(candidate, evaluation);
            evaluationId++;
        }

        return new EvolutionRunResult<ProgramGenome>(
            EvolutionStopReason.EvaluationBudgetReached,
            new IEvolutionArchiveView<ProgramGenome>[] { archive },
            new EvolutionRunCounters(entries.Length, entries.Length, entries.Length,
                new Dictionary<EvolutionEvaluationStatus, long>
                {
                    [EvolutionEvaluationStatus.Completed] = entries.Length
                }),
            "state-hash");
    }
}
