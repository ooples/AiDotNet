using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Evolution.Programs.Novelty;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

/// <summary>
/// Proves a near-duplicate is refused by the archive before it costs an evaluation, which is the only placement
/// that saves the budget an evaluator-side check would already have spent.
/// </summary>
public sealed class ProgramNoveltyGateTests
{
    [Fact]
    public async Task ANearDuplicateIsRejectedWithoutBeingEvaluated()
    {
        var task = new CountingProgramTask();
        var options = new EvolutionEngineOptions
        {
            RunId = "novelty-gate",
            Seed = 11UL,
            MaxEvaluationAttempts = 12,
            MaxProposals = 40,
            MaxGenerations = 40,
            ProposalBatchSize = 1,

            // Anything closer than this to an existing elite is a duplicate for the purposes of this run.
            NoveltyDistanceThreshold = 0.5
        };

        EvolutionRunResult<ProgramGenome> run = await RunAsync(task, options, new ProgramTokenSetDistance());

        // The variation operator only ever proposes a comment change, so every child is lexically identical to its
        // parent and therefore inside the threshold.
        Assert.Contains(
            run.Counters.StatusCounts,
            pair => pair.Key == EvolutionEvaluationStatus.Rejected && pair.Value > 0);

        // The seeds are evaluated; nothing after them is, because each proposal is refused at the archive.
        Assert.Equal(1, task.Evaluations);
    }

    [Fact]
    public async Task WithoutAThresholdTheSameProposalsAreEvaluated()
    {
        var task = new CountingProgramTask();
        var options = new EvolutionEngineOptions
        {
            RunId = "novelty-gate-off",
            Seed = 11UL,
            MaxEvaluationAttempts = 5,
            MaxProposals = 40,
            MaxGenerations = 40,
            ProposalBatchSize = 1
        };

        EvolutionRunResult<ProgramGenome> run = await RunAsync(task, options, new ProgramTokenSetDistance());

        // Default behaviour is unchanged: the gate is off, so the budget is spent on evaluations.
        Assert.True(task.Evaluations > 1);
        Assert.Equal(EvolutionStopReason.EvaluationBudgetReached, run.StopReason);
    }

    private static async Task<EvolutionRunResult<ProgramGenome>> RunAsync(
        CountingProgramTask task,
        EvolutionEngineOptions options,
        IGenomeDistance<ProgramGenome> distance)
    {
        var descriptors = new[]
        {
            new EvolutionDescriptorDefinition("length", 0, 4_000, 8, EvolutionOutOfRangePolicy.Clamp)
        };

        var engine = new EvolutionEngine<ProgramGenome>(
            task,
            new CommentAppendingVariation(),
            _ => new MapElitesArchive<ProgramGenome>(descriptors),
            options,
            genomeDistance: distance);

        return await engine.RunAsync(new[]
        {
            new ProgramGenome("def solve(x):\n    return x + 1\n", ProgramLanguage.Python)
        });
    }

    /// <summary>Scores every candidate the same and counts how many times it was asked.</summary>
    private sealed class CountingProgramTask : IEvolutionTask<ProgramGenome>
    {
        private int _evaluations;

        public int Evaluations => _evaluations;

        public string Id => "counting-program-task";

        public string VersionHash => "counting-v1";

        public string EvaluatorVersionHash => "counting-eval-v1";

        public ValueTask<EvolutionCanonicalGenome<ProgramGenome>> CanonicalizeAsync(
            ProgramGenome genome, CancellationToken cancellationToken = default) =>
            new(new EvolutionCanonicalGenome<ProgramGenome>(genome, genome.Id));

        public ValueTask<EvolutionTaskResult> EvaluateAsync(
            EvolutionCandidate<ProgramGenome> candidate,
            EvolutionEvaluationContext context,
            CancellationToken cancellationToken = default)
        {
            Interlocked.Increment(ref _evaluations);
            var descriptors = new Dictionary<string, double>(StringComparer.Ordinal)
            {
                ["length"] = candidate.CanonicalGenome.Genome.NormalizedSource.Length
            };

            return new ValueTask<EvolutionTaskResult>(new EvolutionTaskResult(
                EvolutionEvaluationStatus.Completed,
                0.5,
                EvolutionOptimizationDirection.Maximize,
                descriptors));
        }
    }

    /// <summary>Appends a distinct comment, so each child is a new identity but lexically the same program.</summary>
    private sealed class CommentAppendingVariation : IVariationOperator<ProgramGenome>
    {
        private int _counter;

        public string Id => "comment-appending";

        public string VersionHash => "comment-appending-v1";

        public ValueTask<ProgramGenome> ProposeAsync(
            EvolutionVariationContext<ProgramGenome> context, CancellationToken cancellationToken = default)
        {
            int index = Interlocked.Increment(ref _counter);
            ProgramGenome parent = context.Parent.Candidate.CanonicalGenome.Genome;
            return new ValueTask<ProgramGenome>(new ProgramGenome(
                parent.Source + "# " + index.ToString(System.Globalization.CultureInfo.InvariantCulture) + "\n",
                parent.Language));
        }
    }
}
