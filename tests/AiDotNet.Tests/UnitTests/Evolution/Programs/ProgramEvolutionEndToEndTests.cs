using System.Globalization;
using System.Text;
using AiDotNet.Agentic.Models;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

/// <summary>
/// Drives the real <see cref="EvolutionEngine{TGenome}"/> over the program adapters with a scripted chat client and
/// a scripted fitness evaluator, so the whole wiring is exercised without a network, a sandbox, or a model.
/// </summary>
public sealed class ProgramEvolutionEndToEndTests
{
    private const int EvaluationBudget = 12;

    [Fact]
    public async Task RealEngineRunImprovesTheProgramAndFillsTheArchive()
    {
        var client = new ScriptedRewriteChatClient();
        RunPieces pieces = BuildRun(client);

        EvolutionRunResult<ProgramGenome> run = await pieces.Engine.RunAsync(pieces.Seeds);

        Assert.Equal(EvolutionStopReason.EvaluationBudgetReached, run.StopReason);
        Assert.Equal(EvaluationBudget, run.Counters.EvaluationAttempts);
        Assert.Equal(EvaluationBudget, run.Counters.CompletedEvaluations);

        int archived = run.Islands.Sum(island => island.Entries.Count);
        Assert.True(archived >= 3, "the archive should hold several behaviour cells, held " + archived);

        Assert.NotNull(run.Best);
        double seedQuality = Quality(new ProgramGenome(CannedProgram(0), ProgramLanguage.Python));
        Assert.NotNull(run.Best?.Evaluation.Quality);
        Assert.True(run.Best?.Evaluation.Quality > seedQuality,
            "the best archived program should beat the seed; got " + run.Best?.Evaluation.Quality);

        // Every evaluated candidate really came from the model, not from the seed being re-archived.
        Assert.True(client.Calls >= EvaluationBudget - pieces.Seeds.Count);
    }

    [Fact]
    public async Task TwoRunsWithTheSameSeedProduceAnIdenticalStateHash()
    {
        RunPieces first = BuildRun(new ScriptedRewriteChatClient());
        RunPieces second = BuildRun(new ScriptedRewriteChatClient());

        EvolutionRunResult<ProgramGenome> firstRun = await first.Engine.RunAsync(first.Seeds);
        EvolutionRunResult<ProgramGenome> secondRun = await second.Engine.RunAsync(second.Seeds);

        Assert.Equal(firstRun.StateHash, secondRun.StateHash);
        Assert.Equal(
            firstRun.Islands.SelectMany(island => island.Entries).Select(entry => entry.Evaluation.GenomeId),
            secondRun.Islands.SelectMany(island => island.Entries).Select(entry => entry.Evaluation.GenomeId));
        Assert.Equal(first.Operator.GetUsage().ChatCalls, second.Operator.GetUsage().ChatCalls);
    }

    [Fact]
    public async Task ADifferentSeedProducesADifferentStateHash()
    {
        RunPieces first = BuildRun(new ScriptedRewriteChatClient());
        RunPieces second = BuildRun(new ScriptedRewriteChatClient(), seed: 99UL);

        EvolutionRunResult<ProgramGenome> firstRun = await first.Engine.RunAsync(first.Seeds);
        EvolutionRunResult<ProgramGenome> secondRun = await second.Engine.RunAsync(second.Seeds);

        Assert.NotEqual(firstRun.StateHash, secondRun.StateHash);
    }

    [Fact]
    public async Task RunSummaryCarriesTheBestProgramTheElitesAndTheModelBill()
    {
        var client = new ScriptedRewriteChatClient();
        RunPieces pieces = BuildRun(client);

        EvolutionRunResult<ProgramGenome> run = await pieces.Engine.RunAsync(pieces.Seeds);
        ProgramEvolutionResult summary = ProgramEvolutionResult.Create(
            run, pieces.Options, pieces.Operator.GetUsage(), checkpointPath: null);

        Assert.True(summary.HasBestProgram);
        Assert.Equal(run.Best?.Evaluation.Quality, summary.BestQuality);
        Assert.Equal(run.StateHash, summary.StateHash);
        Assert.Equal(EvolutionOptimizationDirection.Maximize, summary.Direction);
        Assert.Equal(run.Islands.Sum(island => island.Entries.Count), summary.ArchiveCount);
        Assert.Equal(2, summary.IslandCount);

        Assert.NotEmpty(summary.Elites);
        Assert.True(summary.Elites.Count <= pieces.Options.IncludeEliteSourceCount);
        Assert.Equal(summary.BestQuality, summary.Elites[0].Quality);
        Assert.Contains("length", summary.Elites[0].Descriptors.Keys);

        // Elites are ordered best first in the archive's direction, with no gaps in the ordering.
        for (int index = 1; index < summary.Elites.Count; index++)
        {
            Assert.True(summary.Elites[index - 1].Quality >= summary.Elites[index].Quality);
        }

        Assert.True(summary.LlmUsage.ChatCalls > 0);
        Assert.Equal(summary.LlmUsage.Proposals, pieces.Operator.GetUsage().Proposals);
        Assert.Null(summary.CheckpointPath);
    }

    [Fact]
    public async Task EliteSourceIsBoundedAndFlaggedWhenTruncated()
    {
        var client = new ScriptedRewriteChatClient();
        RunPieces pieces = BuildRun(client);

        EvolutionRunResult<ProgramGenome> run = await pieces.Engine.RunAsync(pieces.Seeds);
        ProgramEvolutionResult summary = ProgramEvolutionResult.Create(
            run, includeEliteSourceCount: 2, maxEliteSourceChars: 40);

        Assert.Equal(2, summary.Elites.Count);
        Assert.All(summary.Elites, elite => Assert.True(elite.Source.Length <= 40));
        Assert.Contains(summary.Elites, elite => elite.IsSourceTruncated);
    }

    [Fact]
    public async Task UnusableAnswersCostNoEvaluationBudget()
    {
        // This client never produces a usable edit, so every proposal must fall back to the parent, be recognised
        // as a duplicate, and leave the evaluator untouched beyond the seeds. Upstream OpenEvolve evaluates the
        // identical-to-parent child instead, spending a full iteration on it. The evaluation cache is switched off
        // so the duplicate path itself is what is measured rather than a cache hit standing in for it.
        var client = new ProseOnlyChatClient();
        RunPieces pieces = BuildRun(client, enableEvaluationCache: false);

        EvolutionRunResult<ProgramGenome> run = await pieces.Engine.RunAsync(pieces.Seeds);

        Assert.Equal(pieces.Seeds.Count, run.Counters.CompletedEvaluations);
        Assert.Equal(pieces.Seeds.Count, pieces.Evaluations.Count);
        Assert.True(run.Counters.StatusCounts[EvolutionEvaluationStatus.Duplicate] > 0);
        Assert.True(pieces.Operator.GetUsage().AbandonedProposals > 0);
        Assert.Equal(0, pieces.Operator.GetUsage().ProviderErrors);
        Assert.Contains(
            pieces.Operator.GetRecentAttempts(),
            attempt => attempt.Outcome == ProgramProposalOutcome.Exhausted);
    }

    [Fact]
    public async Task UnusableAnswersAlsoCostNothingWithTheEvaluationCacheOn()
    {
        // With the cache on the repeat resolves as a cache hit rather than a duplicate, but the property that
        // matters is unchanged: the evaluator is never asked to score a child identical to its parent.
        var client = new ProseOnlyChatClient();
        RunPieces pieces = BuildRun(client);

        await pieces.Engine.RunAsync(pieces.Seeds);

        Assert.Equal(pieces.Seeds.Count, pieces.Evaluations.Count);
        Assert.True(pieces.Operator.GetUsage().AbandonedProposals > 0);
    }

    [Fact]
    public async Task OversizedCandidatesAreRejectedWithoutBeingScored()
    {
        var client = new GiantProgramChatClient();
        RunPieces pieces = BuildRun(client, maxProgramChars: 500, enableEvaluationCache: false);

        EvolutionRunResult<ProgramGenome> run = await pieces.Engine.RunAsync(pieces.Seeds);

        // The operator refuses the oversized answer itself, so the task's own rejection path is never even needed
        // and the evaluator only ever sees the seeds.
        Assert.Equal(pieces.Seeds.Count, pieces.Evaluations.Count);
        Assert.True(run.Counters.StatusCounts[EvolutionEvaluationStatus.Duplicate] > 0);
        Assert.Contains(
            pieces.Operator.GetRecentAttempts(),
            attempt => attempt.Outcome == ProgramProposalOutcome.TooLong);
    }

    private static RunPieces BuildRun(
        IChatClient<double> client,
        ulong seed = 4242UL,
        int maxProgramChars = 100_000,
        bool enableEvaluationCache = true)
    {
        var options = new ProgramEvolutionOptions
        {
            Language = ProgramLanguage.Python,
            MaxProgramChars = maxProgramChars,
            TaskDescription = "Accumulate as much as possible into the running total.",
            IncludeEliteSourceCount = 5,
            Variation =
            {
                Mode = ProgramEvolutionMode.FullRewrite,
                MaxInspirations = 2,
                FeatureDimensions = new List<string> { "length" },
                FeatureBinCounts = new List<int> { 12 }
            }
        };
        options.SeedPrograms.Add(CannedProgram(0));
        options.Descriptors.Add(new ProgramLengthDescriptor());
        options.Validate();

        var evaluations = new List<string>();
        var evaluator = new DelegateProgramFitnessEvaluator(genome =>
        {
            evaluations.Add(genome.Id);
            return Quality(genome);
        });

        var task = new ProgramEvolutionTask(evaluator, options.CreateDescriptorSet(), options);
        var variation = new LlmProgramVariationOperator<double>(client, options, options.Variation);

        var engineOptions = new EvolutionEngineOptions
        {
            RunId = "program-e2e",
            Seed = seed,
            MaxEvaluationAttempts = EvaluationBudget,
            MaxProposals = 40,
            MaxGenerations = 40,
            ProposalBatchSize = 4,
            MaxDegreeOfParallelism = 1,
            IslandCount = 2,
            MigrationInterval = 2,
            MigrantsPerIsland = 1,
            InspirationCount = 2,
            CheckpointInterval = 0,
            EnableEvaluationCache = enableEvaluationCache
        };

        var engine = new EvolutionEngine<ProgramGenome>(
            task,
            variation,
            _ => new MapElitesArchive<ProgramGenome>(new[]
            {
                new EvolutionDescriptorDefinition("length", 0, 600, 12, EvolutionOutOfRangePolicy.Clamp)
            }),
            engineOptions);

        return new RunPieces(engine, variation, options, options.CreateSeedGenomes(), evaluations);
    }

    private static double Quality(ProgramGenome genome)
    {
        int accumulations = 0;
        foreach (string line in genome.NormalizedSource.Split('\n'))
        {
            if (line.IndexOf("total +=", StringComparison.Ordinal) >= 0) accumulations++;
        }

        return accumulations / 32.0;
    }

    private static string CannedProgram(int index)
    {
        var builder = new StringBuilder("def solve(x):\n    total = x\n");
        for (int step = 0; step <= index; step++)
        {
            builder.Append("    total += ").Append((step + 1).ToString(CultureInfo.InvariantCulture)).Append('\n');
        }

        return builder.Append("    return total\n").ToString();
    }

    private sealed class RunPieces
    {
        public RunPieces(
            EvolutionEngine<ProgramGenome> engine,
            LlmProgramVariationOperator<double> variationOperator,
            ProgramEvolutionOptions options,
            IReadOnlyList<ProgramGenome> seeds,
            IReadOnlyList<string> evaluations)
        {
            Engine = engine;
            Operator = variationOperator;
            Options = options;
            Seeds = seeds;
            Evaluations = evaluations;
        }

        public EvolutionEngine<ProgramGenome> Engine { get; }
        public LlmProgramVariationOperator<double> Operator { get; }
        public ProgramEvolutionOptions Options { get; }
        public IReadOnlyList<ProgramGenome> Seeds { get; }
        public IReadOnlyList<string> Evaluations { get; }
    }

    /// <summary>Replays a fixed ladder of increasingly capable programs, one per request.</summary>
    private sealed class ScriptedRewriteChatClient : ChatClientDouble
    {
        protected override string Respond(int callIndex) =>
            "```python\n" + CannedProgram(callIndex + 1) + "```";
    }

    /// <summary>Answers with prose that carries no fenced block, so no proposal can ever be used.</summary>
    private sealed class ProseOnlyChatClient : ChatClientDouble
    {
        protected override string Respond(int callIndex) =>
            "I would suggest accumulating more, but I will not write the code out.";
    }

    /// <summary>Answers with a valid but oversized program.</summary>
    private sealed class GiantProgramChatClient : ChatClientDouble
    {
        protected override string Respond(int callIndex) =>
            "```python\n" + CannedProgram(200) + "```";
    }

    private abstract class ChatClientDouble : IChatClient<double>
    {
        private int _calls;

        public string ModelId => "scripted";

        public int Calls => _calls;

        public Task<ChatResponse> GetResponseAsync(
            IReadOnlyList<ChatMessage> messages,
            ChatOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int index = _calls;
            _calls++;
            return Task.FromResult(new ChatResponse(
                ChatMessage.Assistant(Respond(index)), usage: new ChatUsage(10, 5)));
        }

        public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
            IReadOnlyList<ChatMessage> messages,
            ChatOptions? options = null,
            CancellationToken cancellationToken = default) =>
            throw new NotSupportedException("The scripted chat client does not stream.");

        protected abstract string Respond(int callIndex);
    }
}
