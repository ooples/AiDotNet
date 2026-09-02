using System.Globalization;
using AiDotNet;
using AiDotNet.Agentic.Models;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.Models.Results;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Interfaces;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNetTests.UnitTests.Evolution;
using Newtonsoft.Json;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Evolution;

/// <summary>
/// Proves a user can run both kinds of evolution through <see cref="AiModelBuilder{T,TInput,TOutput}"/> alone and
/// read every result through <see cref="AiModelResult{T,TInput,TOutput}"/>, with no network, no sandboxed process,
/// and no direct use of the engine types.
/// </summary>
public sealed class EvolutionFacadeIntegrationTests
{
    private const int EvaluationBudget = 12;

    [Fact(Timeout = 120000)]
    public async Task TypedGenomeEvolutionRunsEndToEndThroughTheBuilderAlone()
    {
        AiModelResult<double, Matrix<double>, Vector<double>> result = await CreateTypedBuilder()
            .BuildAsync();

        EvolutionRunSummary summary = RequireSummary(result);
        Assert.Equal("facade-typed", summary.RunId);
        Assert.Equal(EvolutionStopReason.EvaluationBudgetReached, summary.StopReason);
        Assert.Equal(EvaluationBudget, summary.EvaluationAttempts);

        // Completions can exceed attempts: the evaluation cache returns a finished result for a candidate that has
        // already been scored, which commits an evaluation without spending any of the attempt budget. The
        // attempt count is the one bounded by MaxEvaluationAttempts, and that is what stopped this run.
        Assert.True(
            summary.CompletedEvaluations >= summary.EvaluationAttempts,
            summary.CompletedEvaluations + " completions should cover the " + summary.EvaluationAttempts +
            " attempts");
        Assert.NotEmpty(summary.StateHash);
        Assert.NotEmpty(summary.CompatibilityHash);
        Assert.Equal(EvolutionOptimizationDirection.Maximize, summary.Direction);

        // The archive really illuminated: filled cells, each carrying its descriptor value and its bin.
        Assert.True(summary.ArchiveCount >= 1, "expected at least one filled cell, got " + summary.ArchiveCount);
        Assert.NotEmpty(summary.Elites);
        Assert.All(summary.Elites, elite => Assert.Contains("x", elite.Descriptors.Keys));
        Assert.All(summary.Elites, elite => Assert.Single(elite.Cell));
        for (int index = 1; index < summary.Elites.Count; index++)
        {
            Assert.True(summary.Elites[index - 1].Quality >= summary.Elites[index].Quality);
        }

        // Two islands were configured, so both are reported, both describe the same ten-cell grid, and the
        // per-island elite counts add up to the archive coverage the summary reports.
        Assert.Equal(2, summary.IslandCount);
        Assert.Equal(2, summary.Islands.Count);
        Assert.Equal(new[] { 0, 1 }, summary.Islands.Select(island => island.Island).ToArray());
        Assert.All(summary.Islands, island => Assert.Equal(10, island.TotalCells));
        Assert.Equal(summary.ArchiveCount, summary.Islands.Sum(island => island.EliteCount));
        Assert.Contains(summary.Islands, island => island.EliteCount > 0);

        // The seed was 30, the variation operator only ever increments, so the winner must beat the seed.
        Assert.True(summary.BestQuality > 30, "best quality was " + summary.BestQuality);
        Assert.NotNull(summary.BestGenomeId);
        Assert.Equal(summary.BestQuality, summary.Elites[0].Quality);

        // The typed result is reachable for the genome type the run used, and only for that type.
        EvolutionRunResult<TestGenome> typed =
            Assert.IsType<EvolutionRunResult<TestGenome>>(result.GetEvolutionRunResult<TestGenome>());
        Assert.Equal(summary.StateHash, typed.StateHash);
        Assert.Equal(summary.BestQuality, typed.Best?.Evaluation.Quality);
        Assert.Null(result.GetEvolutionRunResult<ProgramGenome>());

        // Nothing was written, because nothing asked for a checkpoint or a trace.
        Assert.Null(summary.OutputDirectory);
        Assert.Null(summary.CheckpointPath);
        Assert.Null(summary.TracePath);
    }

    [Fact(Timeout = 120000)]
    public async Task TwoIdenticallyConfiguredRunsAgreeOnEveryHashAndEliteOrder()
    {
        AiModelResult<double, Matrix<double>, Vector<double>> first =
            await CreateTypedBuilder().BuildAsync();
        AiModelResult<double, Matrix<double>, Vector<double>> second =
            await CreateTypedBuilder().BuildAsync();

        EvolutionRunSummary firstSummary = RequireSummary(first);
        EvolutionRunSummary secondSummary = RequireSummary(second);
        Assert.Equal(firstSummary.StateHash, secondSummary.StateHash);
        Assert.Equal(firstSummary.CompatibilityHash, secondSummary.CompatibilityHash);
        Assert.Equal(
            firstSummary.Elites.Select(elite => elite.GenomeId).ToArray(),
            secondSummary.Elites.Select(elite => elite.GenomeId).ToArray());

        AiModelResult<double, Matrix<double>, Vector<double>> different =
            await CreateTypedBuilder(seed: 99UL).BuildAsync();
        Assert.NotEqual(firstSummary.StateHash, RequireSummary(different).StateHash);
    }

    [Fact(Timeout = 120000)]
    public async Task ProgramEvolutionRunsEndToEndWithAFakeModelAndAFakeSandbox()
    {
        var client = new LadderChatClient();
        var engine = new AdditionProgramExecutionEngine();

        AiModelResult<double, Matrix<double>, Vector<double>> result = await CreateProgramBuilder(client, engine)
            .BuildAsync();

        ProgramEvolutionResult program = Assert.IsType<ProgramEvolutionResult>(result.ProgramEvolution);
        Assert.True(program.HasBestProgram);
        ProgramGenome best = Assert.IsType<ProgramGenome>(program.BestProgram);
        Assert.Equal(ProgramLanguage.Python, best.Language);

        // The scored ladder reaches the exact answer, so the winner solves every example.
        Assert.Equal(1.0, Assert.IsType<double>(program.BestQuality));
        Assert.Contains("+ 2", best.Source);
        Assert.True(client.Calls > 0, "the run must have asked the model for edits");
        Assert.True(engine.Calls > 0, "the run must have executed candidates");
        Assert.True(program.LlmUsage.ChatCalls > 0);

        // The same run is also described by the genome-agnostic summary, and the two agree.
        EvolutionRunSummary summary = RequireSummary(result);
        Assert.Equal("facade-program", summary.RunId);
        Assert.Equal(program.StateHash, summary.StateHash);
        Assert.Equal(program.BestQuality, summary.BestQuality);
        Assert.Equal(program.ArchiveCount, summary.ArchiveCount);
        Assert.Equal(
            program.LlmUsage.ChatCalls,
            Assert.IsType<ProgramEvolutionLlmUsage>(summary.LlmUsage).ChatCalls);

        // No behaviour axis was configured, so the builder supplied the documented program-length axis.
        Assert.All(summary.Elites, elite => Assert.Contains("length", elite.Descriptors.Keys));

        EvolutionRunResult<ProgramGenome> typed =
            Assert.IsType<EvolutionRunResult<ProgramGenome>>(result.GetEvolutionRunResult<ProgramGenome>());
        Assert.Equal(summary.StateHash, typed.StateHash);
    }

    [Fact(Timeout = 120000)]
    public async Task TwoIdenticalProgramRunsProduceTheSameStateHash()
    {
        AiModelResult<double, Matrix<double>, Vector<double>> first =
            await CreateProgramBuilder(new LadderChatClient(), new AdditionProgramExecutionEngine())
                .BuildAsync();
        AiModelResult<double, Matrix<double>, Vector<double>> second =
            await CreateProgramBuilder(new LadderChatClient(), new AdditionProgramExecutionEngine())
                .BuildAsync();

        Assert.Equal(RequireSummary(first).StateHash, RequireSummary(second).StateHash);
        ProgramEvolutionResult firstProgram = Assert.IsType<ProgramEvolutionResult>(first.ProgramEvolution);
        ProgramEvolutionResult secondProgram = Assert.IsType<ProgramEvolutionResult>(second.ProgramEvolution);
        Assert.Equal(
            Assert.IsType<ProgramGenome>(firstProgram.BestProgram).Id,
            Assert.IsType<ProgramGenome>(secondProgram.BestProgram).Id);
    }

    [Fact(Timeout = 120000)]
    public async Task AGenomeOnlyResultRefusesToPredictInsteadOfNullReferencing()
    {
        AiModelResult<double, Matrix<double>, Vector<double>> result = await CreateTypedBuilder()
            .BuildAsync();

        Assert.True(result.IsGenomeOnlyResult);
        Assert.Null(result.Model);

        var exception = Assert.Throws<InvalidOperationException>(() => result.Predict(new Matrix<double>(1, 1)));
        Assert.Contains("Model", exception.Message, StringComparison.Ordinal);
    }

    [Fact(Timeout = 120000)]
    public async Task TheRunSummarySurvivesANewtonsoftRoundTrip()
    {
        AiModelResult<double, Matrix<double>, Vector<double>> result = await CreateTypedBuilder()
            .BuildAsync();
        EvolutionRunSummary original = RequireSummary(result);

        string json = JsonConvert.SerializeObject(original);
        EvolutionRunSummary restored = Assert.IsType<EvolutionRunSummary>(
            JsonConvert.DeserializeObject<EvolutionRunSummary>(json));

        Assert.Equal(original.RunId, restored.RunId);
        Assert.Equal(original.StopReason, restored.StopReason);
        Assert.Equal(original.StateHash, restored.StateHash);
        Assert.Equal(original.CompatibilityHash, restored.CompatibilityHash);
        Assert.Equal(original.BestQuality, restored.BestQuality);
        Assert.Equal(original.BestGenomeId, restored.BestGenomeId);
        Assert.Equal(original.ArchiveCount, restored.ArchiveCount);
        Assert.Equal(original.Elites.Count, restored.Elites.Count);
        Assert.Equal(original.Elites[0].GenomeId, restored.Elites[0].GenomeId);
        Assert.Equal(original.Elites[0].Cell, restored.Elites[0].Cell);
        Assert.Equal(original.Elites[0].Descriptors["x"], restored.Elites[0].Descriptors["x"]);
        Assert.Equal(original.Islands.Count, restored.Islands.Count);
        Assert.Equal(original.StatusCounts.Count, restored.StatusCounts.Count);
        foreach (KeyValuePair<string, long> pair in original.StatusCounts)
        {
            Assert.Equal(pair.Value, restored.StatusCounts[pair.Key]);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task CheckpointingAndTracingWriteUnderTheConfiguredOutputDirectory()
    {
        string root = Path.Combine(Path.GetTempPath(), "aidotnet-facade-tests", Guid.NewGuid().ToString("N"));
        try
        {
            EvolutionOptions options = CreateTypedOptions();
            options.RunId = "facade-persisted";
            options.OutputDirectory = root;
            options.CheckpointInterval = 4;
            options.Trace.Enabled = true;

            AiModelResult<double, Matrix<double>, Vector<double>> result =
                await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                    .ConfigureEvolution(
                        new SyntheticEvolutionTask(),
                        new IncrementVariation(),
                        new TestGenomeCodec(),
                        options)
                    .ConfigureEvolutionSeeds(new[] { new TestGenome(30) })
                    .BuildAsync();

            EvolutionRunSummary summary = RequireSummary(result);
            Assert.Equal(Path.GetFullPath(root), summary.OutputDirectory);

            string checkpointPath = Assert.IsType<string>(summary.CheckpointPath);
            Assert.True(File.Exists(checkpointPath), "expected a checkpoint at " + checkpointPath);
            string tracePath = Assert.IsType<string>(summary.TracePath);
            Assert.True(File.Exists(tracePath), "expected a trace at " + tracePath);
            Assert.True(summary.TraceRecordCount > 0);
            Assert.StartsWith(Path.GetFullPath(root), checkpointPath, StringComparison.OrdinalIgnoreCase);
            Assert.StartsWith(Path.GetFullPath(root), tracePath, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void CheckpointingWithoutAGenomeCodecIsRefusedAtConfigureTime()
    {
        EvolutionOptions options = CreateTypedOptions();
        options.CheckpointInterval = 4;

        var exception = Assert.Throws<ArgumentException>(() =>
            new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureEvolution(new SyntheticEvolutionTask(), new IncrementVariation(), options));

        Assert.Contains("IEvolutionGenomeCodec", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnArchiveWithNoBehaviourAxisIsRefusedAtConfigureTime()
    {
        var exception = Assert.Throws<ArgumentException>(() =>
            new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureEvolution(new SyntheticEvolutionTask(), new IncrementVariation(), new EvolutionOptions()));

        Assert.Contains("behaviour axis", exception.Message, StringComparison.Ordinal);
    }

    [Fact(Timeout = 120000)]
    public async Task NoveltySearchIsReachableThroughTheBuilderAlone()
    {
        // The last engine constructor slot the facade did not surface was the structural distance metric, which
        // left EvolutionOptions.NoveltyDistanceThreshold a public knob that could only ever throw. This proves the
        // whole novelty gate now runs from the builder: the metric is consulted, and it changes what gets archived.
        var metric = new CountingGenomeDistance();
        EvolutionOptions options = CreateTypedOptions();
        options.NoveltyDistanceThreshold = 6;

        AiModelResult<double, Matrix<double>, Vector<double>> result =
            await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureEvolution(
                    new SyntheticEvolutionTask(), new IncrementVariation(), options,
                    selection: null, refiner: null, migration: null, observer: null, genomeDistance: metric)
                .ConfigureEvolutionSeeds(new[] { new TestGenome(30) })
                .BuildAsync();

        EvolutionRunSummary novel = RequireSummary(result);
        Assert.True(metric.Calls > 0, "the distance metric was never consulted");

        // Same run, same seed, no novelty gate: the gate can only ever reject candidates, so it cannot archive more.
        EvolutionRunSummary plain = RequireSummary(await CreateTypedBuilder().BuildAsync());
        Assert.True(
            novel.ArchiveCount <= plain.ArchiveCount,
            "novelty archived " + novel.ArchiveCount + " cells, ungated archived " + plain.ArchiveCount);
    }

    [Fact(Timeout = 120000)]
    public async Task NoveltySearchWithoutADistanceMetricSaysSo()
    {
        EvolutionOptions options = CreateTypedOptions();
        options.NoveltyDistanceThreshold = 6;

        IAiModelBuilder<double, Matrix<double>, Vector<double>> builder =
            new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureEvolution(new SyntheticEvolutionTask(), new IncrementVariation(), options)
                .ConfigureEvolutionSeeds(new[] { new TestGenome(30) });

        ArgumentException exception = await Assert.ThrowsAsync<ArgumentException>(() => builder.BuildAsync());
        Assert.Contains("genome distance metric", exception.Message, StringComparison.Ordinal);
    }

    [Fact(Timeout = 120000)]
    public async Task SeedsOfTheWrongGenomeTypeAreReportedClearly()

    {
        IAiModelBuilder<double, Matrix<double>, Vector<double>> builder =
            new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureEvolution(new SyntheticEvolutionTask(), new IncrementVariation(), CreateTypedOptions())
                .ConfigureEvolutionSeeds(new[] { "not a genome" });

        InvalidOperationException exception =
            await Assert.ThrowsAsync<InvalidOperationException>(() => builder.BuildAsync());
        Assert.Contains("TestGenome", exception.Message, StringComparison.Ordinal);
    }

    [Fact(Timeout = 120000)]
    public async Task ProgramEvolutionWithoutAChatClientSaysSo()
    {
        var options = new ProgramEvolutionOptions { Language = ProgramLanguage.Python };
        options.SeedPrograms.Add(Program(0));

        IAiModelBuilder<double, Matrix<double>, Vector<double>> builder =
            new AiModelBuilder<double, Matrix<double>, Vector<double>>().ConfigureProgramEvolution(options);

        InvalidOperationException exception =
            await Assert.ThrowsAsync<InvalidOperationException>(() => builder.BuildAsync());
        Assert.Contains("ConfigureChatClient", exception.Message, StringComparison.Ordinal);
    }

    [Fact(Timeout = 120000)]
    public async Task ProgramEvolutionWithNoWayToScoreACandidateSaysSo()
    {
        var options = new ProgramEvolutionOptions { Language = ProgramLanguage.Python };
        options.SeedPrograms.Add(Program(0));

        IAiModelBuilder<double, Matrix<double>, Vector<double>> builder =
            new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureChatClient(new LadderChatClient())
                .ConfigureProgramExecutionEngine(new AdditionProgramExecutionEngine())
                .ConfigureProgramEvolution(options);

        InvalidOperationException exception =
            await Assert.ThrowsAsync<InvalidOperationException>(() => builder.BuildAsync());
        Assert.Contains("TestCases", exception.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheSelectionPolicySetThroughProgramEvolutionSurvivesTheDefensiveCopy()
    {
        // ConfigureProgramEvolution stores a Clone(), and that clone copies EvolutionEngineOptions field by field.
        // SelectionPolicy is new, so it has to be added there too or a program run silently ignores it — the same
        // class of defect as an option dropped by a deserialize constructor.
        var options = new ProgramEvolutionOptions { Language = ProgramLanguage.Python };
        options.SeedPrograms.Add(Program(0));
        options.Engine.SelectionPolicy = EvolutionSelectionPolicyKind.Ratio;

        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        builder.ConfigureProgramEvolution(options);

        var view = (IConfiguredView<double, Matrix<double>, Vector<double>>)builder;
        ProgramEvolutionOptions stored = Assert.IsType<ProgramEvolutionOptions>(view.ConfiguredProgramEvolution);
        Assert.NotSame(options, stored);
        Assert.Equal(EvolutionSelectionPolicyKind.Ratio, stored.Engine.SelectionPolicy);
    }

    [Fact]
    public void EveryEvolutionSlotIsVisibleOnTheConfiguredView()
    {
        var sandbox = new ProgramSandboxOptions();
        var executionEngine = new AdditionProgramExecutionEngine();
        var chatClient = new LadderChatClient();
        var programOptions = new ProgramEvolutionOptions { Language = ProgramLanguage.Python };
        programOptions.SeedPrograms.Add(Program(0));
        var seedOptions = new EvolutionSeedOptions();
        seedOptions.ProgramSources.Add(Program(1));

        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        builder.ConfigureEvolution(CreateTypedOptions())
            .ConfigureEvolutionSeeds(seedOptions)
            .ConfigureProgramEvolution(programOptions)
            .ConfigureChatClient(chatClient, new ChatClientOptions { MaxRetries = 2 })
            .ConfigureProgramSandbox(sandbox)
            .ConfigureProgramExecutionEngine(executionEngine);

        var view = (IConfiguredView<double, Matrix<double>, Vector<double>>)builder;
        Assert.Equal("facade-typed", Assert.IsType<EvolutionOptions>(view.ConfiguredEvolution).RunId);
        Assert.Equal(
            Program(1),
            Assert.Single(Assert.IsType<EvolutionSeedOptions>(view.ConfiguredEvolutionSeeds).ProgramSources));
        Assert.Equal(
            ProgramLanguage.Python,
            Assert.IsType<ProgramEvolutionOptions>(view.ConfiguredProgramEvolution).Language);
        Assert.Same(chatClient, view.ConfiguredChatClient);
        Assert.Equal(2, Assert.IsType<ChatClientOptions>(view.ConfiguredChatClientOptions).MaxRetries);
        Assert.NotNull(view.ConfiguredProgramSandbox);
        Assert.Same(executionEngine, view.ConfiguredProgramExecutionEngine);
        Assert.False(view.HasConfiguredEvolutionRun);
        Assert.Equal(0, view.ConfiguredEvolutionSeedCount);

        builder.ConfigureEvolution(new SyntheticEvolutionTask(), new IncrementVariation(), CreateTypedOptions());
        builder.ConfigureEvolutionSeeds(new[] { new TestGenome(1), new TestGenome(2) });
        Assert.True(view.HasConfiguredEvolutionRun);
        Assert.Equal(2, view.ConfiguredEvolutionSeedCount);
    }

    [Fact]
    public void TheOpenEvolveDefaultsFactoryMirrorsTheEngineFactory()
    {
        EvolutionOptions facade = EvolutionOptions.CreateOpenEvolveDefaults();
        EvolutionEngineOptions engine = EvolutionEngineOptions.CreateOpenEvolveDefaults();

        Assert.Equal(engine.Seed, facade.Seed);
        Assert.Equal(engine.IslandCount, facade.IslandCount);
        Assert.Equal(engine.MigrationInterval, facade.MigrationInterval);
        Assert.Equal(engine.MigrationTrigger, facade.MigrationTrigger);
        Assert.Equal(engine.GlobalEliteCount, facade.GlobalEliteCount);
        Assert.Equal(engine.HistorySize, facade.HistorySize);
        Assert.Equal(engine.SelectionPolicy, facade.SelectionPolicy);
        Assert.Equal(engine.MaxRetries, facade.MaxRetries);
        Assert.Equal(engine.EvaluationTimeout, facade.EvaluationTimeout);
        Assert.True(facade.Artifacts.Enabled);

        // The class defaults are untouched by the factory.
        var plain = new EvolutionOptions();
        Assert.Equal(1, plain.IslandCount);
        Assert.Equal(0, plain.MaxRetries);
        Assert.Equal(EvolutionSelectionPolicyKind.Uniform, plain.SelectionPolicy);
        Assert.True(plain.RetainOutput);
        Assert.False(plain.Trace.Enabled);
    }

    [Fact(Timeout = 120000)]
    public async Task TheTokenGivenToBuildAsyncReachesTheEvaluator()
    {
        // The task cancels the source from inside its own evaluator and then honours the token. That can only end
        // the run if BuildAsync propagated the caller's token all the way into evaluation; had it passed
        // CancellationToken.None, the throw would have been recorded as one failed candidate and the run would have
        // finished normally.
        using var cancellation = new CancellationTokenSource();
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        builder.ConfigureEvolution(
                new SyntheticEvolutionTask(cancelOnEvaluation: cancellation),
                new IncrementVariation(),
                CreateTypedOptions())
            .ConfigureEvolutionSeeds(new[] { new TestGenome(30) });

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => builder.BuildAsync(cancellation.Token));
    }

    private static EvolutionRunSummary RequireSummary(AiModelResult<double, Matrix<double>, Vector<double>> result) =>
        Assert.IsType<EvolutionRunSummary>(result.EvolutionSummary);

    private static EvolutionOptions CreateTypedOptions(ulong seed = 4242UL)
    {
        var options = new EvolutionOptions
        {
            RunId = "facade-typed",
            Seed = seed,
            MaxEvaluationAttempts = EvaluationBudget,
            // The increment operator proposes the same child from the same parent, so most proposals are
            // duplicates that cost proposal budget and no evaluation. The proposal budget is therefore set well
            // above the evaluation budget, which is the limit this run is meant to stop on.
            MaxProposals = 400,
            MaxGenerations = 400,
            ProposalBatchSize = 4,
            IslandCount = 2,
            MigrationInterval = 2,
            MigrantsPerIsland = 1,
            InspirationCount = 2
        };

        options.Descriptors.Add(
            new EvolutionDescriptorDefinition("x", 0, 100, 10, EvolutionOutOfRangePolicy.Clamp));
        return options;
    }

    private static IAiModelBuilder<double, Matrix<double>, Vector<double>> CreateTypedBuilder(ulong seed = 4242UL) =>
        new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureEvolution(new SyntheticEvolutionTask(), new IncrementVariation(), CreateTypedOptions(seed))
            .ConfigureEvolutionSeeds(new[] { new TestGenome(30) });

    private static IAiModelBuilder<double, Matrix<double>, Vector<double>> CreateProgramBuilder(
        IChatClient<double> client,
        IProgramExecutionEngine engine)
    {
        var options = new ProgramEvolutionOptions
        {
            Language = ProgramLanguage.Python,
            TaskDescription = "Return the input plus two.",
            IncludeEliteSourceCount = 5,
            Variation = { Mode = ProgramEvolutionMode.FullRewrite, MaxInspirations = 2 }
        };
        options.SeedPrograms.Add(Program(0));
        options.TestCases.Add(new ProgramInputOutputExample { Input = "1", ExpectedOutput = "3" });
        options.TestCases.Add(new ProgramInputOutputExample { Input = "5", ExpectedOutput = "7" });
        options.Engine.RunId = "facade-program";
        options.Engine.Seed = 4242UL;
        options.Engine.MaxEvaluationAttempts = EvaluationBudget;
        options.Engine.MaxProposals = 40;
        options.Engine.MaxGenerations = 40;
        options.Engine.ProposalBatchSize = 4;
        options.Engine.InspirationCount = 2;

        return new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureChatClient(client)
            .ConfigureProgramExecutionEngine(engine)
            .ConfigureProgramEvolution(options);
    }

    /// <summary>Builds the candidate that adds <paramref name="addend"/> to its input.</summary>
    private static string Program(int addend) =>
        "def solve(x):\n    return x + " + addend.ToString(CultureInfo.InvariantCulture) + "\n";

    /// <summary>Answers with a ladder of candidate programs, so the correct one is reached without a network.</summary>
    private sealed class LadderChatClient : IChatClient<double>
    {
        private int _calls;

        public string ModelId => "ladder";

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
                ChatMessage.Assistant("```python\n" + Program(index + 1) + "```"),
                usage: new ChatUsage(12, 6)));
        }

        public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
            IReadOnlyList<ChatMessage> messages,
            ChatOptions? options = null,
            CancellationToken cancellationToken = default) =>
            throw new NotSupportedException("The ladder chat client does not stream.");
    }

    /// <summary>Interprets a candidate as "read a number, add the program's addend, print it" without any process.</summary>
    private sealed class AdditionProgramExecutionEngine : IProgramExecutionEngine
    {
        private int _calls;

        public int Calls => _calls;

        public bool TryExecute(
            ProgramLanguage language,
            string sourceCode,
            string input,
            out string output,
            out string? errorMessage,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Interlocked.Increment(ref _calls);
            if (!TryRun(sourceCode, input, out string produced))
            {
                output = string.Empty;
                errorMessage = "the candidate did not parse";
                return false;
            }

            output = produced;
            errorMessage = null;
            return true;
        }

        public Task<ProgramExecuteResponse> ExecuteAsync(
            ProgramExecuteRequest request,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Interlocked.Increment(ref _calls);
            bool parsed = TryRun(request.SourceCode, request.StdIn ?? string.Empty, out string produced);
            return Task.FromResult(new ProgramExecuteResponse
            {
                Success = parsed,
                Language = request.Language,
                ExitCode = parsed ? 0 : 1,
                StdOut = produced,
                StdErr = parsed ? string.Empty : "the candidate did not parse",
                Error = parsed ? null : "the candidate did not parse",
                ErrorCode = parsed ? null : ProgramExecuteErrorCode.ExecutionFailed
            });
        }

        private static bool TryRun(string sourceCode, string input, out string output)
        {
            output = string.Empty;
            int marker = sourceCode.LastIndexOf("+ ", StringComparison.Ordinal);
            if (marker < 0) return false;

            string addendText = sourceCode.Substring(marker + 2).Trim();
            if (!int.TryParse(addendText, NumberStyles.Integer, CultureInfo.InvariantCulture, out int addend))
            {
                return false;
            }

            if (!int.TryParse(input.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int value))
            {
                return false;
            }

            output = (value + addend).ToString(CultureInfo.InvariantCulture);
            return true;
        }
    }

    /// <summary>
    /// The absolute gap between two candidate values, counting how often the engine asked. Pure, symmetric and
    /// deterministic, as <see cref="IGenomeDistance{TGenome}"/> requires.
    /// </summary>
    private sealed class CountingGenomeDistance : IGenomeDistance<TestGenome>
    {
        public int Calls { get; private set; }

        public string Id => "test-absolute-gap";

        public string VersionHash => "test-absolute-gap-v1";

        public double Distance(TestGenome first, TestGenome second)
        {
            Calls++;
            return Math.Abs(first.Value - second.Value);
        }
    }
}
