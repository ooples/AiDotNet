using AiDotNet;
using AiDotNet.Configuration;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

public sealed class EvolutionRunControlTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "evolution-control-" + Guid.NewGuid().ToString("N"));

    [Fact]
    public void HandleAcceptsEarlyRequestsButCannotBeReboundOrReused()
    {
        var control = new EvolutionRunControl();
        int calls = 0;
        Assert.True(control.RequestStop());
        Assert.True(control.IsStopRequested);
        using (control.Attach(() => calls++))
        {
            Assert.Equal(1, calls);
            Assert.Throws<InvalidOperationException>(() => control.Attach(() => { }));
            Assert.True(control.RequestStop());
            Assert.Equal(2, calls);
        }
        Assert.True(control.IsFinished);
        Assert.False(control.RequestStop());
        Assert.Throws<InvalidOperationException>(() => control.Attach(() => { }));
    }

    [Fact]
    public async Task StopBeforeBuildReturnsAResultWithoutStartingEvaluation()
    {
        var control = new EvolutionRunControl();
        control.RequestStop();
        int calls = 0;
        var builder = Builder(control, () => calls++);
        var result = await builder.BuildAsync();
        Assert.Equal(0, calls);
        Assert.Equal(EvolutionStopReason.Canceled, result.EvolutionSummary!.StopReason);
        Assert.Equal(0, result.EvolutionSummary.CompletedEvaluations);
        Assert.True(control.IsFinished);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task GracefulStopCommitsTheBatchKeepsTracingAndCanResumeFromItsCheckpoint(bool observerThrows)
    {
        var control = new EvolutionRunControl();
        int calls = 0;
        var observer = new StopObserver(control, observerThrows);
        var options = new EvolutionOptions
        {
            RunId = "controlled", Seed = 7, OutputDirectory = _root, CheckpointInterval = 1,
            MaxEvaluationAttempts = 4, MaxProposals = 8, MaxGenerations = 4, ProposalBatchSize = 2
        };
        options.Trace.Enabled = true;
        var builder = Builder(control, () => calls++).ObserveProgramEvolution(observer);
        builder.ConfigureEvolution(options);
        var stopped = await builder.BuildAsync();
        Assert.Equal(2, calls);
        Assert.Equal(2, stopped.EvolutionSummary!.CompletedEvaluations);
        Assert.Equal(EvolutionStopReason.Canceled, stopped.EvolutionSummary.StopReason);
        Assert.NotNull(stopped.ProgramEvolution!.BestProgram);
        Assert.True(File.Exists(stopped.EvolutionSummary.CheckpointPath));
        Assert.Equal(2, stopped.EvolutionSummary.TraceRecordCount); // One terminal record per measured seed, not per engine event.
        Assert.True(control.IsFinished);
        Assert.False(control.RequestStop());

        var nextControl = new EvolutionRunControl();
        options.Resume = true;
        var resumedBuilder = Builder(nextControl, () => calls++);
        resumedBuilder.ConfigureEvolution(options);
        var resumed = await resumedBuilder.BuildAsync();
        Assert.Equal(4, calls); // The two saved seeds are not reacquired on resume.
        Assert.Equal(4, resumed.EvolutionSummary!.CompletedEvaluations);
        Assert.True(nextControl.IsFinished);
    }

    [Fact]
    public void NullHandlesAndObserversAreRejected()
    {
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        Assert.Throws<ArgumentNullException>(() => builder.WithEvolutionControl(null!));
        Assert.Throws<ArgumentNullException>(() => builder.ObserveProgramEvolution(null!));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task CheckpointOutputAndLiveArchivesAreAvailableBeforeRunEndUnderTheResolvedOutputRoot(bool writeAtRunEnd)
    {
        var observer = new ArchiveObserver(_root);
        var builder = Builder(new EvolutionRunControl(), () => { }, programs =>
        {
            programs.Engine.OutputDirectory = Path.Combine(_root, "superseded-root");
            programs.RunOutput = new ProgramRunOutputOptions { WriteAtRunEnd = writeAtRunEnd };
        }).ObserveProgramEvolution(observer);
        builder.ConfigureEvolution(new EvolutionOptions
        {
            RunId = "output", OutputDirectory = _root, CheckpointInterval = 1,
            MaxEvaluationAttempts = 2, ProposalBatchSize = 2, IslandCount = 2
        });
        var result = await builder.BuildAsync();
        Assert.Equal(2, observer.Archives.Count);
        Assert.True(observer.CheckpointSeen);
        Assert.True(observer.CheckpointHadWinner);
        Assert.True(observer.CheckpointOutputExisted);
        Assert.True(observer.EventsHadRegisteredArchives);
        Assert.False(Directory.Exists(Path.Combine(_root, "superseded-root")));
        Assert.Equal(writeAtRunEnd, File.Exists(Path.Combine(_root, "best", "best_program.cs")));
        Assert.DoesNotContain(result.EvolutionSummary!.RetainedFailures, item => item.Code == "program_output_incomplete");
    }

    [Fact]
    public async Task OutputFailureIsReportedWithoutPrivateFileSystemDiagnostics()
    {
        Directory.CreateDirectory(_root);
        File.WriteAllText(Path.Combine(_root, "best"), "authored obstruction");
        var builder = Builder(new EvolutionRunControl(), () => { }, programs =>
        {
            programs.Engine.OutputDirectory = _root;
            programs.RunOutput = new ProgramRunOutputOptions();
        });
        builder.ConfigureEvolution(new EvolutionOptions { OutputDirectory = _root, MaxEvaluationAttempts = 2 });
        var result = await builder.BuildAsync();
        var failure = Assert.Single(result.EvolutionSummary!.RetainedFailures, item => item.Code == "program_output_incomplete");
        Assert.True(failure.IsRedacted);
        Assert.DoesNotContain(_root, failure.Message);
        Assert.Equal("authored obstruction", File.ReadAllText(Path.Combine(_root, "best")));
    }

    private static AiModelBuilder<double, Matrix<double>, Vector<double>> Builder(
        EvolutionRunControl control, Action measured, Action<ProgramEvolutionOptions>? configure = null)
    {
        var programs = new ProgramEvolutionOptions
        {
            Language = ProgramLanguage.CSharp, CustomVariation = new Variation(),
            CustomFitnessEvaluator = new DelegateProgramFitnessEvaluator((_, _, _) =>
            {
                measured();
                return new ValueTask<EvolutionTaskResult>(new EvolutionTaskResult(EvolutionEvaluationStatus.Completed, 1, costUnits: 1));
            })
        };
        programs.SeedPrograms.Add("return 1;");
        programs.SeedPrograms.Add("return 2;");
        configure?.Invoke(programs);
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>().WithEvolutionControl(control);
        builder.ConfigureProgramEvolution(programs);
        return builder;
    }

    private sealed class ArchiveObserver(string root) : IProgramEvolutionArchiveObserver
    {
        public List<IEvolutionArchiveView<ProgramGenome>> Archives { get; } = new();
        public bool CheckpointSeen { get; private set; }
        public bool CheckpointHadWinner { get; private set; }
        public bool CheckpointOutputExisted { get; private set; }
        public bool EventsHadRegisteredArchives { get; private set; } = true;
        public void AddArchive(IEvolutionArchiveView<ProgramGenome> archive) => Archives.Add(archive);
        public ValueTask OnEventAsync(EvolutionEvent<ProgramGenome> item, CancellationToken cancellationToken = default)
        {
            EventsHadRegisteredArchives &= Archives.Count == 2;
            if (item.Kind == EvolutionEventKind.Checkpointed)
            {
                CheckpointSeen = true;
                CheckpointHadWinner |= Archives.Any(archive => archive.Best is not null);
                CheckpointOutputExisted |= File.Exists(Path.Combine(root, "checkpoints", "checkpoint_1", "best_program.cs"));
            }
            return default;
        }
    }

    private sealed class StopObserver : IEvolutionObserver<ProgramGenome>
    {
        private readonly EvolutionRunControl _control;
        private readonly bool _throws;
        public StopObserver(EvolutionRunControl control, bool throws) { _control = control; _throws = throws; }
        public ValueTask OnEventAsync(EvolutionEvent<ProgramGenome> item, CancellationToken cancellationToken = default)
        {
            if (item.Kind == EvolutionEventKind.Evaluated)
            {
                _control.RequestStop();
                if (_throws) throw new InvalidOperationException("authored observer failure");
            }
            return default;
        }
    }

    private sealed class Variation : IProgramVariationOperator
    {
        public string Id => "control-test";
        public string VersionHash => "control-test-v1";
        public ValueTask<ProgramGenome> ProposeAsync(EvolutionVariationContext<ProgramGenome> context, CancellationToken cancellationToken = default) =>
            new(new ProgramGenome("return " + (3 + context.Random.NextInt(65536)) + ";", ProgramLanguage.CSharp));
        public void Observe(EvolutionEvaluation evaluation, EvolutionArchiveInsertionResult? insertionResult) { }
        public ProgramEvolutionLlmUsage GetUsage() => new();
    }

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); // This instance's own GUID directory only.
    }
}
