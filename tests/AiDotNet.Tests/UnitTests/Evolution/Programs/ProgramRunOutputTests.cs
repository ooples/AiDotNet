using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Evolution.Programs.Outputs;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class ProgramRunOutputTests
{
    private static EvolutionArchiveEntry<ProgramGenome> Entry(
        string source,
        double quality,
        ProgramLanguage language = ProgramLanguage.Python,
        long evaluationId = 7,
        EvolutionOptimizationDirection direction = EvolutionOptimizationDirection.Maximize)
    {
        var genome = new ProgramGenome(source, language, "widened the search window");
        var canonical = new EvolutionCanonicalGenome<ProgramGenome>(genome, genome.Id);
        var lineage = new EvolutionLineage(
            new[] { "parent-a", "parent-b" },
            new[] { "inspiration-a" },
            "llm-variation",
            "diff-refiner",
            generation: 4,
            island: 2,
            seedStream: 12345UL);
        var candidate = new EvolutionCandidate<ProgramGenome>(evaluationId, canonical, lineage);
        var evaluation = new EvolutionEvaluation(
            evaluationId,
            genome.Id,
            EvolutionEvaluationStatus.Completed,
            quality,
            direction,
            new Dictionary<string, double>(StringComparer.Ordinal) { ["length"] = 42.0, ["tokens"] = 13.0 },
            new[] { 0.9, 0.1 },
            new[] { 0.0 },
            new EvolutionEvaluationCost(TimeSpan.FromMilliseconds(250), 2, 3.5),
            lineage,
            EvolutionCacheStatus.Miss,
            Array.Empty<EvolutionDiagnostic>(),
            "task-hash",
            "evaluator-hash",
            "configuration-hash");
        return new EvolutionArchiveEntry<ProgramGenome>(new EvolutionCellKey(new[] { 2, 5 }), candidate, evaluation);
    }

    [Fact]
    public void TheFinalWriteLandsInTheBestDirectoryWithTheLanguageExtension()
    {
        using var directory = new TemporaryDirectory();
        var writer = new ProgramRunOutputWriter(directory.Path);

        ProgramRunOutputRecord record = writer.WriteFinal(Entry("def solve():\n    return 1\n", 0.8), "EvaluationBudgetReached");

        Assert.Equal(Path.Combine(directory.Path, "best", "best_program.py"), record.ProgramPath);
        Assert.Equal(Path.Combine(directory.Path, "best", "best_program_info.json"), record.InfoPath);
        Assert.True(File.Exists(record.ProgramPath));
        Assert.True(File.Exists(record.InfoPath));
        Assert.Equal("def solve():\n    return 1\n", File.ReadAllText(record.ProgramPath));
    }

    [Theory]
    [InlineData(ProgramLanguage.Python, ".py")]
    [InlineData(ProgramLanguage.CSharp, ".cs")]
    [InlineData(ProgramLanguage.Rust, ".rs")]
    [InlineData(ProgramLanguage.Generic, ".txt")]
    public void TheProgramFileTakesTheExtensionOfItsLanguage(ProgramLanguage language, string extension)
    {
        using var directory = new TemporaryDirectory();
        var writer = new ProgramRunOutputWriter(directory.Path);

        ProgramRunOutputRecord record = writer.WriteFinal(Entry("value = 1\n", 0.5, language));

        Assert.Equal("best_program" + extension, Path.GetFileName(record.ProgramPath));
    }

    [Fact]
    public void ACheckpointWriteLandsInANumberedCheckpointDirectory()
    {
        using var directory = new TemporaryDirectory();
        var writer = new ProgramRunOutputWriter(directory.Path);

        ProgramRunOutputRecord record = writer.WriteCheckpoint(Entry("x = 1\n", 0.4), 3, "checkpoint 3");

        Assert.Equal(
            Path.Combine(directory.Path, "checkpoints", "checkpoint_3", "best_program.py"),
            record.ProgramPath);
        Assert.Equal(ProgramRunOutputTrigger.Checkpoint, record.Trigger);
        Assert.Equal(3, record.Ordinal);
    }

    [Fact]
    public void TheInfoDocumentCarriesMetricsDescriptorsCellLineageAndConfigurationHashes()
    {
        using var directory = new TemporaryDirectory();
        var writer = new ProgramRunOutputWriter(directory.Path, new ProgramRunOutputOptions { RunId = "run-42" });

        ProgramRunOutputRecord record = writer.WriteFinal(Entry("def solve():\n    return 1\n", 0.8), "TimeLimitReached");
        JObject info = JObject.Parse(File.ReadAllText(record.InfoPath));

        Assert.Equal("run-42", (string?)info["RunId"]);
        Assert.Equal("RunEnd", (string?)info["Trigger"]);
        Assert.Equal("TimeLimitReached", (string?)info["Note"]);
        Assert.Equal(record.GenomeId, (string?)info["GenomeId"]);
        Assert.Equal("Python", (string?)info["Language"]);
        Assert.Equal("best_program.py", (string?)info["ProgramFileName"]);
        Assert.Equal(0.8, (double?)info["Quality"]);
        Assert.Equal("Maximize", (string?)info["Direction"]);
        Assert.Equal("Completed", (string?)info["Status"]);
        Assert.Equal("Miss", (string?)info["CacheStatus"]);

        Assert.Equal(new[] { 2, 5 }, info["Cell"]?.ToObject<int[]>());
        Assert.Equal("2,5", (string?)info["CellKey"]);
        Assert.Equal(42.0, (double?)info["Descriptors"]?["length"]);
        Assert.Equal(13.0, (double?)info["Descriptors"]?["tokens"]);
        Assert.Equal(new[] { 0.9, 0.1 }, info["Objectives"]?.ToObject<double[]>());
        Assert.Equal(new[] { 0.0 }, info["ConstraintViolations"]?.ToObject<double[]>());

        Assert.Equal(4L, (long?)info["Generation"]);
        Assert.Equal(2, (int?)info["Island"]);
        Assert.Equal(new[] { "parent-a", "parent-b" }, info["ParentIds"]?.ToObject<string[]>());
        Assert.Equal(new[] { "inspiration-a" }, info["InspirationIds"]?.ToObject<string[]>());
        Assert.Equal("llm-variation", (string?)info["VariationOperatorId"]);
        Assert.Equal("diff-refiner", (string?)info["RefinerId"]);

        Assert.Equal(2, (int?)info["AttemptCount"]);
        Assert.Equal(3.5, (double?)info["CostUnits"]);
        Assert.Equal("task-hash", (string?)info["TaskVersionHash"]);
        Assert.Equal("evaluator-hash", (string?)info["EvaluatorVersionHash"]);
        Assert.Equal("configuration-hash", (string?)info["ConfigurationHash"]);
        Assert.NotNull(info["SavedAtUtc"]);
        Assert.NotNull((string?)info["SourceSha256"]);
    }

    [Fact]
    public void RewritingTheSameDirectoryReplacesTheFilesAndLeavesNoTemporaryFiles()
    {
        using var directory = new TemporaryDirectory();
        var writer = new ProgramRunOutputWriter(directory.Path);
        writer.WriteFinal(Entry("first = 1\n", 0.4));
        ProgramRunOutputRecord second = writer.WriteFinal(Entry("second = 2\n", 0.9));

        Assert.Equal("second = 2\n", File.ReadAllText(second.ProgramPath));
        Assert.Empty(Directory.EnumerateFiles(Path.Combine(directory.Path, "best"), "*.tmp"));
        Assert.Equal(2, Directory.EnumerateFiles(Path.Combine(directory.Path, "best")).Count());
    }

    [Fact]
    public void AnOversizedProgramIsCutAndTheInfoDocumentSaysSo()
    {
        using var directory = new TemporaryDirectory();
        var writer = new ProgramRunOutputWriter(directory.Path, new ProgramRunOutputOptions { MaxSourceBytes = 64 });

        ProgramRunOutputRecord record = writer.WriteFinal(Entry(new string('a', 5_000) + "\n", 0.3));
        JObject info = JObject.Parse(File.ReadAllText(record.InfoPath));

        Assert.True(record.IsSourceTruncated);
        Assert.Equal(64, new FileInfo(record.ProgramPath).Length);
        Assert.True((bool?)info["IsSourceTruncated"]);
        Assert.Equal(5_001, (int?)info["SourceLength"]);
    }

    [Fact]
    public void InvalidOutputNamesAreRejectedAtConstructionTime()
    {
        using var directory = new TemporaryDirectory();

        Assert.Throws<ArgumentException>(() => new ProgramRunOutputWriter(
            directory.Path, new ProgramRunOutputOptions { BestDirectoryName = "../escape" }));
        Assert.Throws<ArgumentException>(() => new ProgramRunOutputWriter(
            directory.Path, new ProgramRunOutputOptions { InfoFileName = " " }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ProgramRunOutputWriter(
            directory.Path, new ProgramRunOutputOptions { MaxSourceBytes = 0 }));
    }

    [Fact]
    public async Task TheObserverWritesAtEveryCheckpointAndOnceAtRunEnd()
    {
        using var directory = new TemporaryDirectory();
        var archive = new FakeArchiveView(Entry("def solve():\n    return 1\n", 0.8));
        var observer = new ProgramRunOutputObserver(new ProgramRunOutputWriter(directory.Path));
        observer.AddArchive(archive);

        await observer.OnEventAsync(new EvolutionEvent<ProgramGenome>(EvolutionEventKind.Proposed, 0));
        await observer.OnEventAsync(new EvolutionEvent<ProgramGenome>(EvolutionEventKind.Checkpointed, 1, message: "checkpoint 1"));
        await observer.OnEventAsync(new EvolutionEvent<ProgramGenome>(EvolutionEventKind.Checkpointed, 2, message: "checkpoint 2"));
        await observer.OnEventAsync(new EvolutionEvent<ProgramGenome>(EvolutionEventKind.Stopped, 3, message: "EvaluationBudgetReached"));

        Assert.Equal(3, observer.Records.Count);
        Assert.True(File.Exists(Path.Combine(directory.Path, "checkpoints", "checkpoint_1", "best_program.py")));
        Assert.True(File.Exists(Path.Combine(directory.Path, "checkpoints", "checkpoint_2", "best_program.py")));
        Assert.True(File.Exists(Path.Combine(directory.Path, "best", "best_program.py")));
        Assert.Null(observer.LastError);
        Assert.Equal(ProgramRunOutputTrigger.RunEnd, observer.LastRecord?.Trigger);
    }

    [Fact]
    public async Task TheObserverCanBeToldToSkipCheckpointSnapshots()
    {
        using var directory = new TemporaryDirectory();
        var writer = new ProgramRunOutputWriter(directory.Path, new ProgramRunOutputOptions { WriteAtCheckpoints = false });
        var observer = new ProgramRunOutputObserver(writer, new[] { (IEvolutionArchiveView<ProgramGenome>)new FakeArchiveView(Entry("x = 1\n", 0.2)) });

        await observer.OnEventAsync(new EvolutionEvent<ProgramGenome>(EvolutionEventKind.Checkpointed, 1, message: "checkpoint 1"));
        await observer.OnEventAsync(new EvolutionEvent<ProgramGenome>(EvolutionEventKind.Stopped, 2, message: "NoCandidates"));

        Assert.Single(observer.Records);
        Assert.False(Directory.Exists(Path.Combine(directory.Path, "checkpoints")));
    }

    [Fact]
    public async Task TheObserverWritesNothingWhenNoArchiveHoldsAProgram()
    {
        using var directory = new TemporaryDirectory();
        var observer = new ProgramRunOutputObserver(new ProgramRunOutputWriter(directory.Path));
        observer.AddArchive(new FakeArchiveView(null));

        await observer.OnEventAsync(new EvolutionEvent<ProgramGenome>(EvolutionEventKind.Stopped, 1, message: "NoCandidates"));

        Assert.Empty(observer.Records);
        Assert.Null(observer.LastError);
        Assert.False(Directory.Exists(Path.Combine(directory.Path, "best")));
    }

    [Fact]
    public void TheObserverPicksTheBestAcrossIslandsInTheArchiveDirection()
    {
        using var directory = new TemporaryDirectory();
        var observer = new ProgramRunOutputObserver(new ProgramRunOutputWriter(directory.Path));
        observer.AddArchive(new FakeArchiveView(Entry("weak = 1\n", 0.2)));
        observer.AddArchive(new FakeArchiveView(Entry("strong = 1\n", 0.9)));

        EvolutionArchiveEntry<ProgramGenome>? best = observer.SelectBest();

        Assert.NotNull(best);
        Assert.Equal("strong = 1\n", best.Candidate.CanonicalGenome.Genome.Source);
    }

    [Fact]
    public void TheObserverMinimizesWhenTheArchiveMinimizes()
    {
        using var directory = new TemporaryDirectory();
        var observer = new ProgramRunOutputObserver(new ProgramRunOutputWriter(directory.Path));
        observer.AddArchive(new FakeArchiveView(
            Entry("high = 1\n", 0.9, direction: EvolutionOptimizationDirection.Minimize),
            EvolutionOptimizationDirection.Minimize));
        observer.AddArchive(new FakeArchiveView(
            Entry("low = 1\n", 0.2, direction: EvolutionOptimizationDirection.Minimize),
            EvolutionOptimizationDirection.Minimize));

        EvolutionArchiveEntry<ProgramGenome>? best = observer.SelectBest();

        Assert.NotNull(best);
        Assert.Equal("low = 1\n", best.Candidate.CanonicalGenome.Genome.Source);
    }

    [Fact]
    public void AManualWriteGoesToTheBestDirectoryAndIsRecorded()
    {
        using var directory = new TemporaryDirectory();
        var observer = new ProgramRunOutputObserver(new ProgramRunOutputWriter(directory.Path));
        observer.AddArchive(new FakeArchiveView(Entry("x = 1\n", 0.5)));

        ProgramRunOutputRecord? record = observer.WriteNow("on demand");

        Assert.NotNull(record);
        Assert.Equal(ProgramRunOutputTrigger.Manual, record.Trigger);
        Assert.True(File.Exists(record.ProgramPath));
        Assert.Single(observer.Records);
    }

    private sealed class FakeArchiveView : IEvolutionArchiveView<ProgramGenome>
    {
        private readonly EvolutionArchiveEntry<ProgramGenome>? _best;

        public FakeArchiveView(
            EvolutionArchiveEntry<ProgramGenome>? best,
            EvolutionOptimizationDirection direction = EvolutionOptimizationDirection.Maximize)
        {
            _best = best;
            Direction = direction;
        }

        public IReadOnlyList<EvolutionDescriptorDefinition> Descriptors => Array.Empty<EvolutionDescriptorDefinition>();

        public string DefinitionHash => "fake-archive";

        public EvolutionOptimizationDirection Direction { get; }

        public int Count => _best is null ? 0 : 1;

        public long Version => 1;

        public IReadOnlyList<EvolutionArchiveEntry<ProgramGenome>> Entries =>
            _best is null
                ? Array.Empty<EvolutionArchiveEntry<ProgramGenome>>()
                : new[] { _best };

        public EvolutionArchiveEntry<ProgramGenome>? Best => _best;

        public EvolutionArchiveEntry<ProgramGenome>? Get(EvolutionCellKey cell) =>
            _best is not null && _best.Cell.Equals(cell) ? _best : null;
    }
}
