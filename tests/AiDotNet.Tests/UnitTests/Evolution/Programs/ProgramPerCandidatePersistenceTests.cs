using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Evolution.Programs.Outputs;
using AiDotNet.ProgramSynthesis.Enums;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

/// <summary>
/// Covers per-candidate program persistence: one file per evaluated program, written as it commits. The archive
/// keeps one candidate per cell and discards the rest, so without this the programs that led to a winner exist
/// nowhere by the time a run ends, and no amount of reading the result afterwards recovers them.
/// </summary>
public sealed class ProgramPerCandidatePersistenceTests
{
    [Fact]
    public void EveryEvaluatedProgramIsWrittenAsItsOwnRecordWithSourceAndLineage()
    {
        using var directory = new TemporaryDirectory();
        var observer = Observer(directory.Path, new ProgramRunOutputOptions { WriteEveryProgram = true });

        observer.OnEventAsync(Evaluated("def solve(x):\n    return x + 1\n", 0.4, 11)).GetAwaiter().GetResult();
        observer.OnEventAsync(Evaluated("def solve(x):\n    return x + 2\n", 0.9, 12)).GetAwaiter().GetResult();

        string programs = Path.Combine(directory.Path, "programs");
        string[] files = Directory.GetFiles(programs, "*.json");
        Assert.Equal(2, files.Length);
        Assert.Equal(2, observer.ProgramsWritten);
        Assert.Null(observer.LastError);

        JObject document = JObject.Parse(File.ReadAllText(files.OrderBy(file => file, StringComparer.Ordinal).First()));
        Assert.StartsWith("def solve(x):", (string?)document["Source"], StringComparison.Ordinal);
        Assert.Equal("Python", (string?)document["Language"]);
        Assert.Equal("Completed", (string?)document["Status"]);
        Assert.Equal("llm-variation", (string?)document["VariationOperatorId"]);
        Assert.Equal("parent-a", (string?)document["ParentIds"]?[0]);
        Assert.Equal(42.0, (double?)document["Descriptors"]?["length"]);
        Assert.Equal(0.75, (double?)document["Metrics"]?["accuracy"]);

        // The cell is left null rather than guessed: the archives belong to the engine and are not visible while the
        // run is in progress, and the descriptors a cell is derived from are recorded either way.
        Assert.Equal(JTokenType.Null, document["CellKey"]?.Type);
    }

    [Fact]
    public void NothingIsWrittenUnlessTheRunAsksForIt()
    {
        using var directory = new TemporaryDirectory();
        var observer = Observer(directory.Path, new ProgramRunOutputOptions());

        observer.OnEventAsync(Evaluated("value = 1\n", 0.5, 3)).GetAwaiter().GetResult();

        Assert.False(Directory.Exists(Path.Combine(directory.Path, "programs")));
        Assert.Equal(0, observer.ProgramsWritten);
    }

    [Fact]
    public void EveryTerminalOutcomeIsRecorded_NotOnlyTheSuccesses()
    {
        using var directory = new TemporaryDirectory();
        var observer = Observer(directory.Path, new ProgramRunOutputOptions { WriteEveryProgram = true });

        observer.OnEventAsync(Evaluated("value = 1\n", null, 4, EvolutionEvaluationStatus.Failed)).GetAwaiter().GetResult();
        observer.OnEventAsync(Evaluated("value = 2\n", null, 5, EvolutionEvaluationStatus.Duplicate)).GetAwaiter().GetResult();

        // A candidate that failed is the more informative row for auditing a run and for training on it, and the
        // document carries the status and the diagnostics that say which outcome it was.
        Assert.Equal(2, observer.ProgramsWritten);

        string[] files = Directory.GetFiles(Path.Combine(directory.Path, "programs"), "*.json");
        string[] statuses = files.Select(file => (string?)JObject.Parse(File.ReadAllText(file))["Status"] ?? "")
            .OrderBy(status => status, StringComparer.Ordinal).ToArray();
        Assert.Equal(new[] { "Duplicate", "Failed" }, statuses);
        Assert.All(files, file => Assert.Equal(JTokenType.Null, JObject.Parse(File.ReadAllText(file))["Quality"]?.Type));
    }

    [Fact]
    public void RewritingTheSameCandidateOverwritesRatherThanAccumulating()
    {
        using var directory = new TemporaryDirectory();
        var writer = new ProgramRunOutputWriter(directory.Path, new ProgramRunOutputOptions { WriteEveryProgram = true });
        (EvolutionCandidate<ProgramGenome> candidate, EvolutionEvaluation evaluation) = Candidate("value = 1\n", 0.5, 9);

        string? first = writer.WriteProgram(candidate, evaluation);
        string? second = writer.WriteProgram(candidate, evaluation);

        Assert.Equal(first, second);
        Assert.Single(Directory.GetFiles(Path.Combine(directory.Path, "programs"), "*.json"));

        // One file, two writes: the count records what the writer did, and the directory records what survives.
        Assert.Equal(2, writer.WrittenProgramCount);
    }

    [Fact]
    public void TheRetentionLimitStopsWritingRatherThanGrowingWithoutBound()
    {
        using var directory = new TemporaryDirectory();
        var writer = new ProgramRunOutputWriter(directory.Path,
            new ProgramRunOutputOptions { WriteEveryProgram = true, MaxRetainedPrograms = 2 });

        var written = new List<string?>();
        for (int index = 0; index < 5; index++)
        {
            (EvolutionCandidate<ProgramGenome> candidate, EvolutionEvaluation evaluation) =
                Candidate($"value = {index}\n", 0.5, 20 + index);
            written.Add(writer.WriteProgram(candidate, evaluation));
        }

        Assert.Equal(2, written.Count(path => path is not null));
        Assert.Equal(3, written.Count(path => path is null));
        Assert.Equal(2, Directory.GetFiles(Path.Combine(directory.Path, "programs"), "*.json").Length);
    }

    [Fact]
    public void ARetentionLimitIsRejectedWhenItCannotMeanAnything()
    {
        var options = new ProgramRunOutputOptions { MaxRetainedPrograms = -1 };
        Assert.Throws<ArgumentOutOfRangeException>(() => options.Validate());

        var named = new ProgramRunOutputOptions { ProgramsDirectoryName = "../escape" };
        Assert.Throws<ArgumentException>(() => named.Validate());

        // The setting survives a copy, so a writer handed a cloned options object behaves the same.
        var cloned = new ProgramRunOutputOptions { WriteEveryProgram = true, MaxRetainedPrograms = 7 }.Clone();
        Assert.True(cloned.WriteEveryProgram);
        Assert.Equal(7, cloned.MaxRetainedPrograms);
    }

    private static ProgramRunOutputObserver Observer(string path, ProgramRunOutputOptions options) =>
        new(new ProgramRunOutputWriter(path, options));

    private static EvolutionEvent<ProgramGenome> Evaluated(
        string source, double? quality, long evaluationId,
        EvolutionEvaluationStatus status = EvolutionEvaluationStatus.Completed)
    {
        (EvolutionCandidate<ProgramGenome> candidate, EvolutionEvaluation evaluation) =
            Candidate(source, quality, evaluationId, status);
        return new EvolutionEvent<ProgramGenome>(
            EvolutionEventKind.Evaluated, evaluationId, candidate, evaluation);
    }

    private static (EvolutionCandidate<ProgramGenome>, EvolutionEvaluation) Candidate(
        string source, double? quality, long evaluationId,
        EvolutionEvaluationStatus status = EvolutionEvaluationStatus.Completed)
    {
        var genome = new ProgramGenome(source, ProgramLanguage.Python, "widened the search window");
        var canonical = new EvolutionCanonicalGenome<ProgramGenome>(genome, genome.Id);
        var lineage = new EvolutionLineage(
            new[] { "parent-a" }, new[] { "inspiration-a" }, "llm-variation", null,
            generation: 3, island: 0, seedStream: 99UL);
        var candidate = new EvolutionCandidate<ProgramGenome>(evaluationId, canonical, lineage);
        var evaluation = new EvolutionEvaluation(
            evaluationId,
            genome.Id,
            status,
            quality,
            EvolutionOptimizationDirection.Maximize,
            new Dictionary<string, double>(StringComparer.Ordinal) { ["length"] = 42.0 },
            Array.Empty<double>(),
            Array.Empty<double>(),
            new EvolutionEvaluationCost(TimeSpan.FromMilliseconds(120), 1, 1.0),
            lineage,
            EvolutionCacheStatus.Miss,
            Array.Empty<EvolutionDiagnostic>(),
            "task-hash",
            "evaluator-hash",
            "configuration-hash",
            metrics: new Dictionary<string, double>(StringComparer.Ordinal) { ["accuracy"] = 0.75 });
        return (candidate, evaluation);
    }
}
