using System.Globalization;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

public sealed class EvolutionTraceTests
{
    [Fact]
    public void FullyPopulatedRecordSurvivesAJsonLinesRoundTrip()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Path, "trace.jsonl");
        EvolutionTraceRecord original = FullRecord();

        EvolutionTraceFile.Write(new[] { original }, path);
        EvolutionTraceReadResult result = EvolutionTraceFile.Read(path);

        Assert.True(result.IsComplete);
        Assert.Equal(EvolutionTraceFormat.JsonLines, result.Format);
        Assert.False(result.Compressed);
        AssertRecordsEqual(original, Assert.Single(result.Records));
    }

    [Fact]
    public void FullyPopulatedRecordSurvivesAJsonDocumentRoundTrip()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Path, "trace.json");
        EvolutionTraceRecord original = FullRecord();
        var written = new EvolutionTraceSummary("round-trip", EvolutionTraceFormat.Json, compressed: false)
        {
            RecordsWritten = 1,
            IsClosed = true
        };

        EvolutionTraceFile.Write(new[] { original }, path, EvolutionTraceFormat.Json, compress: false, written);
        EvolutionTraceReadResult result = EvolutionTraceFile.Read(path);

        Assert.True(result.IsComplete);
        Assert.Equal(EvolutionTraceFormat.Json, result.Format);
        AssertRecordsEqual(original, Assert.Single(result.Records));
        EvolutionTraceSummary sidecar = Assert.IsType<EvolutionTraceSummary>(result.Summary);
        Assert.Equal("round-trip", sidecar.RunId);
        Assert.True(sidecar.IsClosed);
    }

    [Fact]
    public void CompressedTraceRoundTripsAndIsDetectedFromItsMagicBytesNotItsName()
    {
        using var directory = new TemporaryDirectory();
        // Deliberately a plain ".jsonl" name for gzip content: OpenEvolve's loader trusts the suffix first and fails.
        string path = Path.Combine(directory.Path, "misnamed.jsonl");
        EvolutionTraceRecord[] originals = Enumerable.Range(0, 25).Select(SimpleRecord).ToArray();

        EvolutionTraceFile.Write(originals, path, EvolutionTraceFormat.JsonLines, compress: true);
        EvolutionTraceReadResult result = EvolutionTraceFile.Read(path);

        Assert.True(result.Compressed);
        Assert.True(result.IsComplete);
        Assert.Equal(EvolutionTraceFormat.JsonLines, result.Format);
        Assert.Equal(originals.Length, result.Records.Count);
        for (int index = 0; index < originals.Length; index++) AssertRecordsEqual(originals[index], result.Records[index]);
        Assert.Equal(0x1F, File.ReadAllBytes(path)[0]);
    }

    [Fact]
    public async Task ObserverWritesOneRecordPerTerminalEvaluationAndGzipsOnRequest()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Path, "run.trace.jsonl.gz");
        EvolutionTraceSummary summary;
        using (EvolutionTraceObserver<TestGenome> tracer = Tracer(path, options => options.Compress = true))
        {
            EvolutionRunResult<TestGenome> result = await CreateEngine(new SyntheticEvolutionTask(),
                Options(maxAttempts: 6, batchSize: 3), tracer).RunAsync(Seeds(6));
            Assert.Equal(6, result.Counters.CompletedEvaluations);
            summary = tracer.Summary;
        }

        EvolutionTraceReadResult trace = EvolutionTraceFile.Read(path);
        Assert.True(trace.Compressed);
        Assert.True(trace.IsComplete);
        Assert.Equal(6, trace.Records.Count);
        Assert.True(summary.IsClosed);
        Assert.False(summary.IsTruncated);
        Assert.Equal(6, summary.RecordsWritten);
        Assert.Equal(0, summary.RecordsDropped);
        Assert.All(trace.Records, record => Assert.Equal(EvolutionEvaluationStatus.Completed, record.Status));
        Assert.All(trace.Records, record => Assert.NotNull(record.Cell));
        Assert.Equal(Enumerable.Range(0, 6).Select(index => (long)index),
            trace.Records.Select(record => record.EvaluationId));
    }

    [Fact]
    public async Task EveryTerminalStatusIsTracedIncludingDuplicatesOpenEvolveWouldDrop()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Path, "statuses.jsonl");
        EvolutionRunResult<TestGenome> result;
        using (EvolutionTraceObserver<TestGenome> tracer = Tracer(path))
        {
            result = await CreateEngine(new SyntheticEvolutionTask(), Options(maxAttempts: 4, batchSize: 4), tracer)
                .RunAsync(new[] { new TestGenome(1), new TestGenome(1), new TestGenome(2) });
        }

        EvolutionTraceReadResult trace = EvolutionTraceFile.Read(path);
        // One record per terminal evaluation, whatever its status: the trace and the run's own counters must agree.
        Assert.Equal(result.Counters.StatusCounts.Values.Sum(), trace.Records.Count);
        foreach (KeyValuePair<EvolutionEvaluationStatus, long> status in result.Counters.StatusCounts)
            Assert.Equal(status.Value, trace.Records.Count(record => record.Status == status.Key));
        Assert.True(result.Counters.StatusCounts[EvolutionEvaluationStatus.Duplicate] > 0);
        Assert.Contains(trace.Records, record => record.Status == EvolutionEvaluationStatus.Duplicate);
    }

    [Fact]
    public async Task RecordBoundStopsTheTraceAndReportsExactlyWhatItDropped()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Path, "bounded.jsonl");
        EvolutionTraceSummary summary;
        using (EvolutionTraceObserver<TestGenome> tracer = Tracer(path, options =>
        {
            options.MaxRecords = 2;
            options.FlushEveryRecords = 1;
        }))
        {
            EvolutionRunResult<TestGenome> result = await CreateEngine(new SyntheticEvolutionTask(),
                Options(maxAttempts: 6, batchSize: 3), tracer).RunAsync(Seeds(6));
            Assert.Equal(6, result.Counters.CompletedEvaluations);
            summary = tracer.Summary;
        }

        Assert.True(summary.IsTruncated);
        Assert.Equal(2, summary.RecordsWritten);
        Assert.Equal(4, summary.RecordsDropped);
        Assert.Equal(2, EvolutionTraceFile.Read(path).Records.Count);
        EvolutionTraceSummary sidecar = Assert.IsType<EvolutionTraceSummary>(EvolutionTraceFile.ReadSummary(path));
        Assert.True(sidecar.IsTruncated);
        Assert.Equal(4, sidecar.RecordsDropped);
    }

    [Fact]
    public async Task ByteBoundStopsTheTraceBeforeItExceedsTheConfiguredSize()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Path, "bytes.jsonl");
        EvolutionTraceSummary summary;
        using (EvolutionTraceObserver<TestGenome> tracer = Tracer(path, options =>
        {
            options.MaxBytes = 1200;
            options.FlushEveryRecords = 1;
        }))
        {
            await CreateEngine(new SyntheticEvolutionTask(), Options(maxAttempts: 8, batchSize: 4), tracer)
                .RunAsync(Seeds(8));
            summary = tracer.Summary;
        }

        Assert.True(summary.IsTruncated);
        Assert.InRange(summary.RecordsWritten, 1L, 7L);
        Assert.Equal(8L - summary.RecordsWritten, summary.RecordsDropped);
        Assert.True(summary.BytesWritten <= 1200);
        Assert.True(new FileInfo(path).Length <= 1200);
        Assert.Equal(summary.RecordsWritten, EvolutionTraceFile.Read(path).Records.Count);
    }

    [Fact]
    public async Task AFailingSinkNeitherStopsTheRunNorHidesItsOwnFailure()
    {
        using var directory = new TemporaryDirectory();
        // A directory cannot be opened as a file, so every write attempt fails the way a full disk would.
        string path = Path.Combine(directory.Path, "occupied");
        Directory.CreateDirectory(path);

        EvolutionTraceSummary summary;
        EvolutionRunResult<TestGenome> result;
        using (EvolutionTraceObserver<TestGenome> tracer = Tracer(path, options => options.FlushEveryRecords = 1))
        {
            result = await CreateEngine(new SyntheticEvolutionTask(), Options(maxAttempts: 4, batchSize: 2), tracer)
                .RunAsync(Seeds(4));
            summary = tracer.Summary;
        }

        Assert.Equal(EvolutionStopReason.EvaluationBudgetReached, result.StopReason);
        Assert.Equal(4, result.Counters.CompletedEvaluations);
        Assert.NotNull(result.Best);
        Assert.True(summary.ObserverFailures > 0);
        Assert.NotNull(summary.FirstFailureMessage);
        Assert.Equal(0, summary.RecordsWritten);
        Assert.True(summary.IsTruncated);
    }

    [Fact]
    public async Task TraceIsFlushedToDiskWhenTheEngineCheckpointsRatherThanOnlyAtTheEnd()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Path, "checkpointed.jsonl");
        var store = new InMemoryEvolutionCheckpointStore();
        TraceProbeObserver probe;
        using (EvolutionTraceObserver<TestGenome> tracer = Tracer(path, options => options.FlushEveryRecords = 1000))
        {
            probe = new TraceProbeObserver(tracer, path);
            EvolutionEngineOptions options = Options(maxAttempts: 8, batchSize: 2);
            options.CheckpointInterval = 2;
            await CreateEngine(new SyntheticEvolutionTask(), options, probe, store).RunAsync(Seeds(8));
        }

        Assert.True(probe.RecordsVisibleAtFirstCheckpoint > 0,
            "The trace must be durable at the first checkpoint, not only when the run ends.");
        Assert.Equal(probe.EvaluationsBeforeFirstCheckpoint, probe.RecordsVisibleAtFirstCheckpoint);
        Assert.True(probe.RecordsVisibleAtFirstCheckpoint < probe.TotalEvaluations,
            "The probe must observe a mid-run checkpoint, not the final one.");
        Assert.Equal(8, EvolutionTraceFile.Read(path).Records.Count);
    }

    [Fact]
    public async Task DisabledTracingWritesNothingAndLeavesTheRunStateHashUnchanged()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Path, "never-written.jsonl");
        var options = new EvolutionTraceOptions { Path = path };
        Assert.False(options.Enabled);

        using var tracer = new EvolutionTraceObserver<TestGenome>(options, "test-run", Descriptors());
        Assert.False(tracer.IsEnabled);
        Assert.Equal(string.Empty, tracer.TracePath);

        EvolutionRunResult<TestGenome> traced = await CreateEngine(new SyntheticEvolutionTask(),
            Options(maxAttempts: 6, batchSize: 3), tracer).RunAsync(Seeds(6));
        EvolutionRunResult<TestGenome> untraced = await CreateEngine(new SyntheticEvolutionTask(),
            Options(maxAttempts: 6, batchSize: 3)).RunAsync(Seeds(6));

        Assert.Equal(untraced.StateHash, traced.StateHash);
        Assert.False(File.Exists(path));
        Assert.False(File.Exists(EvolutionOutputLayout.SummaryPathFor(path)));
        Assert.Empty(Directory.GetFiles(directory.Path));
        Assert.Equal(0, tracer.Summary.RecordsWritten);
    }

    [Fact]
    public async Task EnabledTracingLeavesTheRunStateHashUnchanged()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Path, "identity.jsonl");
        EvolutionRunResult<TestGenome> traced;
        using (EvolutionTraceObserver<TestGenome> tracer = Tracer(path))
        {
            traced = await CreateEngine(new SyntheticEvolutionTask(), Options(maxAttempts: 6, batchSize: 3), tracer)
                .RunAsync(Seeds(6));
        }
        EvolutionRunResult<TestGenome> untraced = await CreateEngine(new SyntheticEvolutionTask(),
            Options(maxAttempts: 6, batchSize: 3)).RunAsync(Seeds(6));

        Assert.Equal(untraced.StateHash, traced.StateHash);
        Assert.Equal(6, EvolutionTraceFile.Read(path).Records.Count);
    }

    [Fact]
    public async Task SummaryStatisticsEqualARecomputationFromTheWrittenRecords()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Path, "statistics.jsonl");
        EvolutionTraceSummary summary;
        using (EvolutionTraceObserver<TestGenome> tracer = Tracer(path))
        {
            await CreateEngine(new SyntheticEvolutionTask(), Options(maxAttempts: 10, batchSize: 2), tracer)
                .RunAsync(Seeds(10));
            summary = tracer.Summary;
        }

        IReadOnlyList<EvolutionTraceRecord> records = EvolutionTraceFile.Read(path).Records;
        Assert.Equal(records.Count, summary.RecordsWritten);
        Assert.Equal(records.Count(record => record.IsImprovement), summary.ImprovementCount);
        Assert.Equal(records.Count(record => record.QualityDelta.HasValue), summary.DeltaSampleCount);
        Assert.Equal(records.Count(record => record.CacheStatus == EvolutionCacheStatus.Hit), summary.CacheHits);
        Assert.Equal(records.Sum(record => record.CostUnits), summary.TotalCostUnits, 9);
        Assert.Equal(records.Sum(record => record.Elapsed.Ticks), summary.TotalElapsed.Ticks);
        foreach (EvolutionEvaluationStatus status in records.Select(record => record.Status).Distinct())
            Assert.Equal(records.Count(record => record.Status == status), summary.StatusCounts[status]);
    }

    [Fact]
    public void ParentDeltaSurvivesTheParentLeavingTheArchive()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Path, "delta.jsonl");
        using (EvolutionTraceObserver<TestGenome> tracer = Tracer(path))
        {
            Feed(tracer, Evaluated(0, "parent", quality: 1.0));
            Feed(tracer, Evaluated(1, "child", quality: 4.0, parentIds: new[] { "parent" }));
        }

        IReadOnlyList<EvolutionTraceRecord> records = EvolutionTraceFile.Read(path).Records;
        Assert.Equal(2, records.Count);
        Assert.Null(records[0].QualityDelta);
        Assert.Equal("parent", records[1].ParentGenomeId);
        Assert.Equal(1.0, records[1].ParentQuality);
        Assert.Equal(3.0, records[1].QualityDelta);
        Assert.True(records[1].IsImprovement);
        Assert.Equal(3.0, records[1].MetricDeltas["score"]);
    }

    [Fact]
    public void ImprovementIsJudgedInTheOptimizationDirectionNotBySign()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Path, "minimize.jsonl");
        using (EvolutionTraceObserver<TestGenome> tracer = Tracer(path))
        {
            Feed(tracer, Evaluated(0, "parent", quality: 5.0, direction: EvolutionOptimizationDirection.Minimize));
            Feed(tracer, Evaluated(1, "child", quality: 2.0, parentIds: new[] { "parent" },
                direction: EvolutionOptimizationDirection.Minimize));
        }

        EvolutionTraceRecord child = EvolutionTraceFile.Read(path).Records[1];
        Assert.Equal(-3.0, child.QualityDelta);
        Assert.True(child.IsImprovement);
    }

    [Fact]
    public void ParentQualityCacheIsBoundedAndEvictsInInsertionOrder()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Path, "bounded-parents.jsonl");
        using (EvolutionTraceObserver<TestGenome> tracer = Tracer(path, options => options.ParentQualityCacheSize = 1))
        {
            Feed(tracer, Evaluated(0, "first", quality: 1.0));
            Feed(tracer, Evaluated(1, "second", quality: 2.0));
            Feed(tracer, Evaluated(2, "childOfFirst", quality: 9.0, parentIds: new[] { "first" }));
            Feed(tracer, Evaluated(3, "childOfChild", quality: 10.0, parentIds: new[] { "childOfFirst" }));
        }

        IReadOnlyList<EvolutionTraceRecord> records = EvolutionTraceFile.Read(path).Records;
        // Capacity one: "first" was evicted by "second", so its child has no delta ...
        Assert.Null(records[2].QualityDelta);
        // ... while the most recent completed record is still there for the child that follows it.
        Assert.Equal(1.0, records[3].QualityDelta);
        Assert.Equal("childOfFirst", records[3].ParentGenomeId);
    }

    [Fact]
    public void TruncatedJsonLinesTraceReportsIncompletenessAndKeepsEveryWholeRecord()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Path, "cut.jsonl");
        EvolutionTraceFile.Write(Enumerable.Range(0, 10).Select(SimpleRecord), path);

        byte[] full = File.ReadAllBytes(path);
        File.WriteAllBytes(path, full.Take(full.Length - 40).ToArray());

        EvolutionTraceReadResult result = EvolutionTraceFile.Read(path);
        Assert.False(result.IsComplete);
        Assert.Equal(9, result.Records.Count);
        Assert.Equal(Enumerable.Range(0, 9).Select(index => (long)index),
            result.Records.Select(record => record.Sequence));
    }

    [Fact]
    public void TruncatedJsonDocumentStillYieldsMostOfItsRecords()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Path, "cut.json");
        EvolutionTraceFile.Write(Enumerable.Range(0, 100).Select(SimpleRecord), path, EvolutionTraceFormat.Json);

        byte[] full = File.ReadAllBytes(path);
        File.WriteAllBytes(path, full.Take(full.Length / 2).ToArray());

        EvolutionTraceReadResult result = EvolutionTraceFile.Read(path);
        Assert.False(result.IsComplete);
        Assert.True(result.Records.Count >= 40, $"Recovered only {result.Records.Count} of about 50 whole records.");
        Assert.Equal(EvolutionTraceFormat.Json, result.Format);
    }

    [Fact]
    public void StreamingReaderYieldsTheSameRecordsAsAFullRead()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Path, "streamed.jsonl.gz");
        EvolutionTraceRecord[] originals = Enumerable.Range(0, 50).Select(SimpleRecord).ToArray();
        EvolutionTraceFile.Write(originals, path, EvolutionTraceFormat.JsonLines, compress: true);

        long[] streamed = EvolutionTraceFile.ReadRecords(path).Select(record => record.Sequence).ToArray();

        Assert.Equal(originals.Select(record => record.Sequence), streamed);
    }

    [Fact]
    public void TraceOptionsRejectAnEnabledTraceWithoutAPathAndAnImpossibleBound()
    {
        Assert.Throws<ArgumentException>(() =>
            new EvolutionTraceObserver<TestGenome>(new EvolutionTraceOptions { Enabled = true }, "run"));
        Assert.Throws<ArgumentOutOfRangeException>(() => new EvolutionTraceObserver<TestGenome>(
            new EvolutionTraceOptions { Enabled = true, Path = "trace.jsonl", FlushEveryRecords = 0 }, "run"));
        Assert.Throws<ArgumentOutOfRangeException>(() => new EvolutionTraceObserver<TestGenome>(
            new EvolutionTraceOptions { Enabled = true, Path = "trace.jsonl", MaxBytes = 0 }, "run"));
        Assert.Throws<ArgumentOutOfRangeException>(() => new EvolutionTraceObserver<TestGenome>(
            new EvolutionTraceOptions { Enabled = true, Path = "trace.jsonl", MaxRecords = 0 }, "run"));
    }

    [Fact]
    public void OutputLayoutDerivesDistinctDeterministicPathsAndNeverTouchesTheDiskItself()
    {
        using var directory = new TemporaryDirectory();
        var first = new EvolutionOutputLayout(directory.Path, "nightly/sweep");
        var second = new EvolutionOutputLayout(directory.Path, "nightly:sweep");
        var repeat = new EvolutionOutputLayout(directory.Path, "nightly/sweep");

        Assert.Equal(first.CheckpointPath, repeat.CheckpointPath);
        Assert.NotEqual(first.CheckpointPath, second.CheckpointPath);
        Assert.Equal(Path.Combine(directory.Path, "checkpoints"), first.CheckpointsDirectory);
        Assert.Equal(Path.Combine(directory.Path, "traces"), first.TracesDirectory);
        Assert.EndsWith(".trace.jsonl.gz", first.TracePath(EvolutionTraceFormat.JsonLines, compress: true),
            StringComparison.Ordinal);
        Assert.EndsWith(".trace.json", first.TracePath(EvolutionTraceFormat.Json, compress: false),
            StringComparison.Ordinal);
        Assert.False(Directory.Exists(first.CheckpointsDirectory));
        Assert.False(Directory.Exists(first.TracesDirectory));
        Assert.Equal("plain-run", EvolutionOutputLayout.CreateStem("plain-run"));
    }

    [Fact]
    public async Task CheckpointStoreForAnOutputDirectoryResumesFromItsDerivedPath()
    {
        using var directory = new TemporaryDirectory();
        var layout = new EvolutionOutputLayout(directory.Path, "derived-run");
        JsonEvolutionCheckpointStore store = JsonEvolutionCheckpointStore.ForOutputDirectory(directory.Path, "derived-run");

        EvolutionEngineOptions options = Options(maxAttempts: 6, batchSize: 2);
        options.RunId = "derived-run";
        options.OutputDirectory = directory.Path;
        options.CheckpointInterval = 1;
        await CreateEngine(new SyntheticEvolutionTask(), options, observer: null, store).RunAsync(Seeds(6));

        Assert.True(File.Exists(layout.CheckpointPath));
        EvolutionCheckpoint reloaded = Assert.IsType<EvolutionCheckpoint>(await JsonEvolutionCheckpointStore
            .ForOutputDirectory(directory.Path, "derived-run").LoadLatestAsync("derived-run"));
        Assert.Equal("derived-run", reloaded.RunId);
    }

    [Fact]
    public void OutputDirectoryIsValidatedAndKeptOutOfTheConfigurationHash()
    {
        using var directory = new TemporaryDirectory();
        EvolutionEngineOptions bare = Options(maxAttempts: 2, batchSize: 1);
        EvolutionEngineOptions rooted = Options(maxAttempts: 2, batchSize: 1);
        rooted.OutputDirectory = directory.Path;

        string bareHash = CreateEngine(new SyntheticEvolutionTask(), bare).CompatibilityHash;
        string rootedHash = CreateEngine(new SyntheticEvolutionTask(), rooted).CompatibilityHash;

        Assert.Null(bare.OutputDirectory);
        Assert.Equal(bareHash, rootedHash);
        Assert.False(Directory.Exists(Path.Combine(directory.Path, "checkpoints")));

        EvolutionEngineOptions blank = Options(maxAttempts: 2, batchSize: 1);
        blank.OutputDirectory = "   ";
        Assert.Throws<ArgumentException>(() => CreateEngine(new SyntheticEvolutionTask(), blank));
    }

    private static void Feed(EvolutionTraceObserver<TestGenome> tracer, EvolutionEvent<TestGenome> evolutionEvent) =>
        tracer.OnEventAsync(evolutionEvent).AsTask().GetAwaiter().GetResult();

    private static EvolutionEvent<TestGenome> Evaluated(long sequence, string genomeId, double quality,
        IReadOnlyList<string>? parentIds = null,
        EvolutionOptimizationDirection direction = EvolutionOptimizationDirection.Maximize)
    {
        var lineage = new EvolutionLineage(parentIds, null, "increment", null, sequence, 0, 7UL);
        var candidate = new EvolutionCandidate<TestGenome>(sequence,
            new EvolutionCanonicalGenome<TestGenome>(new TestGenome((int)sequence), genomeId), lineage);
        var evaluation = new EvolutionEvaluation(sequence, genomeId, EvolutionEvaluationStatus.Completed, quality,
            direction, new Dictionary<string, double> { ["x"] = 1 }, Array.Empty<double>(), Array.Empty<double>(),
            new EvolutionEvaluationCost(TimeSpan.FromMilliseconds(3), 1, 1), lineage, EvolutionCacheStatus.Miss,
            Array.Empty<EvolutionDiagnostic>(), "task-v1", "evaluator-v1", "configuration-v1",
            new Dictionary<string, double> { ["score"] = quality });
        return new EvolutionEvent<TestGenome>(EvolutionEventKind.Evaluated, sequence, candidate, evaluation,
            EvolutionArchiveInsertionResult.Inserted);
    }

    private static EvolutionTraceRecord SimpleRecord(int index) => new(index, index,
        index.ToString(CultureInfo.InvariantCulture), EvolutionEvaluationStatus.Completed,
        EvolutionOptimizationDirection.Maximize, 0, index, EvolutionCacheStatus.Miss,
        new DateTimeOffset(2026, 9, 2, 10, 0, 0, TimeSpan.Zero), "task-v1", "evaluator-v1", "configuration-v1")
    {
        Quality = index
    };

    private static EvolutionTraceRecord FullRecord() => new(42, 17, "genome-17",
        EvolutionEvaluationStatus.Completed, EvolutionOptimizationDirection.Minimize, 3, 9,
        EvolutionCacheStatus.Hit, new DateTimeOffset(2026, 9, 2, 11, 22, 33, 444, TimeSpan.Zero),
        "task-v1", "evaluator-v1", "configuration-v1")
    {
        Quality = -0.5,
        InsertionResult = EvolutionArchiveInsertionResult.Replaced,
        Cell = "2,4",
        Descriptors = new Dictionary<string, double> { ["x"] = 1.25, ["y"] = -3.5 },
        Metrics = new Dictionary<string, double> { ["loss"] = 0.5 },
        ParentGenomeId = "genome-11",
        ParentQuality = 0.25,
        QualityDelta = -0.75,
        IsImprovement = true,
        MetricDeltas = new Dictionary<string, double> { ["loss"] = -0.125 },
        ParentIds = new[] { "genome-11" },
        InspirationIds = new[] { "genome-3", "genome-5" },
        VariationOperatorId = "increment",
        RefinerId = "polish",
        SeedStream = ulong.MaxValue,
        AttemptCount = 2,
        CostUnits = 12.5,
        Elapsed = TimeSpan.FromMilliseconds(1234),
        RejectedStage = 1,
        Diagnostics = new[]
        {
            new EvolutionDiagnostic("stage_gate", "rejected at stage 1", false,
                new Dictionary<string, string> { ["stage"] = "1" })
        }
    };

    private static void AssertRecordsEqual(EvolutionTraceRecord expected, EvolutionTraceRecord actual)
    {
        Assert.Equal(expected.SchemaVersion, actual.SchemaVersion);
        Assert.Equal(expected.Sequence, actual.Sequence);
        Assert.Equal(expected.EvaluationId, actual.EvaluationId);
        Assert.Equal(expected.GenomeId, actual.GenomeId);
        Assert.Equal(expected.Status, actual.Status);
        Assert.Equal(expected.Direction, actual.Direction);
        Assert.Equal(expected.Island, actual.Island);
        Assert.Equal(expected.Generation, actual.Generation);
        Assert.Equal(expected.CacheStatus, actual.CacheStatus);
        Assert.Equal(expected.RecordedAtUtc, actual.RecordedAtUtc);
        Assert.Equal(expected.TaskVersionHash, actual.TaskVersionHash);
        Assert.Equal(expected.EvaluatorVersionHash, actual.EvaluatorVersionHash);
        Assert.Equal(expected.ConfigurationHash, actual.ConfigurationHash);
        Assert.Equal(expected.Quality, actual.Quality);
        Assert.Equal(expected.InsertionResult, actual.InsertionResult);
        Assert.Equal(expected.Cell, actual.Cell);
        Assert.Equal(expected.Descriptors, actual.Descriptors);
        Assert.Equal(expected.Metrics, actual.Metrics);
        Assert.Equal(expected.ParentGenomeId, actual.ParentGenomeId);
        Assert.Equal(expected.ParentQuality, actual.ParentQuality);
        Assert.Equal(expected.QualityDelta, actual.QualityDelta);
        Assert.Equal(expected.IsImprovement, actual.IsImprovement);
        Assert.Equal(expected.MetricDeltas, actual.MetricDeltas);
        Assert.Equal(expected.ParentIds, actual.ParentIds);
        Assert.Equal(expected.InspirationIds, actual.InspirationIds);
        Assert.Equal(expected.VariationOperatorId, actual.VariationOperatorId);
        Assert.Equal(expected.RefinerId, actual.RefinerId);
        Assert.Equal(expected.SeedStream, actual.SeedStream);
        Assert.Equal(expected.AttemptCount, actual.AttemptCount);
        Assert.Equal(expected.CostUnits, actual.CostUnits);
        Assert.Equal(expected.Elapsed, actual.Elapsed);
        Assert.Equal(expected.RejectedStage, actual.RejectedStage);
        Assert.Equal(expected.Diagnostics.Count, actual.Diagnostics.Count);
        for (int index = 0; index < expected.Diagnostics.Count; index++)
        {
            Assert.Equal(expected.Diagnostics[index].Code, actual.Diagnostics[index].Code);
            Assert.Equal(expected.Diagnostics[index].Message, actual.Diagnostics[index].Message);
            Assert.Equal(expected.Diagnostics[index].IsRedacted, actual.Diagnostics[index].IsRedacted);
            Assert.Equal(expected.Diagnostics[index].Data, actual.Diagnostics[index].Data);
        }
    }

    private static EvolutionTraceObserver<TestGenome> Tracer(string path, Action<EvolutionTraceOptions>? configure = null)
    {
        var options = new EvolutionTraceOptions { Enabled = true, Path = path };
        configure?.Invoke(options);
        return new EvolutionTraceObserver<TestGenome>(options, "test-run", Descriptors());
    }

    private static TestGenome[] Seeds(int count) =>
        Enumerable.Range(1, count).Select(value => new TestGenome(value)).ToArray();

    private static EvolutionEngineOptions Options(int maxAttempts, int batchSize) => new()
    {
        RunId = "test-run",
        Seed = 77,
        MaxEvaluationAttempts = maxAttempts,
        MaxProposals = 100,
        MaxGenerations = 100,
        ProposalBatchSize = batchSize,
        MaxDegreeOfParallelism = 1,
        IslandCount = 2,
        MigrationInterval = 1,
        MigrantsPerIsland = 1,
        CheckpointInterval = 0
    };

    private static EvolutionEngine<TestGenome> CreateEngine(SyntheticEvolutionTask task, EvolutionEngineOptions options,
        IEvolutionObserver<TestGenome>? observer = null, IEvolutionCheckpointStore? checkpointStore = null) => new(
        task,
        new IncrementVariation(),
        _ => new MapElitesArchive<TestGenome>(Descriptors()),
        options,
        observer: observer,
        checkpointStore: checkpointStore,
        genomeCodec: checkpointStore is null ? null : new TestGenomeCodec());

    private static IReadOnlyList<EvolutionDescriptorDefinition> Descriptors() => new[]
    {
        new EvolutionDescriptorDefinition("x", 0, 100, 10, EvolutionOutOfRangePolicy.Clamp)
    };
}
