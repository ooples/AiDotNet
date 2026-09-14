using System.IO;
using System.Text;
using System.Text.Json;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class DirectoryProgramSampleEvidenceStoreTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "raw-samples-" + Guid.NewGuid().ToString("N"));
    private static readonly ProgramGenome Genome = new("return 1;", ProgramLanguage.CSharp);
    private static readonly EvolutionEvaluationContext Context = new(0, 1, 2, 1);
    private readonly ProgramMeasurementObservation[] _samples = { new("sample-a", 1), new("sample-b", 3), new("sample-c", 5) };
    private readonly DateTimeOffset _observed = new(2026, 9, 11, 12, 0, 0, TimeSpan.Zero);

    private DirectoryProgramSampleEvidenceStore Store(int capacity = 8) => new(_root, "test-collector-v1",
        (_, _, _, _) => new ValueTask<IReadOnlyList<ProgramMeasurementObservation>?>(_samples), capacity);

    private EvolutionTaskResult Measurement(double quality = 3, double? error = null, string statistics = ProgramMeasurementStatistics.Version,
        EvolutionMeasurementOriginKind kind = EvolutionMeasurementOriginKind.Measured, string[]? ids = null, double? lower = null) =>
        EvolutionTaskResult.Completed(quality, new Dictionary<string, double> { ["size"] = 1 }, costUnits: 3)
            .WithMeasurementOrigin(new EvolutionMeasurementOrigin(new string('a', 64), "source", "0", ids ?? _samples.Select(s => s.SampleId).ToArray(),
                _observed, 3, "observations-v1", statistics, kind, error ?? Math.Sqrt(4d / 3), lower, lower.HasValue ? 9 : null, lower.HasValue ? 0.95 : null));

    [Fact]
    public async Task Reopened_provider_recomputes_raw_statistics_and_preserves_origin()
    {
        var original = Measurement();
        string digest = (await Store().RetainAsync(Genome, original, Context))!;
        Assert.NotNull(digest);
        var reused = EvolutionTaskResult.Completed(3, new Dictionary<string, double> { ["size"] = 1 }, costUnits: 0)
            .WithMeasurementOrigin(original.MeasurementOrigin!.AsReused(EvolutionMeasurementOriginKind.PersistentReuse));
        var reopened = new DirectoryProgramSampleEvidenceStore(_root, "test-collector-v1", (_, _, _, _) => throw new InvalidOperationException("Verify must not acquire new samples."));
        Assert.True(await reopened.VerifyAsync(Genome, reused, digest, new EvolutionEvaluationContext(19, 55, 9, 1)));
        Assert.Equal(3, reused.MeasurementOrigin!.SampleCount);
        Assert.Equal(original.MeasurementOrigin.StandardError, reused.MeasurementOrigin.StandardError);
        using var artifact = JsonDocument.Parse(File.ReadAllText(Path.Combine(_root, digest + ".json")));
        Assert.Equal(new[] { 1d, 3d, 5d }, artifact.RootElement.GetProperty("Observations").EnumerateArray().Select(e => e.GetProperty("Value").GetDouble()));
        Assert.False(reopened.HasTemporaryCleanupFailure);
    }

    [Theory]
    [InlineData("mean")]
    [InlineData("error")]
    [InlineData("statistics")]
    [InlineData("identities")]
    [InlineData("interval")]
    [InlineData("reused")]
    public async Task Unverified_statistics_or_original_identity_cannot_be_retained(string change)
    {
        var measurement = change switch
        {
            "mean" => Measurement(4),
            "error" => Measurement(error: 0),
            "statistics" => Measurement(statistics: "arbitrary-median"),
            "identities" => Measurement(ids: new[] { "other-a", "sample-b", "sample-c" }),
            "interval" => Measurement(lower: 0),
            _ => Measurement(kind: EvolutionMeasurementOriginKind.PersistentReuse)
        };
        Assert.Null(await Store().RetainAsync(Genome, measurement, Context));
        Assert.Empty(Directory.GetFiles(_root, "*.json"));
    }

    [Fact]
    public async Task Verification_binds_exact_program_values_origin_and_provider_semantics()
    {
        var store = Store(); var original = Measurement();
        string digest = (await store.RetainAsync(Genome, original, Context))!;
        Assert.False(await store.VerifyAsync(new ProgramGenome("return 2;", ProgramLanguage.CSharp), original, digest, Context));
        Assert.False(await store.VerifyAsync(Genome, Measurement(4), digest, Context));
        Assert.False(await store.VerifyAsync(Genome, Measurement(error: 0), digest, Context));
        Assert.False(await store.VerifyAsync(Genome, Measurement(ids: new[] { "x", "y", "z" }), digest, Context));
        var changed = new DirectoryProgramSampleEvidenceStore(_root, "other-collector", (_, _, _, _) => new((IReadOnlyList<ProgramMeasurementObservation>?)null));
        Assert.False(await changed.VerifyAsync(Genome, original, digest, Context));
    }

    [Theory]
    [InlineData("raw-value")]
    [InlineData("duplicate-field")]
    [InlineData("unknown-field")]
    [InlineData("truncated")]
    [InlineData("invalid-utf8")]
    public async Task Corrupt_raw_artifacts_fail_even_with_a_new_content_digest(string change)
    {
        var store = Store(); var original = Measurement();
        string digest = (await store.RetainAsync(Genome, original, Context))!;
        string json = File.ReadAllText(Path.Combine(_root, digest + ".json"));
        if (change == "invalid-utf8")
        {
            File.WriteAllBytes(Path.Combine(_root, digest + ".json"), new byte[] { 255 });
        }
        else
        {
            json = change switch
            {
                "raw-value" => json.Replace("\"Value\":1", "\"Value\":2"),
                "duplicate-field" => json.Replace("\"SchemaVersion\":1", "\"SchemaVersion\":1,\"SchemaVersion\":1"),
                "unknown-field" => json.Replace("\"SchemaVersion\":1", "\"SchemaVersion\":1,\"Unknown\":1"),
                _ => "{"
            };
            digest = EvolutionHash.Compute(json);
            File.WriteAllText(Path.Combine(_root, digest + ".json"), json, new UTF8Encoding(false));
        }
        Assert.False(await store.VerifyAsync(Genome, original, digest, Context));
    }

    [Fact]
    public async Task Capacity_is_non_evicting_and_identical_artifacts_are_idempotent()
    {
        var store = Store(1); var original = Measurement();
        string digest = (await store.RetainAsync(Genome, original, Context))!;
        byte[] bytes = File.ReadAllBytes(Path.Combine(_root, digest + ".json"));
        Assert.Equal(digest, await Store(1).RetainAsync(Genome, original, Context));
        Assert.Null(await Store(1).RetainAsync(new ProgramGenome("return 2;", ProgramLanguage.CSharp), original, Context));
        Assert.Equal(bytes, File.ReadAllBytes(Path.Combine(_root, digest + ".json")));
        Assert.Single(Directory.GetFiles(_root, "*.json"));
        Assert.Empty(Directory.GetFiles(_root, "*.tmp"));
    }

    [Fact]
    public async Task Writer_contention_and_cancellation_do_not_publish_partial_artifacts()
    {
        var store = Store();
        using (var gate = new FileStream(Path.Combine(_root, ".writer.lock"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None))
            await Assert.ThrowsAsync<IOException>(() => store.RetainAsync(Genome, Measurement(), Context).AsTask());
        using var canceled = new CancellationTokenSource(); canceled.Cancel();
        await Assert.ThrowsAsync<OperationCanceledException>(() => store.RetainAsync(Genome, Measurement(), Context, canceled.Token).AsTask());
        Assert.Empty(Directory.GetFiles(_root, "*.json"));
        Assert.NotNull(await store.RetainAsync(Genome, Measurement(), Context));
    }

    [Fact]
    public async Task Missing_traversal_and_oversized_artifacts_are_never_reused()
    {
        var store = Store();
        Assert.False(await store.VerifyAsync(Genome, Measurement(), new string('a', 64), Context));
        Assert.False(await store.VerifyAsync(Genome, Measurement(), "../outside", Context));
        string path = Path.Combine(_root, new string('a', 64) + ".json");
        using (var file = File.Create(path)) file.SetLength(DirectoryProgramSampleEvidenceStore.MaximumArtifactBytes + 1);
        await Assert.ThrowsAsync<InvalidDataException>(() => store.VerifyAsync(Genome, Measurement(), new string('a', 64), Context).AsTask());
        Assert.Throws<ArgumentException>(() => new DirectoryProgramSampleEvidenceStore("relative", "v1", (_, _, _, _) => new((IReadOnlyList<ProgramMeasurementObservation>?)null)));
        Assert.Throws<ArgumentException>(() => new DirectoryProgramSampleEvidenceStore(Path.GetPathRoot(_root)!, "v1", (_, _, _, _) => new((IReadOnlyList<ProgramMeasurementObservation>?)null)));
    }

    [Fact]
    public void Statistics_reject_missing_repeated_nonfinite_or_unbounded_observations()
    {
        var stats = ProgramMeasurementStatistics.Calculate(_samples);
        Assert.Equal(3, stats.Mean); Assert.Equal(Math.Sqrt(4d / 3), stats.StandardError);
        Assert.Throws<ArgumentException>(() => ProgramMeasurementStatistics.Calculate(_samples.Take(1)));
        Assert.Throws<ArgumentException>(() => ProgramMeasurementStatistics.Calculate(new[] { _samples[0], _samples[0] }));
        Assert.Throws<ArgumentException>(() => ProgramMeasurementStatistics.Calculate(Enumerable.Range(0, 257).Select(i => new ProgramMeasurementObservation("s" + i, i))));
        Assert.Throws<ArgumentException>(() => ProgramMeasurementStatistics.Calculate(new[] { new ProgramMeasurementObservation("a", double.MaxValue), new ProgramMeasurementObservation("b", -double.MaxValue) }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ProgramMeasurementObservation("a", double.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ProgramMeasurementObservation("a", double.PositiveInfinity));
    }

    [Theory]
    [InlineData("warm")]
    [InlineData("force-fresh")]
    [InlineData("expired")]
    [InlineData("corrupt")]
    [InlineData("missing")]
    public async Task Real_directory_cache_and_raw_provider_preserve_freshness_and_current_correctness(string phase)
    {
        var backend = new SampleBackend(_observed);
        int correctnessCalls = 0;
        var correctness = new DelegateProgramFitnessEvaluator((_, _, _) =>
        {
            correctnessCalls++;
            return new(new EvolutionTaskResult(EvolutionEvaluationStatus.Completed, 1, costUnits: 2));
        });
        var ledgers = new List<EvolutionResourceLedger>();
        async Task<EvolutionTaskResult> Evaluate(string run, bool forceFresh = false)
        {
            var ledger = new EvolutionResourceLedger(run, new EvolutionResources(new Dictionary<string, decimal>
            {
                [EvolutionPersistentEvaluationCache.StoreInvocationResource] = 4,
                [PersistentProgramFitnessEvaluator.EvidenceInvocationResource] = 4
            }));
            ledgers.Add(ledger);
            var evidence = new DirectoryProgramSampleEvidenceStore(Path.Combine(_root, "raw"), "authored-samples-v1",
                (_, _, _, _) => new ValueTask<IReadOnlyList<ProgramMeasurementObservation>?>(backend.Samples));
            var cached = new PersistentProgramFitnessEvaluator(backend, new DirectoryEvolutionEvaluationStore(Path.Combine(_root, "cache")),
                evidence, backend.Scope, new EvolutionEvaluationReusePolicy(EvolutionEvaluationReuseMode.ExistingSamples, TimeSpan.FromHours(1), 3),
                ledger, "three-observations-v1", run, forceFresh, () => backend.Now);
            return await new CorrectnessGatedProgramFitnessEvaluator(correctness, cached).EvaluateAsync(Genome, Context);
        }
        var cold = await Evaluate("cold");
        if (phase == "expired") backend.Now += TimeSpan.FromHours(2);
        if (phase is "corrupt" or "missing")
        {
            string file = Assert.Single(Directory.GetFiles(Path.Combine(_root, "raw"), "*.json"));
            if (phase == "corrupt") File.WriteAllText(file, "{}");
            else File.Delete(file); // Exact artifact in this test's private directory only.
        }
        var result = await Evaluate("next", phase == "force-fresh");
        bool reused = phase == "warm";
        Assert.Equal(2, correctnessCalls);
        Assert.Equal(reused ? 1 : 2, backend.Calls);
        Assert.Equal(5, cold.CostUnits); Assert.Equal(reused ? 2 : 5, result.CostUnits);
        Assert.Equal(reused ? EvolutionMeasurementOriginKind.PersistentReuse : EvolutionMeasurementOriginKind.Measured, result.MeasurementOrigin!.Kind);
        Assert.Equal(3, result.MeasurementOrigin.SampleCount);
        Assert.Equal(3, result.MeasurementOrigin.OriginalCostUnits);
        Assert.Equal(cold.MeasurementOrigin!.StandardError, result.MeasurementOrigin.StandardError);
        Assert.Equal(reused, cold.MeasurementOrigin.SampleSetHash == result.MeasurementOrigin.SampleSetHash);
        Assert.All(ledgers, ledger => Assert.Equal(0, ledger.Snapshot().Unknown));
        if (reused)
        {
            Assert.Equal(cold.MeasurementOrigin.ObservedAt, result.MeasurementOrigin.ObservedAt);
            Assert.Equal(3m, ledgers.Sum(ledger => ledger.Snapshot().Spent[EvolutionPersistentEvaluationCache.StoreInvocationResource]));
            Assert.Equal(2m, ledgers.Sum(ledger => ledger.Snapshot().Spent[PersistentProgramFitnessEvaluator.EvidenceInvocationResource]));
        }
    }

    private sealed class SampleBackend : IProgramFitnessEvaluator
    {
        internal SampleBackend(DateTimeOffset now)
        {
            Now = now;
            var codec = new ProgramGenomeCodec();
            Scope = new EvolutionReuseScope(Id, VersionHash, VersionHash, codec.Id, codec.VersionHash, "constraints-v1", "authored-data-v1",
                "three-samples-v1", "compiler-na-v1", "runtime-na-v1", "hardware-na-v1", "current-correctness-v1");
        }
        public string Id => "raw-sample-fixture";
        public string VersionHash => "v1";
        internal EvolutionReuseScope Scope { get; }
        internal DateTimeOffset Now { get; set; }
        internal int Calls { get; private set; }
        internal IReadOnlyList<ProgramMeasurementObservation> Samples { get; private set; } = Array.Empty<ProgramMeasurementObservation>();
        public ValueTask<EvolutionTaskResult> EvaluateAsync(ProgramGenome candidate, EvolutionEvaluationContext context, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested(); Calls++;
            Samples = Enumerable.Range(0, 3).Select(index => new ProgramMeasurementObservation($"acquisition-{Calls}-{index}", 1 + 2 * index)).ToArray();
            var statistics = ProgramMeasurementStatistics.Calculate(Samples);
            return new(EvolutionTaskResult.Completed(statistics.Mean, new Dictionary<string, double>(), costUnits: 3).WithMeasurementOrigin(new EvolutionMeasurementOrigin(
                Scope.StableKey, "acquisition-" + Calls, context.EvaluationId.ToString(), Samples.Select(sample => sample.SampleId), Now,
                3, "observations-v1", ProgramMeasurementStatistics.Version, standardError: statistics.StandardError)));
        }
    }

    public void Dispose()
    {
        // Only this test's generated private child, never a shared directory.
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }
}
