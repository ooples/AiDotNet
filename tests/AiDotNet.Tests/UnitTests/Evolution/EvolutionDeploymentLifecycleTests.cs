using System.Text;
using System.Text.Json;
using AiDotNet.Configuration;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Deployment;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Evolution;

public sealed partial class EvolutionDeploymentLifecycleTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "aidotnet-deployment-" + Guid.NewGuid().ToString("N"));
    private static EvolutionDeploymentEnvelope Envelope(int changed = -1)
    {
        string[] hashes = Enumerable.Range(0, 6).Select(i => new string(i == changed ? 'f' : (char)('a' + i % 5), 64)).ToArray();
        return new(hashes[0], hashes[1], hashes[2], hashes[3], hashes[4], hashes[5]);
    }
    private static EvolutionDeployableArtifact Program(string source, EvolutionDeploymentEnvelope? envelope = null) =>
        EvolutionDeployableArtifact.FromProgram(new ProgramGenome(source), envelope ?? Envelope());
    private EvolutionDeploymentArtifactRegistry Registry() => new(_root);
    private static EvolutionDeploymentPolicy Policy(int maximumRetunes = 1, TimeSpan? timeout = null, bool allow = true) =>
        new(allow, EvolutionOptimizationDirection.Maximize, pairedSamples: 5, maximumRetunes: maximumRetunes,
            searchEvaluations: 3, searchProposals: 6, timeout: timeout ?? TimeSpan.FromSeconds(10),
            gracePeriod: TimeSpan.FromMilliseconds(100), cooldown: TimeSpan.Zero, monitoringSamples: 2, consecutiveRegressions: 2);
    private static EvolutionDeploymentMeasurement Measurement(double quality, bool correct = true, bool fresh = true) =>
        new(correct, quality, EvolutionOptimizationDirection.Maximize, TimeSpan.FromMilliseconds(1), 1, fresh);
    private static ValueTask<EvolutionDeploymentMeasurement> Evaluate(EvolutionDeployableArtifact artifact, int pair, CancellationToken token)
    {
        token.ThrowIfCancellationRequested();
        return new(Measurement(artifact.ReadProgram().Source switch { "winner" => 2, "newer" => 3, _ => 1 }));
    }
    private EvolutionDeploymentLifecycle Lifecycle(EvolutionDeploymentPolicy? policy = null,
        Func<EvolutionDeployableArtifact, int, CancellationToken, ValueTask<EvolutionDeploymentMeasurement>>? evaluate = null) =>
        new(Registry(), Envelope(), policy ?? Policy(), envelope => Program("base", envelope), evaluate ?? Evaluate);

    [Fact]
    public void Registry_RoundTripsExactSourceWithoutNormalizingOrActivating()
    {
        var registry = Registry();
        var original = Program("x = '''first\r\nsecond  '''\r\n");
        Assert.Equal(original.Id, registry.Stage(original));
        Assert.Equal(original.Id, registry.Stage(original));
        var loaded = new EvolutionDeploymentArtifactRegistry(_root).Load(original.Id, Envelope());
        Assert.Equal(original.CopyPayload(), loaded.CopyPayload());
        Assert.Equal(original.ReadProgram().Source, loaded.ReadProgram().Source);
        Assert.Single(Directory.GetFiles(_root, "*.artifact"));
        Assert.Null(registry.ReadSlot().ActiveId);
    }

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)] [InlineData(4)] [InlineData(5)]
    public void Registry_RejectsEveryChangedApplicabilityInput(int dimension)
    {
        var registry = Registry();
        string id = registry.Stage(Program("winner"));
        Assert.Throws<InvalidDataException>(() => registry.Load(id, Envelope(dimension)));
    }

    [Fact]
    public void Registry_RejectsTamperingTraversalAndNoncanonicalMetadata()
    {
        var registry = Registry();
        var artifact = Program("winner");
        registry.Stage(artifact);
        Assert.Throws<ArgumentException>(() => registry.Load("../outside", Envelope()));
        string path = Path.Combine(_root, artifact.Id + ".artifact");
        string original = File.ReadAllText(path);
        File.WriteAllText(path, original.Replace("\"SchemaVersion\":1", "\"SchemaVersion\":1,\"SchemaVersion\":1"));
        Assert.Throws<InvalidDataException>(() => registry.Load(artifact.Id, Envelope()));
        Assert.Throws<InvalidDataException>(() => registry.Stage(artifact));
        File.WriteAllText(path, original);
        File.WriteAllText(Path.Combine(_root, artifact.PayloadHash + ".payload"), "corrupt");
        Assert.Throws<InvalidDataException>(() => registry.Load(artifact.Id, Envelope()));
    }

    [Fact]
    public void ModelArtifact_FreezesTrainedBytesAndRestoresOnlyAnExplicitMatchingType()
    {
        using var model = new TinyModel { Weight = 7 };
        var artifact = EvolutionDeployableArtifact.FromModel(model, "tiny-v1", Envelope());
        model.Weight = 99;
        var bytes = artifact.CopyPayload(); bytes[0] ^= 1;
        var registry = Registry(); registry.Stage(artifact);
        var loaded = registry.Load(artifact.Id, Envelope());
        using var restored = loaded.RestoreModel(() => new TinyModel(), "tiny-v1");
        Assert.Equal(7, restored.Weight);
        var wrong = new DifferentTinyModel();
        Assert.Throws<InvalidDataException>(() => loaded.RestoreModel<IModelSerializer>(() => wrong, "tiny-v1"));
        Assert.True(wrong.Disposed);
        Assert.Throws<InvalidDataException>(() => loaded.RestoreModel(() => new TinyModel(), "tiny-v2"));
        Assert.Throws<InvalidDataException>(() => loaded.ReadProgram());
    }

    [Fact]
    public async Task Promotion_RequiresExplicitPolicyAndFreshFixedPairs()
    {
        int calls = 0;
        var denied = Lifecycle(Policy(allow: false), (artifact, pair, token) => { calls++; return Evaluate(artifact, pair, token); });
        Assert.Equal("PersistencePolicyDenied", (await denied.PromoteAsync(Program("winner"))).Outcome);
        Assert.Equal(0, calls);
        var order = new List<string>();
        var lifecycle = Lifecycle(evaluate: (artifact, pair, token) =>
        { order.Add(pair + ":" + artifact.ReadProgram().Source); return Evaluate(artifact, pair, token); });
        var result = await lifecycle.PromoteAsync(Program("winner"));
        Assert.True(result.Activated);
        Assert.Equal(new[] { "0:winner", "0:base", "1:base", "1:winner", "2:winner", "2:base", "3:base", "3:winner", "4:winner", "4:base" }, order);
        Assert.False(lifecycle.Select(Envelope()).IsFallback);
        Assert.Equal(Program("winner").Id, new EvolutionDeploymentLifecycle(Registry(), Envelope(), Policy(),
            env => Program("base", env), Evaluate).Select(Envelope()).Artifact.Id);
        using var evidence = JsonDocument.Parse(Registry().ReadEvidence(result.EvidenceId!));
        Assert.Equal(5, evidence.RootElement.GetProperty("Pairs").GetArrayLength());
    }

    [Theory]
    [InlineData(false, true)]
    [InlineData(true, false)]
    public async Task Promotion_RejectsIncorrectOrReusedValidationBeforeMoreWork(bool correct, bool fresh)
    {
        int calls = 0;
        var lifecycle = Lifecycle(evaluate: (_, _, _) => { calls++; return new(Measurement(100, correct, fresh)); });
        var result = await lifecycle.PromoteAsync(Program("winner"));
        Assert.False(result.Activated);
        Assert.Equal("InvalidValidation", result.Outcome);
        Assert.Equal(1, calls);
        Assert.NotEmpty(Registry().ReadEvidence(result.EvidenceId!));
        Assert.True(lifecycle.Select(Envelope()).IsFallback);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public async Task Promotion_RetainsTiesAndInconclusiveComparisonsWithoutActivation(int kind)
    {
        var lifecycle = Lifecycle(evaluate: (artifact, pair, _) => new(Measurement(
            artifact.ReadProgram().Source == "base" ? 1 : kind == 0 ? 1 : pair == 0 ? 0.99 : 1.01)));
        var result = await lifecycle.PromoteAsync(Program("winner"));
        Assert.Equal("InsufficientImprovement", result.Outcome);
        Assert.False(result.Activated);
        Assert.NotEmpty(Registry().ReadEvidence(result.EvidenceId!));
    }

    [Fact]
    public async Task Promotion_StaleComparisonCannotOverwriteAnotherController()
    {
        var entered = Signal(); var release = Signal();
        var first = Lifecycle(evaluate: async (artifact, pair, token) =>
        { entered.TrySetResult(true); await release.Task; return await Evaluate(artifact, pair, token); });
        var pending = first.PromoteAsync(Program("winner"));
        try
        {
            await Arrived(entered);
            Assert.True((await Lifecycle().PromoteAsync(Program("newer"))).Activated);
        }
        finally { release.TrySetResult(true); }
        Assert.Equal("Stale", (await pending).Outcome);
        Assert.Equal(Program("newer").Id, first.Select(Envelope()).Artifact.Id);
    }

    [Fact]
    public async Task Retuning_CoalescesDriftAndChargesAdmissionOnlyWhenStarted()
    {
        var lifecycle = Lifecycle();
        Assert.True((await lifecycle.PromoteAsync(Program("winner"))).Activated);
        Assert.True(lifecycle.Select(Envelope(0)).IsFallback);
        Assert.True(lifecycle.Select(Envelope(1)).IsFallback);
        int calls = 0;
        var result = await lifecycle.RetunePendingAsync((request, _) =>
        {
            calls++;
            Assert.Equal(Envelope(1).Key, request.Envelope.Key);
            Assert.Equal(3, request.MaximumEvaluations); Assert.Equal(6, request.MaximumProposals);
            return Task.FromResult(Program("winner", request.Envelope));
        }, _ => Task.CompletedTask);
        Assert.True(result.Activated);
        Assert.False(lifecycle.Select(Envelope(1)).IsFallback);
        lifecycle.Select(Envelope(2));
        Assert.Equal("BudgetDenied", (await lifecycle.RetunePendingAsync((_, _) => throw new InvalidOperationException(), _ => Task.CompletedTask)).Outcome);
        Assert.Equal(1, calls); Assert.Equal(1, lifecycle.AdmittedRetunes);
    }

    [Fact]
    public async Task Retuning_AbandonedWorkRetainsCapacityAndCannotPublishLate()
    {
        var lifecycle = Lifecycle(Policy(maximumRetunes: 2, timeout: TimeSpan.FromMilliseconds(500)));
        lifecycle.Select(Envelope(0));
        var entered = Signal(); var release = Signal(); var exited = Signal();
        var pending = lifecycle.RetunePendingAsync(async (request, _) =>
        {
            entered.TrySetResult(true);
            try { await release.Task; return Program("winner", request.Envelope); }
            finally { exited.TrySetResult(true); }
        }, _ => Task.CompletedTask);
        try
        {
            await Arrived(entered);
            Assert.Equal("Abandoned", (await pending).Outcome);
            lifecycle.Select(Envelope(0));
            Assert.Equal("Busy", (await lifecycle.RetunePendingAsync((request, _) => Task.FromResult(Program("winner", request.Envelope)), _ => Task.CompletedTask)).Outcome);
            Assert.Equal(1, lifecycle.AdmittedRetunes);
        }
        finally { release.TrySetResult(true); await Arrived(exited); }
        Assert.True(lifecycle.Select(Envelope(0)).IsFallback);
        Assert.Null(Registry().ReadSlot().ActiveId);
    }

    [Fact]
    public async Task Monitoring_RetainsAllWindowsAndRestoresExactPriorAcrossRestart()
    {
        var lifecycle = Lifecycle();
        await lifecycle.PromoteAsync(Program("winner"));
        var observed = lifecycle.Select(Envelope());
        var bad = new[] { Measurement(0), Measurement(0) };
        var now = DateTimeOffset.UtcNow;
        Assert.Equal("Monitoring", (await lifecycle.ObserveAsync(observed, bad, now)).Outcome);
        var result = await lifecycle.ObserveAsync(observed, bad, now.AddSeconds(1));
        Assert.Equal("RolledBack", result.Outcome);
        Assert.True(result.Activated);
        Assert.Equal(Program("base").Id, result.ArtifactId);
        Assert.True(Registry().IsQuarantined(observed.Artifact.Id));
        using var evidence = JsonDocument.Parse(Registry().ReadEvidence(result.EvidenceId!));
        Assert.Equal(2, evidence.RootElement.GetProperty("Windows").GetArrayLength());
        var restarted = Lifecycle();
        var current = restarted.Select(Envelope());
        Assert.False(current.IsFallback);
        Assert.Equal(Program("base").Id, current.Artifact.Id);
        Assert.Equal("Healthy", (await restarted.ObserveAsync(current, new[] { Measurement(1), Measurement(1) }, now.AddSeconds(2))).Outcome);
        Assert.Equal("Stale", (await lifecycle.ObserveAsync(observed, bad, now.AddSeconds(3))).Outcome);
        Assert.Equal("Quarantined", (await lifecycle.PromoteAsync(Program("winner"))).Outcome);
    }

    [Fact]
    public async Task Monitoring_MissingPriorCannotPreventQuarantineAndFallback()
    {
        var lifecycle = Lifecycle();
        await lifecycle.PromoteAsync(Program("winner"));
        var selected = lifecycle.Select(Envelope());
        File.Delete(Path.Combine(_root, Program("base").PayloadHash + ".payload")); // Exact file created by this fixture.
        var now = DateTimeOffset.UtcNow;
        var bad = new[] { Measurement(0), Measurement(0) };
        await lifecycle.ObserveAsync(selected, bad, now);
        Assert.Equal("QuarantinedFallback", (await lifecycle.ObserveAsync(selected, bad, now.AddSeconds(1))).Outcome);
        Assert.True(Registry().IsQuarantined(selected.Artifact.Id));
        Assert.True(lifecycle.Select(Envelope()).IsFallback);
        Assert.True(lifecycle.RetuneRequested);
    }

    [Fact]
    public async Task Monitoring_HealthyWindowResetsAndStaleRevisionCannotQuarantineNewWinner()
    {
        var lifecycle = Lifecycle();
        await lifecycle.PromoteAsync(Program("winner"));
        var selected = lifecycle.Select(Envelope());
        var now = DateTimeOffset.UtcNow;
        var bad = new[] { Measurement(0), Measurement(0) };
        await lifecycle.ObserveAsync(selected, bad, now);
        await Assert.ThrowsAsync<ArgumentException>(() => lifecycle.ObserveAsync(selected, bad, now));
        Assert.False(lifecycle.StorageFaulted);
        Assert.Equal("Healthy", (await lifecycle.ObserveAsync(selected, new[] { Measurement(2), Measurement(2) }, now.AddSeconds(1))).Outcome);
        Assert.Equal("Monitoring", (await lifecycle.ObserveAsync(selected, bad, now.AddSeconds(2))).Outcome);
        await lifecycle.PromoteAsync(Program("newer"));
        Assert.Equal("Stale", (await lifecycle.ObserveAsync(selected, bad, now.AddSeconds(3))).Outcome);
        Assert.False(Registry().IsQuarantined(selected.Artifact.Id));
    }

    [Fact]
    public async Task StorageLossLatchesFallbackInsteadOfReloadingPersistedWinner()
    {
        var lifecycle = Lifecycle();
        await lifecycle.PromoteAsync(Program("winner"));
        File.WriteAllText(Path.Combine(_root, "active.json"), "corrupt");
        Assert.True(lifecycle.Select(Envelope()).IsFallback);
        Assert.True(lifecycle.StorageFaulted);
        Assert.False(lifecycle.RetuneRequested);
        Assert.False((await lifecycle.PromoteAsync(Program("newer"))).Activated);
    }

    private static TaskCompletionSource<bool> Signal() => new(TaskCreationOptions.RunContinuationsAsynchronously);
    private static async Task Arrived(TaskCompletionSource<bool> signal) =>
        Assert.Same(signal.Task, await Task.WhenAny(signal.Task, Task.Delay(TimeSpan.FromSeconds(5))));

    private class TinyModel : IModelSerializer, IDisposable
    {
        internal double Weight { get; set; }
        internal bool Disposed { get; private set; }
        public byte[] Serialize() => BitConverter.GetBytes(Weight);
        public void Deserialize(byte[] data) => Weight = BitConverter.ToDouble(data, 0);
        public void SaveModel(string path) => throw new NotSupportedException();
        public void LoadModel(string path) => throw new NotSupportedException();
        public void Dispose() => Disposed = true;
    }
    private sealed class DifferentTinyModel : TinyModel { }
    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); // Only this fixture's owned GUID directory.
    }
}
