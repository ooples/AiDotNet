using AiDotNet.Evolution;
using AiDotNet.Evolution.Deployment;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Evolution;

public sealed partial class EvolutionDeploymentLifecycleTests
{
    [Fact]
    public async Task Promotion_CancellationDuringValidationNeverActivates()
    {
        using var cancellation = new CancellationTokenSource();
        var lifecycle = Lifecycle(evaluate: (_, _, _) =>
        {
            cancellation.Cancel();
            return new ValueTask<EvolutionDeploymentMeasurement>(Measurement(2));
        });
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => lifecycle.PromoteAsync(Program("winner"), cancellation.Token));
        Assert.Null(Registry().ReadSlot().ActiveId);
    }

    [Fact]
    public async Task Retuning_EnvironmentChangeDuringSearchDiscardsCandidate()
    {
        var lifecycle = Lifecycle();
        lifecycle.Select(Envelope(0));
        int calls = 0;
        var decision = await lifecycle.RetunePendingAsync((request, _) =>
        {
            calls++;
            lifecycle.Select(Envelope(1));
            return Task.FromResult(Program("winner", request.Envelope));
        }, _ => Task.CompletedTask);
        Assert.Equal("Stale", decision.Outcome);
        Assert.Equal(1, calls);
        Assert.Null(Registry().ReadSlot().ActiveId);
        Assert.True(lifecycle.RetuneRequested);
    }

    [Fact]
    public async Task Promotion_LatencyGateRejectsConsistentlyBetterButSlowerCandidate()
    {
        var lifecycle = Lifecycle(evaluate: (artifact, _, _) => new ValueTask<EvolutionDeploymentMeasurement>(
            new EvolutionDeploymentMeasurement(true, artifact.ReadProgram().Source == "winner" ? 2 : 1,
                EvolutionOptimizationDirection.Maximize,
                TimeSpan.FromMilliseconds(artifact.ReadProgram().Source == "winner" ? 10 : 1), 1)));
        var decision = await lifecycle.PromoteAsync(Program("winner"));
        Assert.Equal("InsufficientImprovement", decision.Outcome);
        Assert.NotEmpty(Registry().ReadEvidence(decision.EvidenceId!));
        Assert.Null(Registry().ReadSlot().ActiveId);
    }

    [Fact]
    public async Task Monitoring_LatencyRegressionAloneRollsBack()
    {
        var lifecycle = Lifecycle();
        await lifecycle.PromoteAsync(Program("winner"));
        var selection = lifecycle.Select(Envelope());
        var slow = new EvolutionDeploymentMeasurement(true, 2, EvolutionOptimizationDirection.Maximize, TimeSpan.FromMilliseconds(10), 1);
        var now = DateTimeOffset.UtcNow;
        Assert.Equal("Monitoring", (await lifecycle.ObserveAsync(selection, new[] { slow, slow }, now)).Outcome);
        Assert.Equal("RolledBack", (await lifecycle.ObserveAsync(selection, new[] { slow, slow }, now.AddSeconds(1))).Outcome);
    }

    [Fact]
    public async Task ModelAdapter_DoesNotMaskEvaluationFailureWhenDisposalAlsoFails()
    {
        var source = new ThrowingDisposeModel();
        var artifact = EvolutionDeployableArtifact.FromModel(source, "failure-v1", Envelope());
        var original = new InvalidOperationException("evaluation failed");
        var evaluator = EvolutionDeploymentEvaluators.Model("failure-v1", _ => new ThrowingDisposeModel(),
            (ThrowingDisposeModel _, int _, CancellationToken _) => throw original);
        var error = await Assert.ThrowsAsync<AggregateException>(async () => await evaluator(artifact, 0, CancellationToken.None));
        Assert.Same(original, error.InnerExceptions[0]);
        Assert.IsType<IOException>(error.InnerExceptions[1]);
    }

    [Fact]
    public void StoredValidation_InvalidRawMeasurementIsAStorageError()
    {
        var evidence = new DeploymentValidationEvidence
        {
            CandidateId = Program("winner").Id, IncumbentId = Program("base").Id, EnvelopeKey = Envelope().Key,
            MaximumPValue = 0.05, MaximumP95LatencyRatio = 1.1,
            Pairs = Enumerable.Range(0, 5).Select(_ => new DeploymentMeasurementPair
            { Candidate = new DeploymentRawMeasurement(), Incumbent = new DeploymentRawMeasurement() }).ToArray()
        };
        Assert.Throws<InvalidDataException>(() => evidence.Validate());
    }

    private sealed class ThrowingDisposeModel : IModelSerializer, IDisposable
    {
        public byte[] Serialize() => new byte[] { 1 };
        public void Deserialize(byte[] data) { }
        public void SaveModel(string path) => throw new NotSupportedException();
        public void LoadModel(string path) => throw new NotSupportedException();
        public void Dispose() => throw new IOException("dispose failed");
    }
}
