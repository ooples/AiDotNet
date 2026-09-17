using AiDotNet.TestImpact;
using AiDotNet.TestImpact.Xunit;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RunnerProtocol")]
public sealed class CpuResetObservationTests
{
    public enum MissingProof { MissingEntry, OtherEngine, Logging, BeforeStartup, AfterCompletion, Duplicate,
        InvalidEngine, InvalidLogging, MissingCompletion, Observers, MissingEngine, DerivedEngine }

    [Fact]
    public void ResetEntryIsRetainedAlongsideSuccessfulCompletion()
    {
        var ledger = new RuntimeInitializationLedger();
        RuntimeEnvironmentBinding inputs = RuntimeContractEnvironment.Capture(_ => null, false);
        ledger.Record(inputs);
        var entry = new RuntimeCpuResetInput(RuntimeCpuEntryMode.PlainCpu, RuntimeCpuLogging.Suppressed);
        ledger.RecordReset(entry);
        RuntimeInitializationBinding snapshot = ledger.Snapshot;
        ledger.Complete(new(RuntimeCpuMode.Cpu, 1));
        Assert.Null(snapshot.Completion);
        Assert.Equal(entry, snapshot.ResetInput);
        Assert.True(RuntimeProfileEvidence.HasObservedCpuResetPreconditions(new(inputs, ledger.Snapshot)));
    }

    [Theory]
    [InlineData(MissingProof.MissingEntry)]
    [InlineData(MissingProof.OtherEngine)]
    [InlineData(MissingProof.Logging)]
    [InlineData(MissingProof.BeforeStartup)]
    [InlineData(MissingProof.AfterCompletion)]
    [InlineData(MissingProof.Duplicate)]
    [InlineData(MissingProof.InvalidEngine)]
    [InlineData(MissingProof.InvalidLogging)]
    [InlineData(MissingProof.MissingCompletion)]
    [InlineData(MissingProof.Observers)]
    [InlineData(MissingProof.MissingEngine)]
    [InlineData(MissingProof.DerivedEngine)]
    public void CpuAtExitDoesNotEstablishResetEntryConditions(MissingProof missing)
    {
        var ledger = new RuntimeInitializationLedger();
        RuntimeEnvironmentBinding inputs = RuntimeContractEnvironment.Capture(_ => null, missing == MissingProof.Observers);
        RuntimeCpuResetInput entry = new(missing == MissingProof.OtherEngine ? RuntimeCpuEntryMode.Other :
            missing == MissingProof.MissingEngine ? RuntimeCpuEntryMode.Missing : missing == MissingProof.DerivedEngine ? RuntimeCpuEntryMode.DerivedCpu :
            missing == MissingProof.InvalidEngine ? (RuntimeCpuEntryMode)99 : RuntimeCpuEntryMode.PlainCpu,
            missing == MissingProof.Logging ? RuntimeCpuLogging.MayInvokeCallbacks :
            missing == MissingProof.InvalidLogging ? (RuntimeCpuLogging)99 : RuntimeCpuLogging.Suppressed);
        if (missing == MissingProof.BeforeStartup) ledger.RecordReset(entry);
        ledger.Record(inputs);
        if (missing == MissingProof.AfterCompletion) ledger.Complete(new(RuntimeCpuMode.Cpu, 1));
        if (missing != MissingProof.MissingEntry) ledger.RecordReset(entry);
        if (missing == MissingProof.Duplicate) ledger.RecordReset(entry);
        if (missing != MissingProof.MissingCompletion) ledger.Complete(new(RuntimeCpuMode.Cpu, 1));
        Assert.False(RuntimeProfileEvidence.HasObservedCpuResetPreconditions(new(inputs, ledger.Snapshot)));
    }
}
