using AiDotNet.TestImpact;
using AiDotNet.TestImpact.Xunit;
using Mono.Cecil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RunnerProtocol")]
public sealed class CpuResetCompletionTests
{
    public enum Mutation { NoCompletion, BeforeStartup, BeforeEntry, Duplicate, AfterOverallCompletion }

    [Fact]
    public void CompletedResetIsSeparateFromFinalEngineObservation()
    {
        var inputs = Inputs();
        var ledger = new RuntimeInitializationLedger();
        ledger.Record(inputs);
        ledger.RecordReset(new(RuntimeCpuEntryMode.PlainCpu, RuntimeCpuLogging.Suppressed));
        ledger.CompleteReset();
        ledger.Complete(new(RuntimeCpuMode.Cpu, 1));
        Assert.True(RuntimeProfileEvidence.HasObservedSuccessfulCpuReset(new(inputs, ledger.Snapshot)));
    }

    [Theory]
    [InlineData(Mutation.NoCompletion)] [InlineData(Mutation.BeforeStartup)] [InlineData(Mutation.BeforeEntry)]
    [InlineData(Mutation.Duplicate)] [InlineData(Mutation.AfterOverallCompletion)]
    public void InvalidCompletionOrderingCannotProveReset(Mutation mutation)
    {
        var inputs = Inputs();
        var ledger = new RuntimeInitializationLedger();
        if (mutation == Mutation.BeforeStartup) ledger.CompleteReset();
        ledger.Record(inputs);
        if (mutation == Mutation.BeforeEntry) ledger.CompleteReset();
        ledger.RecordReset(new(RuntimeCpuEntryMode.PlainCpu, RuntimeCpuLogging.Suppressed));
        if (mutation != Mutation.NoCompletion && mutation != Mutation.AfterOverallCompletion) ledger.CompleteReset();
        if (mutation == Mutation.Duplicate) ledger.CompleteReset();
        ledger.Complete(new(RuntimeCpuMode.Cpu, 1));
        if (mutation == Mutation.AfterOverallCompletion) ledger.CompleteReset();
        Assert.False(RuntimeProfileEvidence.HasObservedSuccessfulCpuReset(new(inputs, ledger.Snapshot)));
    }

    [Fact]
    public void ValidObservationDoesNotAuthenticateLookalikeCpuMethods()
    {
        using var assembly = AssemblyDefinition.ReadAssembly(typeof(CpuResetCompletionTests).Assembly.Location);
        var method = assembly.MainModule.GetType(typeof(CpuResetCompletionTests).FullName).Methods.Single(candidate => candidate.Name == nameof(Inputs));
        var inputs = Inputs();
        var profile = new RuntimeContractProfile(inputs, new(RuntimeInitializationStatus.Recorded, inputs, new(RuntimeCpuMode.Cpu, 1),
            new(RuntimeCpuEntryMode.PlainCpu, RuntimeCpuLogging.Suppressed), RuntimeCpuResetOutcome.Completed));
        Assert.Equal(CpuStartupContract.Unresolved, ReviewedCpuStartup.Read(method, method, profile));
    }

    private static RuntimeEnvironmentBinding Inputs() => new(1, new string('a', 64), RuntimeObserverSignals.NoneReported,
        RuntimeGpuStartupPolicy.Disabled, RuntimeGpuDiagnosticsPolicy.NoDumpRequested, RuntimeLicenseStartupPolicy.DefaultTestLicense,
        RuntimeCpuParallelismPolicy.DefaultSingleThread);
}
