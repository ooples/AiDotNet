using System.Text.Json;
using AiDotNet.TestImpact.Xunit;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RunnerProtocol")]
public sealed class RuntimeContractEnvironmentTests
{
    [Fact]
    public void MissingStartupObservationIsNotInferredFromEffectiveSettings()
    {
        var ledger = new RuntimeInitializationLedger();
        Assert.Equal(RuntimeInitializationStatus.Missing, ledger.Snapshot.Status);
        Assert.Null(ledger.Snapshot.Inputs);
        RuntimeEnvironmentBinding inputs = RuntimeContractEnvironment.Capture(_ => null, false);
        ledger.Record(inputs);
        ledger.Record(inputs);
        Assert.Equal(RuntimeInitializationStatus.Recorded, ledger.Snapshot.Status);
        Assert.Equal(inputs, ledger.Snapshot.Inputs);
    }

    [Fact]
    public void StartupInputChangesCannotBeHiddenByNormalization()
    {
        var first = new RuntimeInitializationLedger();
        var second = new RuntimeInitializationLedger();
        first.Record(RuntimeContractEnvironment.Capture(name => name == "OMP_NUM_THREADS" ? "4" : null, false));
        second.Record(RuntimeContractEnvironment.Capture(name => name == "OMP_NUM_THREADS" ? "8" : null, false));
        Assert.NotEqual(first.Snapshot.Inputs, second.Snapshot.Inputs);
    }

    [Fact]
    public void ConflictingStartupObservationsStayConflicting()
    {
        var ledger = new RuntimeInitializationLedger();
        RuntimeEnvironmentBinding original = RuntimeContractEnvironment.Capture(_ => null, false);
        ledger.Record(original);
        ledger.Record(RuntimeContractEnvironment.Capture(name => name == "AIDOTNET_TEST_CPU_MDOP" ? "2" : null, false));
        ledger.Record(original);
        Assert.Equal(RuntimeInitializationStatus.Conflicting, ledger.Snapshot.Status);
        Assert.Equal(original, ledger.Snapshot.Inputs);
    }

    [Theory]
    [InlineData("AIDOTNET_TEST_CPU_MDOP")]
    [InlineData("OMP_NUM_THREADS")]
    [InlineData("MKL_NUM_THREADS")]
    [InlineData("OPENBLAS_NUM_THREADS")]
    [InlineData("AIDOTNET_LICENSE_KEY")]
    [InlineData("AIDOTNET_LICENSE_TOKEN")]
    [InlineData("AIDOTNET_LICENSE_SCOPE")]
    [InlineData("AIDOTNET_BUILD_KEY")]
    [InlineData("TEMP")]
    [InlineData("DOTNET_STARTUP_HOOKS")]
    public void ContractInputChangesInvalidateTheProfile(string changed)
    {
        RuntimeEnvironmentBinding before = RuntimeContractEnvironment.Capture(_ => null, false);
        RuntimeEnvironmentBinding after = RuntimeContractEnvironment.Capture(name => name == changed ? "changed" : null, false);
        Assert.NotEqual(before.Fingerprint, after.Fingerprint);
    }

    [Theory]
    [InlineData("CORECLR_ENABLE_PROFILING", "1", true)]
    [InlineData("CORECLR_ENABLE_PROFILING", "0", false)]
    [InlineData("DOTNET_STARTUP_HOOKS", "custom-hook.dll", true)]
    [InlineData("CORECLR_PROFILER", "configured-profiler", true)]
    public void ObserverSignalsAreNotMistakenForAnUnobservedHost(string key, string value, bool present)
        => Assert.Equal(present ? RuntimeObserverSignals.Present : RuntimeObserverSignals.NoneReported,
            RuntimeContractEnvironment.Capture(name => name == key ? value : null, false).ObserverSignals);

    [Fact]
    public void DebuggerIsBothBoundAndReported()
    {
        RuntimeEnvironmentBinding clean = RuntimeContractEnvironment.Capture(_ => null, false);
        RuntimeEnvironmentBinding debug = RuntimeContractEnvironment.Capture(_ => null, true);
        Assert.NotEqual(clean.Fingerprint, debug.Fingerprint);
        Assert.Equal(RuntimeObserverSignals.Present, debug.ObserverSignals);
    }

    [Fact]
    public void ArtifactContainsOnlyStableCombinedHashNotCredentials()
    {
        string secret = "private-license-" + Guid.NewGuid().ToString("N");
        RuntimeEnvironmentBinding first = RuntimeContractEnvironment.Capture(name => name == "AIDOTNET_LICENSE_KEY" ? secret : null, false);
        RuntimeEnvironmentBinding second = RuntimeContractEnvironment.Capture(name => name == "AIDOTNET_LICENSE_KEY" ? secret : null, false);
        Assert.Equal(first, second);
        Assert.Equal(64, first.Fingerprint.Length);
        Assert.DoesNotContain(secret, JsonSerializer.Serialize(first));
        Assert.DoesNotContain("AIDOTNET_LICENSE_KEY", JsonSerializer.Serialize(first));
    }

    [Fact]
    public void InvocationOutputPathsDoNotChangeTheRuntimeContractProfile()
    {
        Assert.Equal(RuntimeContractEnvironment.Capture(_ => null, false),
            RuntimeContractEnvironment.Capture(name => name == "ATTRIBUTION_RUN" ? Guid.NewGuid().ToString("N") : null, false));
    }
}
