using AiDotNet.TestImpact;
using Mono.Cecil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class GpuStartupContractTests
{
    public enum Mutation { Missing, UnknownGpu, DetectGpu, Dump, UnknownDump, Logging, DerivedCpu, Observers, Conflicting, InvalidLicensePolicy }

    [Fact]
    public void ObservedInputsCannotAuthenticateAnUnreviewedPackage()
    {
        var profile = Profile();
        Assert.True(ReviewedGpuStartup.Inputs(profile));
        using var assembly = AssemblyDefinition.ReadAssembly(typeof(GpuStartupContractTests).Assembly.Location);
        Assert.Equal(GpuStartupContract.Unresolved, ReviewedGpuStartup.Read(assembly, profile));
    }

    [Theory]
    [InlineData(Mutation.Missing)]
    [InlineData(Mutation.UnknownGpu)]
    [InlineData(Mutation.DetectGpu)]
    [InlineData(Mutation.Dump)]
    [InlineData(Mutation.UnknownDump)]
    [InlineData(Mutation.Logging)]
    [InlineData(Mutation.DerivedCpu)]
    [InlineData(Mutation.Observers)]
    [InlineData(Mutation.Conflicting)]
    [InlineData(Mutation.InvalidLicensePolicy)]
    public void UnknownOrCallbackCapableStartupCannotCloseTheContract(Mutation mutation)
    {
        RuntimeContractProfile profile = Profile();
        RuntimeInitializationBinding startup = profile.Initialization;
        RuntimeEnvironmentBinding inputs = startup.Inputs ?? throw new InvalidOperationException();
        switch (mutation)
        {
            case Mutation.Missing: Assert.False(ReviewedGpuStartup.Inputs(null)); return;
            case Mutation.UnknownGpu: startup = startup with { Inputs = inputs with { GpuStartup = RuntimeGpuStartupPolicy.Unknown } }; break;
            case Mutation.DetectGpu: startup = startup with { Inputs = inputs with { GpuStartup = RuntimeGpuStartupPolicy.AutoDetectionPermitted } }; break;
            case Mutation.Dump: startup = startup with { Inputs = inputs with { GpuDiagnostics = RuntimeGpuDiagnosticsPolicy.DumpRequested } }; break;
            case Mutation.UnknownDump: startup = startup with { Inputs = inputs with { GpuDiagnostics = RuntimeGpuDiagnosticsPolicy.Unknown } }; break;
            case Mutation.Logging: startup = startup with { ResetInput = new(RuntimeCpuEntryMode.PlainCpu, RuntimeCpuLogging.MayInvokeCallbacks) }; break;
            case Mutation.DerivedCpu: startup = startup with { ResetInput = new(RuntimeCpuEntryMode.DerivedCpu, RuntimeCpuLogging.Suppressed) }; break;
            case Mutation.Observers: startup = startup with { Inputs = inputs with { ObserverSignals = RuntimeObserverSignals.Present } }; break;
            case Mutation.Conflicting: startup = startup with { Status = RuntimeInitializationStatus.Conflicting }; break;
            case Mutation.InvalidLicensePolicy: startup = startup with { Inputs = inputs with { LicenseStartup = (RuntimeLicenseStartupPolicy)(-1) } }; break;
        }
        Assert.False(ReviewedGpuStartup.Inputs(profile with { Initialization = startup }));
    }

    private static RuntimeContractProfile Profile()
    {
        var inputs = new RuntimeEnvironmentBinding(1, new string('a', 64), RuntimeObserverSignals.NoneReported,
            RuntimeGpuStartupPolicy.Disabled, RuntimeGpuDiagnosticsPolicy.NoDumpRequested);
        return new(inputs, new(RuntimeInitializationStatus.Recorded, inputs, new(RuntimeCpuMode.Cpu, 1),
            new(RuntimeCpuEntryMode.PlainCpu, RuntimeCpuLogging.Suppressed)));
    }
}
