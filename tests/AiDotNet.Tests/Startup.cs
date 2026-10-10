using Xunit.Abstractions;
#if AIDOTNET_TEST_ATTRIBUTION
using TestFrameworkBase = AiDotNet.TestImpact.Xunit.AttributionTestFramework;
#else
using TestFrameworkBase = Xunit.Sdk.XunitTestFramework;
#endif

[assembly: Xunit.TestFramework("AiDotNet.Tests.CpuOnlyTestFramework", "AiDotNetTests")]

namespace AiDotNet.Tests;

/// <summary>
/// Custom test framework that initializes CPU-only mode before any tests run.
/// </summary>
public class CpuOnlyTestFramework : TestFrameworkBase
{
    public CpuOnlyTestFramework(IMessageSink messageSink)
        : base(messageSink)
    {
        // Initialize CPU-only mode before any tests run
        TestModuleInitializer.EnsureInitialized();
    }
#if !AIDOTNET_TEST_ATTRIBUTION

    /// <summary>
    /// Runs tests through <see cref="TestInfrastructure.HeavyTimeoutScalingExecutor"/>, which lengthens HeavyTimeout tests'
    /// per-test timeout only when the nightly heavy lane sets its scale variable.
    /// </summary>
    protected override Xunit.Abstractions.ITestFrameworkExecutor CreateExecutor(System.Reflection.AssemblyName assemblyName)
        => new TestInfrastructure.HeavyTimeoutScalingExecutor(assemblyName, SourceInformationProvider, DiagnosticMessageSink);
#endif
}
