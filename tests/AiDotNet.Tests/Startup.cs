using Xunit.Abstractions;
using Xunit.Sdk;

[assembly: Xunit.TestFramework("AiDotNet.Tests.CpuOnlyTestFramework", "AiDotNetTests")]

namespace AiDotNet.Tests;

/// <summary>
/// Custom test framework that initializes CPU-only mode before any tests run.
/// </summary>
public class CpuOnlyTestFramework : XunitTestFramework
{
    public CpuOnlyTestFramework(IMessageSink messageSink)
        : base(messageSink)
    {
        // Initialize CPU-only mode before any tests run
        TestModuleInitializer.EnsureInitialized();
    }
}
