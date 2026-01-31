using Xunit;

namespace AiDotNet.Tests.Fixtures;

/// <summary>
/// Assembly fixture that configures CPU-only mode for all tests.
/// This prevents GPU-related test failures on systems without proper GPU support.
/// </summary>
public class CpuOnlyFixture : IDisposable
{
    public CpuOnlyFixture()
    {
        // Ensure CPU-only mode is initialized (especially important for net471)
        TestModuleInitializer.EnsureInitialized();
    }

    public void Dispose()
    {
        // Nothing to dispose - CPU mode remains for the lifetime of the test run
    }
}

/// <summary>
/// Collection definition that applies the CPU-only fixture to all tests in this collection.
/// </summary>
[CollectionDefinition("CpuOnly", DisableParallelization = true)]
public class CpuOnlyCollection : ICollectionFixture<CpuOnlyFixture>
{
    // This class has no code, and is never created.
    // Its purpose is simply to be the place to apply [CollectionDefinition].
}
