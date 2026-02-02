#if !NET462 && !NET471
using System.Runtime.CompilerServices;
#endif
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Tests;

/// <summary>
/// Module initializer that runs before any tests to configure the test environment.
/// </summary>
internal static class TestModuleInitializer
{
    private static bool _initialized;

#if !NET462 && !NET471
    /// <summary>
    /// Called automatically by the runtime before any code in this assembly runs.
    /// Configures CPU-only mode to prevent GPU-related test failures.
    /// </summary>
    [ModuleInitializer]
    internal static void Initialize()
    {
        InitializeCpuMode();
    }
#endif

    /// <summary>
    /// Ensures CPU-only mode is initialized. Can be called from test classes on older frameworks.
    /// </summary>
    public static void EnsureInitialized()
    {
        InitializeCpuMode();
    }

    private static void InitializeCpuMode()
    {
        if (_initialized) return;
        _initialized = true;

        // Force CPU-only mode for all tests
        // This prevents GPU/OpenCL errors on systems without proper GPU support
        try
        {
            AiDotNetEngine.ResetToCpu();
        }
        catch
        {
            // Ignore errors during initialization - CPU mode may already be active
        }
    }
}
