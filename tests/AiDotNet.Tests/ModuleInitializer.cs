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

        // Pin BLAS backends to a single thread per xUnit worker. Without this,
        // MKL / OpenBLAS / OMP default to spinning up ProcessorCount threads
        // PER xUnit worker thread — on a 32-core box with maxParallelThreads=0
        // that is 32×32 = 1024 contending threads, and matmul-heavy tests
        // (ResNet, layer-serialization roundtrips, diffusion forward passes)
        // starve each other out and hit their [Fact(Timeout=30000)] ceiling.
        // Pinning BLAS to 1 thread makes test-level xUnit parallelism the
        // *only* source of concurrency — measured to cut flaky layer-test
        // timeouts from 30 to 4 on a 16-core/32-thread Windows box and to
        // leave BLAS-heavy shards (NN-Classic) slightly faster. See #1166.
        //
        // Must be set BEFORE the native BLAS DLL first resolves, which
        // happens on the first matmul dispatch. [ModuleInitializer] runs
        // on first reference to this assembly — i.e. before any test
        // method body and so before any BLAS call. On net471 the
        // CpuOnlyFixture ctor also calls EnsureInitialized() which flows
        // here; that path is best-effort since the collection fixture may
        // run after MKL is already pinned from an earlier process event,
        // but MKL honours the env var on re-read via mkl_set_num_threads_local.
        System.Environment.SetEnvironmentVariable("OMP_NUM_THREADS",      "1");
        System.Environment.SetEnvironmentVariable("MKL_NUM_THREADS",      "1");
        System.Environment.SetEnvironmentVariable("OPENBLAS_NUM_THREADS", "1");

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

        // Set a default license key for all tests so that serialize/deserialize/save/load
        // tests don't depend on trial state. Tests that explicitly test trial behavior
        // (e.g., ModelPersistenceGuardTests, AiModelBuilderLicensingTests) clear this
        // in their own setup via ClearAllLicenseSources().
        //
        // Injected UNLESS a real OFFLINE-verifiable key is already configured. A key that validates
        // offline — v2 `aidn2` (asymmetric) or legacy `aidn.` (HMAC) — lifts the persistence cap
        // deterministically without the network, so we keep it (this is the CI `aidn2` key path) and must
        // NOT clobber its build key. But a MISSING key, or a server-only `AIDN-*` key, would force the
        // entire save/load-heavy suite onto the online license server every operation AND — for a
        // community-tier key under capability-authoritative gating — fail to lift the save cap, draining
        // the free-trial op budget until persistence tests throw LicenseRequiredException. In those cases
        // install a synthetic offline license so the suite is deterministically licensed; the real
        // online/community-key path is exercised explicitly by LicenseE2ETests, not the whole suite.
        string? _envLicenseKey = System.Environment.GetEnvironmentVariable("AIDOTNET_LICENSE_KEY");
        bool _envKeyOfflineUsable = !string.IsNullOrWhiteSpace(_envLicenseKey)
            && (AiDotNet.Helpers.AsymmetricLicenseVerifier.IsAsymmetricKeyFormat(_envLicenseKey)
                || AiDotNet.Helpers.LicenseValidator.IsSignedKeyFormat(_envLicenseKey));
        if (!_envKeyOfflineUsable)
        {
            // Inject the test build key process-wide so offline license validation
            // (HMAC-SHA256 of the key against the embedded build key, per the
            // ValidateOffline hardening) can actually VERIFY the default key below.
            // This mirrors LicenseBuildKeyFixture but applies globally, so
            // serialize/persistence-gated tests in EVERY collection get a valid
            // license rather than failing closed with LicenseRequiredException.
            // Tests needing a different build key scope it via
            // LicenseTestSupport.WithBuildKey (save/restore); trial tests clear the
            // license key via ClearAllLicenseSources.
            AiDotNet.Helpers.BuildKeyProvider.OverrideForTesting(Helpers.LicenseTestSupport.TestBuildKey);

            // A REAL signed key — sig = base64url(HMACSHA256(testBuildKey, "aidn.testdefault1")) —
            // so ValidateOffline returns Active. The previous hardcoded
            // "aidn.testdefault1.testsignature1" carried a bogus signature that the
            // hardened offline validator (correctly) rejects as Invalid, which made
            // every serialize-gated build throw LicenseRequiredException.
            System.Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY",
                Helpers.LicenseTestSupport.SignedKey("testdefault1"));
        }
    }
}
