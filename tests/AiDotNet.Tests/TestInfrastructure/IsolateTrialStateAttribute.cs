using System.Reflection;
using AiDotNet.Helpers;
using Xunit.Sdk;

// Apply to every test in the assembly so each one gets its own isolated free-trial
// file. Without this, the process-global trial counter (~/.aidotnet/trial.json,
// 10 save/load ops) is shared across ~90 save/load test files that run in parallel
// across xUnit collections, so they pollute each other's counter and throw
// LicenseRequiredException nondeterministically.
[assembly: AiDotNet.Tests.TestInfrastructure.IsolateTrialState]

namespace AiDotNet.Tests.TestInfrastructure;

/// <summary>
/// Gives each test its own free-trial state file via the AsyncLocal-backed
/// <see cref="ModelPersistenceGuard.SetTestTrialFilePathOverride"/>. Because the
/// override flows with the test's execution context (not a process-global), parallel
/// tests get independent trial counters and never race on the shared trial file.
/// </summary>
/// <remarks>
/// Tests that need to drive specific trial state (the licensing tests) read
/// <see cref="ModelPersistenceGuard.CurrentTestTrialFilePath"/> and manipulate the
/// same isolated file the guard reads.
/// </remarks>
[AttributeUsage(AttributeTargets.Assembly | AttributeTargets.Class | AttributeTargets.Method, AllowMultiple = false)]
public sealed class IsolateTrialStateAttribute : BeforeAfterTestAttribute
{
    // The assembly-level attribute is a single instance shared across all parallel
    // test invocations, so the IDisposable scope returned by SetTestTrialFilePathOverride
    // must NOT live on a plain field (that would race across threads). Use AsyncLocal
    // so each test's execution context carries its own scope — matching the AsyncLocal
    // backing of the override itself.
    private static readonly AsyncLocal<IDisposable?> _trialPathScope = new();

    public override void Before(MethodInfo methodUnderTest)
    {
        // Unique per test; the file itself is created lazily by TrialStateManager
        // only if the test actually performs a save/load operation.
        string path = Path.Combine(
            Path.GetTempPath(),
            "aidotnet-trial-tests",
            Guid.NewGuid().ToString("N") + ".json");
        _trialPathScope.Value = ModelPersistenceGuard.SetTestTrialFilePathOverride(path);
    }

    public override void After(MethodInfo methodUnderTest)
    {
        string? path = ModelPersistenceGuard.CurrentTestTrialFilePath;
        // Restore the previous override via the scope's Dispose contract instead of
        // a blunt SetTestTrialFilePathOverride(null), so nested overrides (or future
        // multi-level usages) correctly unwind one level rather than wiping all.
        _trialPathScope.Value?.Dispose();
        _trialPathScope.Value = null;
        TryDelete(path);
        TryDelete(path is null ? null : path + ".tombstone");
    }

    private static void TryDelete(string? path)
    {
        if (string.IsNullOrEmpty(path))
        {
            return;
        }

        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            // Best-effort temp cleanup; a leftover temp trial file is harmless.
        }
    }
}
