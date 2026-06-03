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
    public override void Before(MethodInfo methodUnderTest)
    {
        // Unique per test; the file itself is created lazily by TrialStateManager
        // only if the test actually performs a save/load operation.
        string path = Path.Combine(
            Path.GetTempPath(),
            "aidotnet-trial-tests",
            Guid.NewGuid().ToString("N") + ".json");
        ModelPersistenceGuard.SetTestTrialFilePathOverride(path);
    }

    public override void After(MethodInfo methodUnderTest)
    {
        string? path = ModelPersistenceGuard.CurrentTestTrialFilePath;
        ModelPersistenceGuard.SetTestTrialFilePathOverride(null);
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
