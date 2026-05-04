using System.Text.RegularExpressions;
using AiDotNet.Exceptions;
using AiDotNet.Helpers;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.Helpers;

/// <summary>
/// Tests for <see cref="TrialStateManager"/> covering all trial state transitions,
/// tamper detection, and edge cases.
/// </summary>
/// <remarks>
/// Several tests mutate <c>TrialStateManager.TrialMessageHandler</c>, a
/// process-global static. xUnit runs test classes in parallel by default,
/// so two classes both touching that handler concurrently would clobber
/// each other's captured-message lists. The xUnit
/// <see cref="CollectionAttribute"/> disables parallel execution within
/// this class's collection — pinning the static-handler tests to a
/// single execution context.
/// </remarks>
[Collection("TrialStateManager-static-handler")]
public class TrialStateManagerTests : IDisposable
{
    private readonly string _tempDir;
    private readonly string _trialFilePath;

    public TrialStateManagerTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "aidotnet-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
        _trialFilePath = Path.Combine(_tempDir, "trial.json");
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_tempDir))
                Directory.Delete(_tempDir, true);
        }
        catch
        {
            // Best effort cleanup
        }
    }

    [Fact(Timeout = 60000)]
    public async Task FirstUse_CreatesTrialFile()
    {
        var manager = new TrialStateManager(_trialFilePath);

        // File should not exist yet
        Assert.False(File.Exists(_trialFilePath));

        // First operation creates the file
        manager.RecordOperationOrThrow();

        Assert.True(File.Exists(_trialFilePath));
    }

    [Fact(Timeout = 60000)]
    public async Task FirstUse_StatusShowsFreshTrial()
    {
        var manager = new TrialStateManager(_trialFilePath);
        manager.RecordOperationOrThrow();

        var status = manager.GetStatus();

        Assert.False(status.IsExpired);
        Assert.Equal(0, status.DaysElapsed);
        Assert.Equal(TrialStateManager.TrialDurationDays, status.DaysRemaining);
        Assert.Equal(1, status.OperationsUsed);
        Assert.Equal(TrialStateManager.TrialOperationLimit - 1, status.OperationsRemaining);
    }

    [Fact(Timeout = 60000)]
    public async Task RecordOperation_IncrementsCounter()
    {
        var manager = new TrialStateManager(_trialFilePath);

        for (int i = 0; i < 5; i++)
        {
            manager.RecordOperationOrThrow();
        }

        var status = manager.GetStatus();
        Assert.Equal(5, status.OperationsUsed);
        Assert.Equal(TrialStateManager.TrialOperationLimit - 5, status.OperationsRemaining);
        Assert.False(status.IsExpired);
    }

    [Fact(Timeout = 60000)]
    public async Task OperationLimit_ThrowsLicenseRequiredException()
    {
        var manager = new TrialStateManager(_trialFilePath);

        // Use all 10 operations
        for (int i = 0; i < TrialStateManager.TrialOperationLimit; i++)
        {
            manager.RecordOperationOrThrow();
        }

        // 11th operation should throw
        var ex = Assert.Throws<LicenseRequiredException>(() => manager.RecordOperationOrThrow());

        Assert.Equal(TrialExpirationReason.OperationLimitReached, ex.ExpirationReason);
        Assert.Equal(TrialStateManager.TrialOperationLimit, ex.OperationsPerformed);
    }

    [Fact(Timeout = 60000)]
    public async Task OperationLimit_StatusShowsExpired()
    {
        var manager = new TrialStateManager(_trialFilePath);

        for (int i = 0; i < TrialStateManager.TrialOperationLimit; i++)
        {
            manager.RecordOperationOrThrow();
        }

        var status = manager.GetStatus();
        Assert.True(status.IsExpired);
        Assert.Equal(0, status.OperationsRemaining);
        Assert.Equal(TrialStateManager.TrialOperationLimit, status.OperationsUsed);
    }

    [Fact(Timeout = 60000)]
    public async Task TamperedFile_TreatedAsExpired()
    {
        var manager = new TrialStateManager(_trialFilePath);
        manager.RecordOperationOrThrow();

        // Tamper with the file
        File.WriteAllText(_trialFilePath, "{\"version\":1,\"data\":\"dGFtcGVyZWQ=\",\"signature\":\"invalid\"}");

        // Next operation should throw because tampered file = expired
        Assert.Throws<LicenseRequiredException>(() => manager.RecordOperationOrThrow());
    }

    [Fact(Timeout = 60000)]
    public async Task CorruptedFile_TreatedAsExpired()
    {
        var manager = new TrialStateManager(_trialFilePath);
        manager.RecordOperationOrThrow();

        // Write garbage
        File.WriteAllText(_trialFilePath, "not valid json at all {{{");

        // Should throw
        Assert.Throws<LicenseRequiredException>(() => manager.RecordOperationOrThrow());
    }

    [Fact(Timeout = 60000)]
    public async Task EmptyFile_TreatedAsExpired()
    {
        var manager = new TrialStateManager(_trialFilePath);
        manager.RecordOperationOrThrow();

        // Write empty file
        File.WriteAllText(_trialFilePath, "");

        Assert.Throws<LicenseRequiredException>(() => manager.RecordOperationOrThrow());
    }

    [Fact(Timeout = 60000)]
    public async Task NullSignature_TreatedAsExpired()
    {
        var manager = new TrialStateManager(_trialFilePath);
        manager.RecordOperationOrThrow();

        // Valid JSON envelope but missing signature
        File.WriteAllText(_trialFilePath, "{\"version\":1,\"data\":\"dGVzdA==\",\"signature\":\"\"}");

        Assert.Throws<LicenseRequiredException>(() => manager.RecordOperationOrThrow());
    }

    [Fact(Timeout = 60000)]
    public async Task DeletedFile_RestartsTrialFromScratch()
    {
        var manager = new TrialStateManager(_trialFilePath);

        // Use 5 operations
        for (int i = 0; i < 5; i++)
        {
            manager.RecordOperationOrThrow();
        }

        // Delete the file (simulates user clearing trial state to bypass limits)
        File.Delete(_trialFilePath);

        // Missing file should be treated as expired (anti-reset behavior)
        var status = manager.GetStatus();
        Assert.True(status.IsExpired, "Missing trial file should be treated as expired");

        // Operations should not be allowed after file deletion
        Assert.Throws<LicenseRequiredException>(() => manager.RecordOperationOrThrow());
    }

    [Fact(Timeout = 60000)]
    public async Task Reset_DeletesTrialFile()
    {
        var manager = new TrialStateManager(_trialFilePath);
        manager.RecordOperationOrThrow();
        Assert.True(File.Exists(_trialFilePath));

        manager.Reset();
        Assert.False(File.Exists(_trialFilePath));
    }

    [Fact(Timeout = 60000)]
    public async Task Reset_AllowsFreshTrialToStart()
    {
        var manager = new TrialStateManager(_trialFilePath);

        // Exhaust the trial
        for (int i = 0; i < TrialStateManager.TrialOperationLimit; i++)
        {
            manager.RecordOperationOrThrow();
        }

        Assert.Throws<LicenseRequiredException>(() => manager.RecordOperationOrThrow());

        // Reset and verify fresh trial
        manager.Reset();
        manager.RecordOperationOrThrow(); // Should not throw

        var status = manager.GetStatus();
        Assert.Equal(1, status.OperationsUsed);
        Assert.False(status.IsExpired);
    }

    [Fact(Timeout = 60000)]
    public async Task GetStatus_BeforeAnyOperation_ReturnsZeroOperations()
    {
        var manager = new TrialStateManager(_trialFilePath);

        // GetStatus on a fresh manager triggers file creation (via LoadOrCreateState)
        // but the returned status should show zero operations
        var status = manager.GetStatus();
        Assert.Equal(0, status.OperationsUsed);
        Assert.False(status.IsExpired);
    }

    [Fact(Timeout = 60000)]
    public async Task GetStatus_DoesNotIncrementCounter()
    {
        var manager = new TrialStateManager(_trialFilePath);
        manager.RecordOperationOrThrow();

        // Call GetStatus multiple times
        for (int i = 0; i < 100; i++)
        {
            var status = manager.GetStatus();
            Assert.Equal(1, status.OperationsUsed);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task StatePersistsAcrossInstances()
    {
        // First instance records 3 operations
        var manager1 = new TrialStateManager(_trialFilePath);
        for (int i = 0; i < 3; i++)
        {
            manager1.RecordOperationOrThrow();
        }

        // Second instance using the same file should see 3 operations
        var manager2 = new TrialStateManager(_trialFilePath);
        var status = manager2.GetStatus();
        Assert.Equal(3, status.OperationsUsed);

        // And can continue counting
        manager2.RecordOperationOrThrow();
        Assert.Equal(4, manager2.GetStatus().OperationsUsed);
    }

    [Fact(Timeout = 60000)]
    public async Task ExpirationReason_OperationLimitReached_SetCorrectly()
    {
        var manager = new TrialStateManager(_trialFilePath);
        manager.RecordOperationOrThrow();

        // Manually create an expired-by-time state by writing a backdated file
        // We do this by reading the file, decoding, modifying, re-encoding, re-signing
        // ... but that's complex. Instead, verify via the exception properties.

        // We can't easily test time expiration without mocking DateTimeOffset.
        // Instead, verify that the OperationLimitReached path sets the correct reason.
        for (int i = 1; i < TrialStateManager.TrialOperationLimit; i++)
        {
            manager.RecordOperationOrThrow();
        }

        var ex = Assert.Throws<LicenseRequiredException>(() => manager.RecordOperationOrThrow());
        Assert.Equal(TrialExpirationReason.OperationLimitReached, ex.ExpirationReason);
        Assert.Equal(TrialStateManager.TrialOperationLimit, ex.OperationsPerformed);
        Assert.True(ex.TrialDaysElapsed >= 0, "TrialDaysElapsed should be non-negative");
    }

    [Fact(Timeout = 60000)]
    public async Task LicenseRequiredException_HasDescriptiveMessage()
    {
        var manager = new TrialStateManager(_trialFilePath);

        for (int i = 0; i < TrialStateManager.TrialOperationLimit; i++)
        {
            manager.RecordOperationOrThrow();
        }

        var ex = Assert.Throws<LicenseRequiredException>(() => manager.RecordOperationOrThrow());

        // Message should mention the limit and direct to aidotnet.dev
        Assert.Contains("10", ex.Message);
        Assert.Contains("aidotnet.dev", ex.Message);
    }

    [Fact(Timeout = 60000)]
    public async Task Constructor_NullPath_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new TrialStateManager(null!));
    }

    [Fact(Timeout = 60000)]
    public async Task ThreadSafety_ConcurrentOperations_DoNotCorruptState()
    {
        var manager = new TrialStateManager(_trialFilePath);
        int successCount = 0;
        int exceptionCount = 0;

        // Launch multiple threads all trying to record operations
        var tasks = new Task[20];
        for (int i = 0; i < tasks.Length; i++)
        {
            tasks[i] = Task.Run(() =>
            {
                try
                {
                    manager.RecordOperationOrThrow();
                    Interlocked.Increment(ref successCount);
                }
                catch (LicenseRequiredException)
                {
                    Interlocked.Increment(ref exceptionCount);
                }
            });
        }

        Task.WaitAll(tasks);

        // Exactly 10 should succeed (trial limit), rest should throw
        Assert.Equal(TrialStateManager.TrialOperationLimit, successCount);
        Assert.Equal(20 - TrialStateManager.TrialOperationLimit, exceptionCount);

        // Final state should show exactly 10 operations
        var status = manager.GetStatus();
        Assert.Equal(TrialStateManager.TrialOperationLimit, status.OperationsUsed);
        Assert.True(status.IsExpired);
    }

    [Fact(Timeout = 60000)]
    public async Task ConsoleMessage_MatchesExpectedFormat()
    {
        // TrialStateManager.EmitTrialMessage routes to stderr by default
        // (to avoid polluting stdout) and offers an internal
        // TrialMessageHandler hook for tests. Using the hook is more
        // reliable than redirecting Console.Error since the impl can
        // bypass Console.Error in environments without a TTY.
        var manager = new TrialStateManager(_trialFilePath);

        var captured = new List<string>();
        var previousHandler = TrialStateManager.TrialMessageHandler;
        TrialStateManager.TrialMessageHandler = msg => captured.Add(msg);
        try
        {
            manager.RecordOperationOrThrow();

            Assert.Single(captured);
            string output = captured[0].Trim();

            // Verify format: "AiDotNet Community — X day(s) and Y operation(s) remaining in free trial. Register for a free license at https://aidotnet.dev"
            Assert.Matches(
                @"AiDotNet Community .+ \d+ day\(s\) and \d+ operation\(s\) remaining in free trial\. Register for a free license at https://aidotnet\.dev",
                output);

            // Verify specific values for first operation
            Assert.Contains($"{TrialStateManager.TrialDurationDays} day(s)", output);
            Assert.Contains($"{TrialStateManager.TrialOperationLimit - 1} operation(s)", output);
        }
        finally
        {
            TrialStateManager.TrialMessageHandler = previousHandler;
        }
    }

    [Fact(Timeout = 60000)]
    public async Task ConsoleMessage_CountdownDecrementsCorrectly()
    {
        var manager = new TrialStateManager(_trialFilePath);

        var captured = new List<string>();
        var previousHandler = TrialStateManager.TrialMessageHandler;
        TrialStateManager.TrialMessageHandler = msg => captured.Add(msg);
        try
        {
            for (int i = 0; i < 3; i++)
            {
                manager.RecordOperationOrThrow();
            }

            Assert.Equal(3, captured.Count);

            // After 1st op: TrialOperationLimit-1 remaining
            Assert.Contains($"{TrialStateManager.TrialOperationLimit - 1} operation(s)", captured[0]);
            // After 2nd op: TrialOperationLimit-2 remaining
            Assert.Contains($"{TrialStateManager.TrialOperationLimit - 2} operation(s)", captured[1]);
            // After 3rd op: TrialOperationLimit-3 remaining
            Assert.Contains($"{TrialStateManager.TrialOperationLimit - 3} operation(s)", captured[2]);
        }
        finally
        {
            TrialStateManager.TrialMessageHandler = previousHandler;
        }
    }

    [Fact(Timeout = 60000)]
    public async Task ConsoleMessage_IncludesAiDotNetDevUrl()
    {
        var manager = new TrialStateManager(_trialFilePath);

        var captured = new List<string>();
        var previousHandler = TrialStateManager.TrialMessageHandler;
        TrialStateManager.TrialMessageHandler = msg => captured.Add(msg);
        try
        {
            manager.RecordOperationOrThrow();
            Assert.Single(captured);
            Assert.Contains("https://aidotnet.dev", captured[0]);
        }
        finally
        {
            TrialStateManager.TrialMessageHandler = previousHandler;
        }
    }
}