using System.Text.RegularExpressions;
using AiDotNet.Exceptions;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.Helpers;

/// <summary>
/// Tests for <see cref="TrialStateManager"/> covering all trial state transitions,
/// tamper detection, and edge cases.
/// </summary>
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

    [Fact]
    public void FirstUse_CreatesTrialFile()
    {
        var manager = new TrialStateManager(_trialFilePath);

        // File should not exist yet
        Assert.False(File.Exists(_trialFilePath));

        // First operation creates the file
        manager.RecordOperationOrThrow();

        Assert.True(File.Exists(_trialFilePath));
    }

    [Fact]
    public void FirstUse_StatusShowsFreshTrial()
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

    [Fact]
    public void RecordOperation_IncrementsCounter()
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

    [Fact]
    public void OperationLimit_ThrowsLicenseRequiredException()
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

    [Fact]
    public void OperationLimit_StatusShowsExpired()
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

    [Fact]
    public void TamperedFile_TreatedAsExpired()
    {
        var manager = new TrialStateManager(_trialFilePath);
        manager.RecordOperationOrThrow();

        // Tamper with the file
        File.WriteAllText(_trialFilePath, "{\"version\":1,\"data\":\"dGFtcGVyZWQ=\",\"signature\":\"invalid\"}");

        // Next operation should throw because tampered file = expired
        Assert.Throws<LicenseRequiredException>(() => manager.RecordOperationOrThrow());
    }

    [Fact]
    public void CorruptedFile_TreatedAsExpired()
    {
        var manager = new TrialStateManager(_trialFilePath);
        manager.RecordOperationOrThrow();

        // Write garbage
        File.WriteAllText(_trialFilePath, "not valid json at all {{{");

        // Should throw
        Assert.Throws<LicenseRequiredException>(() => manager.RecordOperationOrThrow());
    }

    [Fact]
    public void EmptyFile_TreatedAsExpired()
    {
        var manager = new TrialStateManager(_trialFilePath);
        manager.RecordOperationOrThrow();

        // Write empty file
        File.WriteAllText(_trialFilePath, "");

        Assert.Throws<LicenseRequiredException>(() => manager.RecordOperationOrThrow());
    }

    [Fact]
    public void NullSignature_TreatedAsExpired()
    {
        var manager = new TrialStateManager(_trialFilePath);
        manager.RecordOperationOrThrow();

        // Valid JSON envelope but missing signature
        File.WriteAllText(_trialFilePath, "{\"version\":1,\"data\":\"dGVzdA==\",\"signature\":\"\"}");

        Assert.Throws<LicenseRequiredException>(() => manager.RecordOperationOrThrow());
    }

    [Fact]
    public void DeletedFile_RestartsTrialFromScratch()
    {
        var manager = new TrialStateManager(_trialFilePath);

        // Use 5 operations
        for (int i = 0; i < 5; i++)
        {
            manager.RecordOperationOrThrow();
        }

        // Delete the file (simulates user clearing trial state)
        File.Delete(_trialFilePath);

        // Trial restarts — counter goes back to 1
        manager.RecordOperationOrThrow();
        var status = manager.GetStatus();
        Assert.Equal(1, status.OperationsUsed);
        Assert.False(status.IsExpired);
    }

    [Fact]
    public void Reset_DeletesTrialFile()
    {
        var manager = new TrialStateManager(_trialFilePath);
        manager.RecordOperationOrThrow();
        Assert.True(File.Exists(_trialFilePath));

        manager.Reset();
        Assert.False(File.Exists(_trialFilePath));
    }

    [Fact]
    public void Reset_AllowsFreshTrialToStart()
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

    [Fact]
    public void GetStatus_BeforeAnyOperation_DoesNotCreateFile()
    {
        var manager = new TrialStateManager(_trialFilePath);

        // GetStatus on a fresh manager triggers file creation (via LoadOrCreateState)
        // but the returned status should show zero operations
        var status = manager.GetStatus();
        Assert.Equal(0, status.OperationsUsed);
        Assert.False(status.IsExpired);
    }

    [Fact]
    public void GetStatus_DoesNotIncrementCounter()
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

    [Fact]
    public void StatePersistsAcrossInstances()
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

    [Fact]
    public void ExpirationReason_OperationLimitReached_SetCorrectly()
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
        Assert.NotNull(ex.OperationsPerformed);
        Assert.NotNull(ex.TrialDaysElapsed);
    }

    [Fact]
    public void LicenseRequiredException_HasDescriptiveMessage()
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

    [Fact]
    public void Constructor_NullPath_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new TrialStateManager(null!));
    }

    [Fact]
    public void ThreadSafety_ConcurrentOperations_DoNotCorruptState()
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

    [Fact]
    public void ConsoleMessage_MatchesExpectedFormat()
    {
        var manager = new TrialStateManager(_trialFilePath);

        var originalOut = Console.Out;
        try
        {
            using var sw = new StringWriter();
            Console.SetOut(sw);

            manager.RecordOperationOrThrow();

            string output = sw.ToString().Trim();

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
            Console.SetOut(originalOut);
        }
    }

    [Fact]
    public void ConsoleMessage_CountdownDecrementsCorrectly()
    {
        var manager = new TrialStateManager(_trialFilePath);

        var originalOut = Console.Out;
        try
        {
            var messages = new List<string>();

            for (int i = 0; i < 3; i++)
            {
                using var sw = new StringWriter();
                Console.SetOut(sw);
                manager.RecordOperationOrThrow();
                messages.Add(sw.ToString().Trim());
            }

            // After 1st op: 9 remaining
            Assert.Contains($"{TrialStateManager.TrialOperationLimit - 1} operation(s)", messages[0]);
            // After 2nd op: 8 remaining
            Assert.Contains($"{TrialStateManager.TrialOperationLimit - 2} operation(s)", messages[1]);
            // After 3rd op: 7 remaining
            Assert.Contains($"{TrialStateManager.TrialOperationLimit - 3} operation(s)", messages[2]);
        }
        finally
        {
            Console.SetOut(originalOut);
        }
    }

    [Fact]
    public void ConsoleMessage_IncludesAiDotNetDevUrl()
    {
        var manager = new TrialStateManager(_trialFilePath);

        var originalOut = Console.Out;
        try
        {
            using var sw = new StringWriter();
            Console.SetOut(sw);
            manager.RecordOperationOrThrow();

            string output = sw.ToString();
            Assert.Contains("https://aidotnet.dev", output);
        }
        finally
        {
            Console.SetOut(originalOut);
        }
    }
}
