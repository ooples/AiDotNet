using AiDotNet.Exceptions;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.Helpers;

/// <summary>
/// Tests for <see cref="ModelPersistenceGuard"/> covering enforcement behavior,
/// internal operation suppression, license key bypass, and thread isolation.
/// </summary>
/// <remarks>
/// These tests manipulate the AIDOTNET_LICENSE_KEY environment variable and
/// the default trial file. Each test restores state in its Dispose method.
/// Tests run sequentially via [Collection] to avoid env var races.
/// </remarks>
[Collection("LicensingTests")]
public class ModelPersistenceGuardTests : IDisposable
{
    private readonly string _tempDir;
    private readonly string _trialFilePath;
    private readonly string? _originalEnvVar;
    private readonly string? _originalLicenseFile;
    private readonly string _licenseFilePath;

    public ModelPersistenceGuardTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), "aidotnet-guard-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tempDir);
        _trialFilePath = Path.Combine(_tempDir, "trial.json");

        // Preserve original env var
        _originalEnvVar = Environment.GetEnvironmentVariable("AIDOTNET_LICENSE_KEY");

        // Preserve original license file if it exists
        string homeDir = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        _licenseFilePath = Path.Combine(homeDir, ".aidotnet", "license.key");
        _originalLicenseFile = File.Exists(_licenseFilePath) ? File.ReadAllText(_licenseFilePath) : null;
    }

    public void Dispose()
    {
        // Restore original env var
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", _originalEnvVar);

        // Restore original license file state
        if (_originalLicenseFile != null)
        {
            File.WriteAllText(_licenseFilePath, _originalLicenseFile);
        }
        else if (File.Exists(_licenseFilePath))
        {
            File.Delete(_licenseFilePath);
        }

        // Clean up temp dir
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

    /// <summary>
    /// Clears all license sources so enforcement falls through to trial counting.
    /// </summary>
    private void ClearAllLicenseSources()
    {
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", null);
        if (File.Exists(_licenseFilePath))
        {
            File.Delete(_licenseFilePath);
        }
    }

    /// <summary>
    /// Resets the default trial file to allow fresh trial counting.
    /// </summary>
    private void ResetDefaultTrial()
    {
        var manager = new TrialStateManager();
        manager.Reset();
    }

    [Fact]
    public void EnforceBeforeSave_WithLicenseKey_DoesNotThrow()
    {
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", "aidn.testkey1234.abcdefghijklmnop");

        // Should not throw — license key present bypasses trial
        ModelPersistenceGuard.EnforceBeforeSave();
    }

    [Fact]
    public void EnforceBeforeLoad_WithLicenseKey_DoesNotThrow()
    {
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", "aidn.testkey1234.abcdefghijklmnop");

        ModelPersistenceGuard.EnforceBeforeLoad();
    }

    [Fact]
    public void EnforceBeforeSerialize_WithLicenseKey_DoesNotThrow()
    {
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", "aidn.testkey1234.abcdefghijklmnop");

        ModelPersistenceGuard.EnforceBeforeSerialize();
    }

    [Fact]
    public void EnforceBeforeDeserialize_WithLicenseKey_DoesNotThrow()
    {
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", "aidn.testkey1234.abcdefghijklmnop");

        ModelPersistenceGuard.EnforceBeforeDeserialize();
    }

    [Fact]
    public void EnforceBeforeSave_WithoutLicense_CountsTrialOperation()
    {
        ClearAllLicenseSources();
        ResetDefaultTrial();

        // First call should succeed (trial is fresh)
        ModelPersistenceGuard.EnforceBeforeSave();

        // Verify trial counter was incremented
        var manager = new TrialStateManager();
        var status = manager.GetStatus();
        Assert.True(status.OperationsUsed >= 1);
    }

    [Fact]
    public void EnforceBeforeLoad_WithoutLicense_CountsTrialOperation()
    {
        ClearAllLicenseSources();
        ResetDefaultTrial();

        ModelPersistenceGuard.EnforceBeforeLoad();

        var manager = new TrialStateManager();
        var status = manager.GetStatus();
        Assert.True(status.OperationsUsed >= 1);
    }

    [Fact]
    public void EnforceBeforeSave_ExhaustedTrial_ThrowsLicenseRequiredException()
    {
        ClearAllLicenseSources();
        ResetDefaultTrial();

        // Exhaust the trial
        var manager = new TrialStateManager();
        for (int i = 0; i < TrialStateManager.TrialOperationLimit; i++)
        {
            manager.RecordOperationOrThrow();
        }

        // Next enforcement should throw
        Assert.Throws<LicenseRequiredException>(() => ModelPersistenceGuard.EnforceBeforeSave());
    }

    [Fact]
    public void EnforceBeforeLoad_ExhaustedTrial_ThrowsLicenseRequiredException()
    {
        ClearAllLicenseSources();
        ResetDefaultTrial();

        var manager = new TrialStateManager();
        for (int i = 0; i < TrialStateManager.TrialOperationLimit; i++)
        {
            manager.RecordOperationOrThrow();
        }

        Assert.Throws<LicenseRequiredException>(() => ModelPersistenceGuard.EnforceBeforeLoad());
    }

    [Fact]
    public void EnforceBeforeSerialize_ExhaustedTrial_ThrowsLicenseRequiredException()
    {
        ClearAllLicenseSources();
        ResetDefaultTrial();

        var manager = new TrialStateManager();
        for (int i = 0; i < TrialStateManager.TrialOperationLimit; i++)
        {
            manager.RecordOperationOrThrow();
        }

        Assert.Throws<LicenseRequiredException>(() => ModelPersistenceGuard.EnforceBeforeSerialize());
    }

    [Fact]
    public void EnforceBeforeDeserialize_ExhaustedTrial_ThrowsLicenseRequiredException()
    {
        ClearAllLicenseSources();
        ResetDefaultTrial();

        var manager = new TrialStateManager();
        for (int i = 0; i < TrialStateManager.TrialOperationLimit; i++)
        {
            manager.RecordOperationOrThrow();
        }

        Assert.Throws<LicenseRequiredException>(() => ModelPersistenceGuard.EnforceBeforeDeserialize());
    }

    [Fact]
    public void InternalOperation_SuppressesSerializeEnforcement()
    {
        ClearAllLicenseSources();
        ResetDefaultTrial();

        // Exhaust the trial
        var manager = new TrialStateManager();
        for (int i = 0; i < TrialStateManager.TrialOperationLimit; i++)
        {
            manager.RecordOperationOrThrow();
        }

        // Within InternalOperation scope, Serialize enforcement is suppressed
        using (ModelPersistenceGuard.InternalOperation())
        {
            // Should NOT throw even though trial is exhausted
            ModelPersistenceGuard.EnforceBeforeSerialize();
        }
    }

    [Fact]
    public void InternalOperation_SuppressesDeserializeEnforcement()
    {
        ClearAllLicenseSources();
        ResetDefaultTrial();

        var manager = new TrialStateManager();
        for (int i = 0; i < TrialStateManager.TrialOperationLimit; i++)
        {
            manager.RecordOperationOrThrow();
        }

        using (ModelPersistenceGuard.InternalOperation())
        {
            ModelPersistenceGuard.EnforceBeforeDeserialize();
        }
    }

    [Fact]
    public void InternalOperation_DoesNotSuppressSaveEnforcement()
    {
        ClearAllLicenseSources();
        ResetDefaultTrial();

        var manager = new TrialStateManager();
        for (int i = 0; i < TrialStateManager.TrialOperationLimit; i++)
        {
            manager.RecordOperationOrThrow();
        }

        // InternalOperation does NOT suppress Save/Load — only Serialize/Deserialize
        using (ModelPersistenceGuard.InternalOperation())
        {
            Assert.Throws<LicenseRequiredException>(() => ModelPersistenceGuard.EnforceBeforeSave());
        }
    }

    [Fact]
    public void InternalOperation_DoesNotSuppressLoadEnforcement()
    {
        ClearAllLicenseSources();
        ResetDefaultTrial();

        var manager = new TrialStateManager();
        for (int i = 0; i < TrialStateManager.TrialOperationLimit; i++)
        {
            manager.RecordOperationOrThrow();
        }

        using (ModelPersistenceGuard.InternalOperation())
        {
            Assert.Throws<LicenseRequiredException>(() => ModelPersistenceGuard.EnforceBeforeLoad());
        }
    }

    [Fact]
    public void InternalOperation_ScopeResetsOnDispose()
    {
        ClearAllLicenseSources();
        ResetDefaultTrial();

        var manager = new TrialStateManager();
        for (int i = 0; i < TrialStateManager.TrialOperationLimit; i++)
        {
            manager.RecordOperationOrThrow();
        }

        // Enter and leave scope
        using (ModelPersistenceGuard.InternalOperation())
        {
            // Suppressed — OK
            ModelPersistenceGuard.EnforceBeforeSerialize();
        }

        // After scope, enforcement should be active again
        Assert.Throws<LicenseRequiredException>(() => ModelPersistenceGuard.EnforceBeforeSerialize());
    }

    [Fact]
    public void InternalOperation_ThreadIsolation()
    {
        ClearAllLicenseSources();
        ResetDefaultTrial();

        var manager = new TrialStateManager();
        for (int i = 0; i < TrialStateManager.TrialOperationLimit; i++)
        {
            manager.RecordOperationOrThrow();
        }

        // The [ThreadStatic] flag should NOT leak to other threads
        bool otherThreadThrew = false;

        using (ModelPersistenceGuard.InternalOperation())
        {
            // This thread is suppressed
            ModelPersistenceGuard.EnforceBeforeSerialize();

            // Other thread should NOT be suppressed
            var task = Task.Run(() =>
            {
                try
                {
                    ModelPersistenceGuard.EnforceBeforeSerialize();
                }
                catch (LicenseRequiredException)
                {
                    otherThreadThrew = true;
                }
            });

            task.Wait();
        }

        Assert.True(otherThreadThrew, "InternalOperation flag leaked to another thread");
    }

    [Fact]
    public void LicenseKey_BypassesTrial_NoOperationCounted()
    {
        ClearAllLicenseSources();
        ResetDefaultTrial();

        // Set license key
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", "aidn.validlicens.abcdefghijklmnop");

        // Call enforce many times — should never count operations
        for (int i = 0; i < 50; i++)
        {
            ModelPersistenceGuard.EnforceBeforeSave();
            ModelPersistenceGuard.EnforceBeforeLoad();
            ModelPersistenceGuard.EnforceBeforeSerialize();
            ModelPersistenceGuard.EnforceBeforeDeserialize();
        }

        // Clear license key and check trial is still fresh
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", null);
        var manager = new TrialStateManager();
        var status = manager.GetStatus();
        Assert.Equal(0, status.OperationsUsed);
        Assert.False(status.IsExpired);
    }

    [Fact]
    public void EnforceBeforeSave_AlwaysEnforces_EvenInInternalScope()
    {
        ClearAllLicenseSources();
        ResetDefaultTrial();

        // Save/Load always enforce, even within InternalOperation scope
        // (only Serialize/Deserialize are suppressed)
        int operationsBefore;
        {
            var manager = new TrialStateManager();
            operationsBefore = manager.GetStatus().OperationsUsed;
        }

        using (ModelPersistenceGuard.InternalOperation())
        {
            ModelPersistenceGuard.EnforceBeforeSave();
        }

        {
            var manager = new TrialStateManager();
            int operationsAfter = manager.GetStatus().OperationsUsed;
            Assert.True(operationsAfter > operationsBefore, "EnforceBeforeSave should count even inside InternalOperation scope");
        }
    }

    [Fact]
    public void EnforceBeforeSerialize_OutsideScope_CountsOperation()
    {
        ClearAllLicenseSources();
        ResetDefaultTrial();

        int operationsBefore;
        {
            var manager = new TrialStateManager();
            operationsBefore = manager.GetStatus().OperationsUsed;
        }

        // Outside InternalOperation scope — should count
        ModelPersistenceGuard.EnforceBeforeSerialize();

        {
            var manager = new TrialStateManager();
            int operationsAfter = manager.GetStatus().OperationsUsed;
            Assert.True(operationsAfter > operationsBefore, "EnforceBeforeSerialize should count operations outside InternalOperation scope");
        }
    }

    [Fact]
    public void EnforceBeforeSerialize_InsideScope_DoesNotCountOperation()
    {
        ClearAllLicenseSources();
        ResetDefaultTrial();

        // Record one operation to establish the trial file
        ModelPersistenceGuard.EnforceBeforeSave();

        int operationsBefore;
        {
            var manager = new TrialStateManager();
            operationsBefore = manager.GetStatus().OperationsUsed;
        }

        // Inside InternalOperation scope — should NOT count
        using (ModelPersistenceGuard.InternalOperation())
        {
            ModelPersistenceGuard.EnforceBeforeSerialize();
        }

        {
            var manager = new TrialStateManager();
            int operationsAfter = manager.GetStatus().OperationsUsed;
            Assert.Equal(operationsBefore, operationsAfter);
        }
    }

    [Fact]
    public void WhitespaceOnlyLicenseKey_TreatedAsNoKey()
    {
        ClearAllLicenseSources();
        ResetDefaultTrial();

        // Exhaust the trial
        var manager = new TrialStateManager();
        for (int i = 0; i < TrialStateManager.TrialOperationLimit; i++)
        {
            manager.RecordOperationOrThrow();
        }

        // Set whitespace-only license key — should NOT bypass
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", "   ");

        Assert.Throws<LicenseRequiredException>(() => ModelPersistenceGuard.EnforceBeforeSave());
    }

    [Fact]
    public void EmptyLicenseKey_TreatedAsNoKey()
    {
        ClearAllLicenseSources();
        ResetDefaultTrial();

        var manager = new TrialStateManager();
        for (int i = 0; i < TrialStateManager.TrialOperationLimit; i++)
        {
            manager.RecordOperationOrThrow();
        }

        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", "");

        Assert.Throws<LicenseRequiredException>(() => ModelPersistenceGuard.EnforceBeforeSave());
    }
}
