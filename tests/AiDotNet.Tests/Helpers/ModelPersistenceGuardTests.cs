using AiDotNet.Enums;
using AiDotNet.Exceptions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using Xunit;
using System.Threading.Tasks;

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
    /// Resets the trial file to allow fresh trial counting.
    /// Uses the isolated temp path to avoid modifying real user trial state.
    /// </summary>
    private void ResetDefaultTrial()
    {
        var manager = new TrialStateManager(_trialFilePath);
        manager.Reset();
    }

    [Fact(Timeout = 60000)]
    public async Task EnforceBeforeSave_WithLicenseKey_DoesNotThrow()
    {
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", "aidn.testkey1234.abcdefghijklmnop");

        // Should not throw — license key present bypasses trial
        ModelPersistenceGuard.EnforceBeforeSave();
    }

    [Fact(Timeout = 60000)]
    public async Task EnforceBeforeLoad_WithLicenseKey_DoesNotThrow()
    {
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", "aidn.testkey1234.abcdefghijklmnop");

        ModelPersistenceGuard.EnforceBeforeLoad();
    }

    [Fact(Timeout = 60000)]
    public async Task EnforceBeforeSerialize_WithLicenseKey_DoesNotThrow()
    {
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", "aidn.testkey1234.abcdefghijklmnop");

        ModelPersistenceGuard.EnforceBeforeSerialize();
    }

    [Fact(Timeout = 60000)]
    public async Task EnforceBeforeDeserialize_WithLicenseKey_DoesNotThrow()
    {
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", "aidn.testkey1234.abcdefghijklmnop");

        ModelPersistenceGuard.EnforceBeforeDeserialize();
    }

    [Fact(Timeout = 60000)]
    public async Task EnforceBeforeSave_WithoutLicense_CountsTrialOperation()
    {
        ClearAllLicenseSources();
        ResetDefaultTrial();

        // First call should succeed (trial is fresh)
        ModelPersistenceGuard.EnforceBeforeSave();

        // Verify trial counter was incremented to exactly 1
        var manager = new TrialStateManager(_trialFilePath);
        var status = manager.GetStatus();
        Assert.Equal(1, status.OperationsUsed);
    }

    [Fact(Timeout = 60000)]
    public async Task EnforceBeforeLoad_WithoutLicense_CountsTrialOperation()
    {
        ClearAllLicenseSources();
        ResetDefaultTrial();

        ModelPersistenceGuard.EnforceBeforeLoad();

        var manager = new TrialStateManager(_trialFilePath);
        var status = manager.GetStatus();
        Assert.Equal(1, status.OperationsUsed);
    }

    [Fact(Timeout = 60000)]
    public async Task EnforceBeforeSave_ExhaustedTrial_ThrowsLicenseRequiredException()
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

    [Fact(Timeout = 60000)]
    public async Task EnforceBeforeLoad_ExhaustedTrial_ThrowsLicenseRequiredException()
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

    [Fact(Timeout = 60000)]
    public async Task EnforceBeforeSerialize_ExhaustedTrial_ThrowsLicenseRequiredException()
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

    [Fact(Timeout = 60000)]
    public async Task EnforceBeforeDeserialize_ExhaustedTrial_ThrowsLicenseRequiredException()
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

    [Fact(Timeout = 60000)]
    public async Task InternalOperation_SuppressesSerializeEnforcement()
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

    [Fact(Timeout = 60000)]
    public async Task InternalOperation_SuppressesDeserializeEnforcement()
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

    [Fact(Timeout = 60000)]
    public async Task InternalOperation_DoesNotSuppressSaveEnforcement()
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

    [Fact(Timeout = 60000)]
    public async Task InternalOperation_DoesNotSuppressLoadEnforcement()
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

    [Fact(Timeout = 60000)]
    public async Task InternalOperation_ScopeResetsOnDispose()
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

    [Fact(Timeout = 60000)]
    public async Task InternalOperation_ThreadIsolation()
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

    [Fact(Timeout = 60000)]
    public async Task LicenseKey_BypassesTrial_NoOperationCounted()
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

    [Fact(Timeout = 60000)]
    public async Task EnforceBeforeSave_AlwaysEnforces_EvenInInternalScope()
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

    [Fact(Timeout = 60000)]
    public async Task EnforceBeforeSerialize_OutsideScope_CountsOperation()
    {
        ClearAllLicenseSources();
        ResetDefaultTrial();

        int operationsBefore;
        {
            var manager = new TrialStateManager(_trialFilePath);
            operationsBefore = manager.GetStatus().OperationsUsed;
        }

        // Outside InternalOperation scope — should count
        ModelPersistenceGuard.EnforceBeforeSerialize();

        {
            var manager = new TrialStateManager(_trialFilePath);
            int operationsAfter = manager.GetStatus().OperationsUsed;
            Assert.Equal(operationsBefore + 1, operationsAfter);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task EnforceBeforeSerialize_InsideScope_DoesNotCountOperation()
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

    [Fact(Timeout = 60000)]
    public async Task WhitespaceOnlyLicenseKey_TreatedAsNoKey()
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

    [Fact(Timeout = 60000)]
    public async Task EmptyLicenseKey_TreatedAsNoKey()
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

    // ---------------------------------------------------------------------
    // Regression tests for issue #1161:
    //   NeuralNetworkBase.DeepCopy() internally calls Serialize(), which
    //   used to trip the ModelPersistenceGuard on an expired trial. The
    //   optimizer's InitializeRandomSolution calls Clone() → DeepCopy() as
    //   part of every training run, so this one-line issue blocked every
    //   AiModelBuilder.BuildAsync() call on an expired trial — even though
    //   the guard's own XML docs state "training and inference are never
    //   restricted."
    //
    //   Fix: wrap DeepCopy's Serialize/Deserialize pair in
    //   ModelPersistenceGuard.InternalOperation(). These tests verify the
    //   fix against the actual model types (not just the guard API).
    // ---------------------------------------------------------------------

    private static FeedForwardNeuralNetwork<double> CreateSimpleFeedForward()
    {
        // Simple feed-forward is used for these tests because its
        // Serialize/Deserialize round-trip is well-exercised by existing
        // integration tests. This lets us isolate the guard-vs-DeepCopy
        // fix from any unrelated serialization bugs elsewhere in the layer
        // catalogue.
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 2);
        return new FeedForwardNeuralNetwork<double>(arch);
    }

    /// <summary>
    /// Runs an action with a transient fresh real trial. Backs up and
    /// restores the user's real ~/.aidotnet/trial.json around the action
    /// so tests can exercise the guard against a known-state trial file
    /// without permanently disturbing the developer's local state.
    /// </summary>
    private static void WithFreshRealTrial(Action action)
    {
        string homeDir = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string trialPath = Path.Combine(homeDir, ".aidotnet", "trial.json");
        string? backupContent = File.Exists(trialPath) ? File.ReadAllText(trialPath) : null;
        try
        {
            if (File.Exists(trialPath)) File.Delete(trialPath);
            action();
        }
        finally
        {
            try
            {
                if (backupContent != null)
                    File.WriteAllText(trialPath, backupContent);
                else if (File.Exists(trialPath))
                    File.Delete(trialPath);
            }
            catch
            {
                // Best-effort restore.
            }
        }
    }

    [Fact(Timeout = 60000)]
    public async Task NeuralNetwork_DeepCopy_DoesNotCountTrialOperation()
    {
        // The heart of issue #1161: DeepCopy is a training-internal
        // in-memory clone. The NormalOptimizer.InitializeRandomSolution
        // call path uses it on every training step. It must NOT count as
        // a billable save/load operation against the free trial — if it
        // did, ten training steps would exhaust the trial and every
        // subsequent AiModelBuilder.BuildAsync() would throw.
        //
        // Before the fix: DeepCopy() called Serialize() directly, which
        // called ModelPersistenceGuard.EnforceBeforeSerialize() and either
        // counted an operation or threw LicenseRequiredException on an
        // exhausted trial. This test proves neither happens now.
        ClearAllLicenseSources();

        WithFreshRealTrial(() =>
        {
            // Record one save to materialize the trial file and set a baseline.
            ModelPersistenceGuard.EnforceBeforeSave();

            int before;
            {
                var m = new TrialStateManager();
                before = m.GetStatus().OperationsUsed;
            }

            var network = CreateSimpleFeedForward();

            // Multiple DeepCopy / Clone calls must not consume trial operations.
            for (int i = 0; i < 5; i++)
            {
                _ = network.DeepCopy();
                _ = network.Clone();
            }

            int after;
            {
                var m = new TrialStateManager();
                after = m.GetStatus().OperationsUsed;
            }
            Assert.Equal(before, after);
        });
    }

    [Fact(Timeout = 60000)]
    public async Task Transformer_DeepCopy_WithValidLicenseKey_RoundTripsMultiHeadAttention()
    {
        // Regression for a second bug surfaced by PR #1163's tests: the
        // DeserializationHelper.CreateMultiHeadAttentionLayer probe
        // looked for a 4-arg constructor, but the public ctor has 5 args
        // (the trailing IInitializationStrategy<T>). Before the
        // DeserializationHelper fix, this test would throw
        // "Cannot find MultiHeadAttentionLayer constructor ..." during
        // Deserialize(). This verifies that a Transformer round-trips
        // through Serialize/Deserialize successfully.
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", "aidn.testkey1234.abcdefghijklmnop");

        var architecture = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 1,
            numDecoderLayers: 0,
            numHeads: 2,
            modelDimension: 8,
            feedForwardDimension: 16,
            inputSize: 4,
            outputSize: 4,
            maxSequenceLength: 4,
            vocabularySize: 4);
        var transformer = new Transformer<float>(architecture);

        // The DeepCopy path calls Serialize + Deserialize internally.
        // Before the DeserializationHelper fix, Deserialize would throw
        // "Cannot find MultiHeadAttentionLayer constructor ...".
        var copy = transformer.DeepCopy();
        Assert.NotNull(copy);
    }

    [Fact(Timeout = 60000)]
    public async Task NeuralNetwork_DeepCopy_WithExhaustedTrial_DoesNotThrow()
    {
        // End-to-end proof of the fix: on a fully exhausted trial, the
        // old behavior was to throw LicenseRequiredException from the
        // first DeepCopy call. With the fix, DeepCopy stays inside an
        // InternalOperation scope so the guard never fires.
        ClearAllLicenseSources();

        WithFreshRealTrial(() =>
        {
            // Exhaust the trial (same pattern as the existing
            // InternalOperation_SuppressesSerializeEnforcement test above).
            var manager = new TrialStateManager();
            for (int i = 0; i < TrialStateManager.TrialOperationLimit; i++)
            {
                manager.RecordOperationOrThrow();
            }

            var network = CreateSimpleFeedForward();

            // DeepCopy / Clone must succeed — training-internal, not user
            // save/load. Before the fix this would throw.
            var copy = network.DeepCopy();
            Assert.NotNull(copy);

            var cloned = network.Clone();
            Assert.NotNull(cloned);
        });
    }
}
