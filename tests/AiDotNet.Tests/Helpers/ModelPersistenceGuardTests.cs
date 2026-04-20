using AiDotNet;
using AiDotNet.Classification.Linear;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Exceptions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
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
    /// Runs an action with an isolated trial file (the fixture's temp path,
    /// not the developer's real <c>~/.aidotnet/trial.json</c>). The
    /// production <see cref="ModelPersistenceGuard.EnforceCore"/> path is
    /// redirected to the temp path for the duration of the action via the
    /// internal <c>SetTestTrialFilePathOverride</c> hook, so the guard's
    /// trial-counting / license-checking branches see a fresh or
    /// test-controlled file without mutating anything on the real user
    /// profile.
    /// </summary>
    private void WithIsolatedTrial(Action action)
    {
        // Make sure the isolated path starts empty so trial state is fresh.
        if (File.Exists(_trialFilePath)) File.Delete(_trialFilePath);
        using (ModelPersistenceGuard.SetTestTrialFilePathOverride(_trialFilePath))
        {
            action();
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

        WithIsolatedTrial(() =>
        {
            // Record one save to materialize the trial file and set a baseline.
            ModelPersistenceGuard.EnforceBeforeSave();

            int before;
            {
                var m = new TrialStateManager(_trialFilePath);
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
                var m = new TrialStateManager(_trialFilePath);
                after = m.GetStatus().OperationsUsed;
            }
            Assert.Equal(before, after);

            // Round-trip structural equivalence check. Trial isn't exhausted
            // here, so the public Serialize() doesn't throw — use it to
            // assert that DeepCopy actually produces a byte-identical clone
            // rather than a malformed partial-deserialization result.
            var copy = (FeedForwardNeuralNetwork<double>)network.DeepCopy();
            Assert.NotSame(network, copy);
            Assert.IsType<FeedForwardNeuralNetwork<double>>(copy);
            Assert.Equal(network.Serialize(), copy.Serialize());
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
        Assert.NotSame(transformer, copy);
        Assert.IsType<Transformer<float>>(copy);

        // Full round-trip semantics: the serialized bytes must match
        // byte-for-byte between original and clone. Validates that every
        // layer (including MultiHeadAttentionLayer) reconstructed correctly.
        Assert.Equal(transformer.Serialize(), copy.Serialize());
    }

    [Fact(Timeout = 60000)]
    public async Task NeuralNetwork_DeepCopy_WithExhaustedTrial_DoesNotThrow()
    {
        // End-to-end proof of the fix: on a fully exhausted trial, the
        // old behavior was to throw LicenseRequiredException from the
        // first DeepCopy call. With the fix, DeepCopy bypasses the guard
        // via the private SerializeInternalUnchecked path.
        ClearAllLicenseSources();

        WithIsolatedTrial(() =>
        {
            // Exhaust the trial on the isolated path. This uses the same
            // TrialStateManager file the guard is now pointed at via
            // SetTestTrialFilePathOverride, so subsequent Save/Load/
            // Serialize/Deserialize calls from the guard would throw
            // LicenseRequiredException if they were reached.
            var manager = new TrialStateManager(_trialFilePath);
            for (int i = 0; i < TrialStateManager.TrialOperationLimit; i++)
            {
                manager.RecordOperationOrThrow();
            }

            var network = CreateSimpleFeedForward();

            // DeepCopy / Clone must succeed — training-internal, not user
            // save/load. Before the fix this would throw.
            //
            // Byte-level round-trip equivalence is asserted separately in
            // NeuralNetwork_DeepCopy_DoesNotCountTrialOperation (fresh trial)
            // because the public Serialize() fires the guard on an
            // exhausted trial — that's the exact boundary enforced by
            // DeepCopy_Output_Serialize_StillFiresGuard below.
            var copy = network.DeepCopy();
            Assert.NotNull(copy);
            Assert.NotSame(network, copy);
            Assert.IsType<FeedForwardNeuralNetwork<double>>(copy);

            var cloned = network.Clone();
            Assert.NotNull(cloned);
            Assert.NotSame(network, cloned);
            Assert.IsType<FeedForwardNeuralNetwork<double>>(cloned);
        });
    }

    // --- Defence-in-depth regression tests --------------------------------
    //
    // The license system's job is to gate user-facing Save/Load/Serialize/
    // Deserialize. DeepCopy is training-internal and so bypasses the guard —
    // but that bypass must NOT leak to any path that a user can reach to
    // exfiltrate weights. The tests below lock in the boundary:
    //
    //  - The COPY returned by DeepCopy is a normal model: Serialize() /
    //    SaveModel() on it fire the guard exactly like they do on the
    //    original. DeepCopy does not create a "licensing-free twin".
    //  - A user subclass that overrides virtual Serialize() cannot intercept
    //    DeepCopy's serialization. The internal round-trip uses a private,
    //    non-virtual serializer that the subclass can't see.

    [Fact(Timeout = 60000)]
    public async Task DeepCopy_Output_Serialize_StillFiresGuard()
    {
        // Lock-in test: the copy is not a "free pass" around the guard.
        // Calling Serialize() on the copy must fire EnforceBeforeSerialize
        // exactly like it does on the original. Without this guarantee,
        // `model.DeepCopy().Serialize()` would be a trivial license-bypass.
        ClearAllLicenseSources();

        WithIsolatedTrial(() =>
        {
            // Exhaust the trial on the isolated file so any call that
            // routes through the guard throws, but any call that bypasses
            // it (DeepCopy, per the fix) stays silent.
            var manager = new TrialStateManager(_trialFilePath);
            for (int i = 0; i < TrialStateManager.TrialOperationLimit; i++)
            {
                manager.RecordOperationOrThrow();
            }

            var network = CreateSimpleFeedForward();
            var copy = (FeedForwardNeuralNetwork<double>)network.DeepCopy();

            // copy.Serialize() MUST throw on an exhausted trial.
            Assert.Throws<LicenseRequiredException>(() => copy.Serialize());

            // And so must copy.SaveModel — guard path is symmetrical.
            string tempPath = Path.Combine(Path.GetTempPath(), $"aidotnet-test-{Guid.NewGuid():N}.bin");
            try
            {
                Assert.Throws<LicenseRequiredException>(() => copy.SaveModel(tempPath));
            }
            finally
            {
                if (File.Exists(tempPath)) File.Delete(tempPath);
            }
        });
    }

    /// <summary>
    /// Subclass whose override of <see cref="Serialize"/> performs a tracked
    /// side-effect (writing to a test-local flag). Used to verify that
    /// DeepCopy's serialization path does NOT invoke this override — i.e.
    /// the user override is only reachable from the public virtual call.
    /// </summary>
    private sealed class ExfilTrackingFeedForward : FeedForwardNeuralNetwork<double>
    {
        public int SerializeOverrideCalls { get; private set; }

        public ExfilTrackingFeedForward(NeuralNetworkArchitecture<double> arch) : base(arch) { }

        public override byte[] Serialize()
        {
            SerializeOverrideCalls++;
            return base.Serialize();
        }
    }

    [Fact(Timeout = 60000)]
    public async Task DeepCopy_DoesNotRouteThroughUserOverrideOfSerialize()
    {
        // Defence in depth: a user subclass that overrides Serialize (e.g.
        // to add logging, telemetry, or in the malicious case to write the
        // bytes to disk) must NOT see DeepCopy's internal round-trip.
        // DeepCopy uses the private, non-virtual SerializeInternalUnchecked
        // so the override's call counter stays at zero.
        //
        // Use a valid license key to avoid tripping the guard on the public
        // Serialize path while we observe DeepCopy's behaviour.
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", "aidn.testkey1234.abcdefghijklmnop");

        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 2);
        var network = new ExfilTrackingFeedForward(arch);

        // Sanity: override fires on direct public call.
        _ = network.Serialize();
        Assert.Equal(1, network.SerializeOverrideCalls);

        // The interesting case: DeepCopy must NOT route through the override.
        int before = network.SerializeOverrideCalls;
        var copy = network.DeepCopy();
        Assert.NotNull(copy);
        Assert.Equal(before, network.SerializeOverrideCalls);

        // Contract: DeepCopy round-trips through private
        // SerializeInternalUnchecked / Deserialize. The serialized bytes
        // carry only the base-class layer catalogue, so the concrete type
        // reconstructed by Deserialize is always the declared base —
        // user subclasses such as ExfilTrackingFeedForward are
        // intentionally NOT preserved. That property IS the defence:
        //   (a) Primary invariant — the original's override counter never
        //       incremented during DeepCopy (checked above at line 785).
        //   (b) Secondary invariant — the returned copy is the base
        //       FeedForwardNeuralNetwork type, so there is no subclass
        //       override on the copy that could be invoked even in theory.
        //
        // This assertion is deliberately unconditional: if a future
        // refactor teaches DeepCopy to preserve subclass identity, this
        // test must fail loudly rather than silently skip its second half —
        // because at that point an explicit check on
        // `((ExfilTrackingFeedForward)copy).SerializeOverrideCalls == 0`
        // must be added to keep the exfil guarantee.
        Assert.IsType<FeedForwardNeuralNetwork<double>>(copy);
    }

    // ---------------------------------------------------------------------
    // Facade-level regression for issue #1161 — DEFERRED.
    //
    // The review asked for an end-to-end test exercising the reported
    // failure path: AiModelBuilder.BuildAsync() → NormalOptimizer.
    // InitializeRandomSolution() → Clone() → DeepCopy() → (was)
    // Serialize()+guard. The intent is solid and we attempted it, but
    // every `AiModelBuilder.BuildAsync` test in the repo currently
    // StackOverflows on master (verified against both `RidgeClassifier`
    // and `RidgeRegression` — see `AiModelBuilderClassificationTests`
    // and `AiModelBuilderLicensingTests.SerializeModel_WithLicenseKey_Succeeds`,
    // both of which abort the test host under the current dependency
    // set). Adding a facade test here would make this PR's CI red for a
    // reason unrelated to the license-guard fix.
    //
    // The unit-level coverage below is strong — it exercises exactly the
    // `DeepCopy()` path the optimizer-snapshot call hits, on an exhausted
    // trial, and verifies no throw. Once the StackOverflow in
    // `BuildAsync` is fixed upstream, adding a facade regression here is
    // straightforward: instantiate a classifier or regressor, exhaust
    // the isolated trial via `WithIsolatedTrial`, await `BuildAsync`,
    // assert no throw + trial counter unchanged. Template:
    //
    //   using (ModelPersistenceGuard.SetTestTrialFilePathOverride(_trialFilePath))
    //   {
    //       // exhaust trial on _trialFilePath
    //       var built = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    //           .ConfigureDataLoader(loader)
    //           .ConfigureModel(classifier)
    //           .BuildAsync();
    //       Assert.NotNull(built);
    //       // assert trial count unchanged
    //   }
    // ---------------------------------------------------------------------
}
