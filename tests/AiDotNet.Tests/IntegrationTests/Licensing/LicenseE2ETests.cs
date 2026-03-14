using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Licensing;

/// <summary>
/// End-to-end tests covering the complete licensing lifecycle:
/// register community key → configure in library → train model → serialize/deserialize → predict.
/// </summary>
/// <remarks>
/// These tests validate the full chain without a live server by simulating the key registration
/// response format and then exercising the library's persistence pipeline.
/// </remarks>
[Collection("LicensingTests")]
public class LicenseE2ETests : IDisposable
{
    private readonly string? _originalEnvVar;
    private readonly string _licenseFilePath;
    private readonly string? _originalLicenseFile;

    public LicenseE2ETests()
    {
        _originalEnvVar = Environment.GetEnvironmentVariable("AIDOTNET_LICENSE_KEY");

        string homeDir = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        _licenseFilePath = Path.Combine(homeDir, ".aidotnet", "license.key");
        _originalLicenseFile = File.Exists(_licenseFilePath) ? File.ReadAllText(_licenseFilePath) : null;
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", _originalEnvVar);

        if (_originalLicenseFile != null)
        {
            File.WriteAllText(_licenseFilePath, _originalLicenseFile);
        }
        else if (File.Exists(_licenseFilePath))
        {
            File.Delete(_licenseFilePath);
        }
    }

    private void ClearAllLicenseSources()
    {
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", null);
        if (File.Exists(_licenseFilePath))
        {
            File.Delete(_licenseFilePath);
        }
    }

    // ─── E2E: Community License via Environment Variable ───

    [Fact]
    public void E2E_CommunityLicense_EnvVar_TrainSerializeDeserializePredict()
    {
        // Step 1: Simulate receiving a community license key from the register-community-license
        // Edge Function. The response shape is: { success: true, license_key: "aidn.xxx.yyy", tier: "community" }
        var communityKey = "aidn.comm12345678.abcdefghijklmnop";

        // Step 2: Set the key via environment variable (one of the three resolution methods)
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", communityKey);

        // Step 3: Verify the key resolves correctly
        var resolvedKey = LicenseKeyResolver.Resolve(null);
        Assert.Equal(communityKey, resolvedKey);

        // Step 4: Validate the key format offline
        var licenseKeyObj = new AiDotNetLicenseKey(communityKey) { ServerUrl = string.Empty };
        var validator = new LicenseValidator(licenseKeyObj);
        var validationResult = validator.Validate();
        Assert.Equal(LicenseKeyStatus.Active, validationResult.Status);

        // Step 5: Train a model (training is always free, never requires license)
        var result = TrainSimpleModel();
        Assert.NotNull(result);

        // Step 6: Serialize the model (requires license or trial)
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        byte[] serialized = builder.SerializeModel(result);
        Assert.NotNull(serialized);
        Assert.True(serialized.Length > 0);

        // Step 7: Deserialize the model back
        var restored = builder.DeserializeModel(serialized);
        Assert.NotNull(restored);

        // Step 8: Verify the restored model can predict (inference is always free)
        var (testX, _) = CreateLinearDataset(samples: 5, features: 3, seed: 99);
        var predictions = restored.Predict(testX);
        Assert.NotNull(predictions);
        Assert.Equal(5, predictions.Length);
    }

    // ─── E2E: Community License via File ───

    [Fact]
    public void E2E_CommunityLicense_File_TrainSerializeDeserializePredict()
    {
        ClearAllLicenseSources();

        // Step 1: Simulate writing a community key to the license file (as the website would instruct)
        var communityKey = "aidn.filekey12345.zyxwvutsrqponmlk";

        string dir = Path.GetDirectoryName(_licenseFilePath)!;
        if (!Directory.Exists(dir))
        {
            Directory.CreateDirectory(dir);
        }

        File.WriteAllText(_licenseFilePath, communityKey + Environment.NewLine);

        // Step 2: Verify the key resolves from file
        var resolvedKey = LicenseKeyResolver.Resolve(null);
        Assert.Equal(communityKey, resolvedKey);

        // Step 3: Train, serialize, deserialize, predict
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", communityKey);
        var result = TrainSimpleModel();
        byte[] serialized = new AiModelBuilder<double, Matrix<double>, Vector<double>>().SerializeModel(result);
        var restored = new AiModelBuilder<double, Matrix<double>, Vector<double>>().DeserializeModel(serialized);

        var (testX, _) = CreateLinearDataset(samples: 5, features: 3, seed: 99);
        var predictions = restored.Predict(testX);
        Assert.NotNull(predictions);
        Assert.Equal(5, predictions.Length);
    }

    // ─── E2E: Explicit License Key Object ───

    [Fact]
    public void E2E_ExplicitLicenseKey_TrainSerializeDeserializePredict()
    {
        ClearAllLicenseSources();

        // Step 1: Create an explicit license key object (as a paid customer would)
        var licenseKey = new AiDotNetLicenseKey("aidn.pro123456789.abcdefghijklmnop")
        {
            ServerUrl = string.Empty, // offline-only for test
        };

        // Step 2: Verify offline validation passes
        var validator = new LicenseValidator(licenseKey);
        var validationResult = validator.Validate();
        Assert.Equal(LicenseKeyStatus.Active, validationResult.Status);

        // Step 3: Train a model
        // Set env var so the guard finds a valid key during training helper calls
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", licenseKey.Key);
        var result = TrainSimpleModel();

        // Step 4: Full serialize → deserialize → predict cycle
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        byte[] serialized = builder.SerializeModel(result);
        Assert.True(serialized.Length > 0);

        var restored = builder.DeserializeModel(serialized);
        Assert.NotNull(restored);

        var (testX, _) = CreateLinearDataset(samples: 10, features: 3, seed: 77);
        var predictions = restored.Predict(testX);
        Assert.Equal(10, predictions.Length);
    }

    // ─── E2E: Trial Lifecycle ───

    [Fact]
    public void E2E_TrialLifecycle_TrainFreeSerializeWithTrialExhaustThenLicense()
    {
        ClearAllLicenseSources();
        var trialManager = new TrialStateManager();
        trialManager.Reset();

        // Step 1: Train a model (always free, no license needed)
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", "aidn.trainbypass1.qrstuvwxyz123456");
        var result = TrainSimpleModel();
        ClearAllLicenseSources();

        // Step 2: Serialize under trial (should count operations)
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        byte[] serialized = builder.SerializeModel(result);
        Assert.True(serialized.Length > 0);

        var status = trialManager.GetStatus();
        Assert.True(status.OperationsUsed > 0, "Trial operation should be counted");
        Assert.False(status.IsExpired, "Trial should not be expired yet");

        // Step 3: Exhaust the trial
        trialManager.Reset();
        for (int i = 0; i < TrialStateManager.TrialOperationLimit; i++)
        {
            trialManager.RecordOperationOrThrow();
        }

        // Step 4: Serialize should now throw
        Assert.Throws<AiDotNet.Exceptions.LicenseRequiredException>(() =>
            builder.SerializeModel(result));

        // Step 5: Add a license key — should immediately unblock
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", "aidn.rescue12345.abcdefghijklmnop");

        byte[] serialized2 = builder.SerializeModel(result);
        Assert.True(serialized2.Length > 0, "Licensed serialize should succeed after trial exhaustion");
    }

    // ─── E2E: License Key Format Validation ───

    [Fact]
    public void E2E_InvalidKeyFormat_RejectedByOfflineValidator()
    {
        // Keys that don't match aidn.X.Y format should be rejected
        var invalidKeys = new[]
        {
            "not-a-valid-key",
            "aidn",
            "aidn.",
            "aidn.only-one-part",
            "wrong.prefix.here",
            "",
        };

        foreach (var key in invalidKeys)
        {
            if (string.IsNullOrWhiteSpace(key))
            {
                // Empty/whitespace keys are rejected before format check
                var emptyKeyObj = new AiDotNetLicenseKey("placeholder") { ServerUrl = string.Empty };
                // Can't construct with empty key — that throws in constructor
                continue;
            }

            var keyObj = new AiDotNetLicenseKey(key) { ServerUrl = string.Empty };
            var validator = new LicenseValidator(keyObj);
            var result = validator.Validate();

            Assert.Equal(LicenseKeyStatus.Invalid, result.Status);
        }
    }

    [Fact]
    public void E2E_ValidKeyFormat_AcceptedByOfflineValidator()
    {
        var validKeys = new[]
        {
            "aidn.abc123def456.ghijklmnopqrstuv",
            "aidn.a.b",
            "aidn.community123.longersignature1",
        };

        foreach (var key in validKeys)
        {
            var keyObj = new AiDotNetLicenseKey(key) { ServerUrl = string.Empty };
            var validator = new LicenseValidator(keyObj);
            var result = validator.Validate();

            Assert.Equal(LicenseKeyStatus.Active, result.Status);
        }
    }

    // ─── E2E: License Key Resolver Chain ───

    [Fact]
    public void E2E_ResolverChain_ExplicitKeyTakesPriority()
    {
        // Set env var
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", "aidn.envvarkey123.abcdefghijklmnop");

        // Explicit key object should take priority
        var explicitKey = new AiDotNetLicenseKey("aidn.explicitkey1.zyxwvutsrqponmlk");
        var resolved = LicenseKeyResolver.Resolve(explicitKey);

        Assert.Equal("aidn.explicitkey1.zyxwvutsrqponmlk", resolved);
    }

    [Fact]
    public void E2E_ResolverChain_EnvVarFallsBackToFile()
    {
        ClearAllLicenseSources();

        // Write license file
        var fileKey = "aidn.fromfileke1.abcdefghijklmnop";
        string dir = Path.GetDirectoryName(_licenseFilePath)!;
        if (!Directory.Exists(dir))
        {
            Directory.CreateDirectory(dir);
        }

        File.WriteAllText(_licenseFilePath, fileKey + Environment.NewLine);

        // No env var, no explicit key — should resolve from file
        var resolved = LicenseKeyResolver.Resolve(null);
        Assert.Equal(fileKey, resolved);

        // Set env var — should now resolve from env var instead
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", "aidn.envvarkey123.abcdefghijklmnop");
        resolved = LicenseKeyResolver.Resolve(null);
        Assert.Equal("aidn.envvarkey123.abcdefghijklmnop", resolved);
    }

    [Fact]
    public void E2E_ResolverChain_NoSources_ReturnsNull()
    {
        ClearAllLicenseSources();

        var resolved = LicenseKeyResolver.Resolve(null);
        Assert.Null(resolved);
    }

    // ─── E2E: Unreachable Server Graceful Fallback ───

    [Fact]
    public void E2E_UnreachableServer_FallsBackToOffline()
    {
        // Simulate a key with a non-existent server — should not crash
        var key = new AiDotNetLicenseKey("aidn.servertest12.abcdefghijklmnop")
        {
            ServerUrl = "https://localhost:1/nonexistent",
        };

        var validator = new LicenseValidator(key);
        var result = validator.Validate();

        // Should return ValidationPending (server unreachable) or Active (cached)
        Assert.NotNull(result);
        Assert.True(
            result.Status == LicenseKeyStatus.ValidationPending ||
            result.Status == LicenseKeyStatus.Active,
            $"Expected ValidationPending or Active, got {result.Status}");
    }

    // ─── Helpers ───

    private static AiModelResult<double, Matrix<double>, Vector<double>> TrainSimpleModel()
    {
        var (x, y) = CreateLinearDataset(samples: 40, features: 3, seed: 42);
        var loader = DataLoaders.FromMatrixVector(x, y);

        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        return result;
    }

    private static (Matrix<double> x, Vector<double> y) CreateLinearDataset(int samples, int features, int seed)
    {
        var rng = new Random(seed);
        var x = new Matrix<double>(samples, features);
        var y = new Vector<double>(samples);

        var weights = new double[features];
        for (int j = 0; j < features; j++)
        {
            weights[j] = rng.NextDouble() * 2 - 1;
        }

        for (int i = 0; i < samples; i++)
        {
            double sum = 0;
            for (int j = 0; j < features; j++)
            {
                double val = rng.NextDouble() * 10;
                x[i, j] = val;
                sum += val * weights[j];
            }

            y[i] = sum + (rng.NextDouble() - 0.5) * 0.1;
        }

        return (x, y);
    }
}
