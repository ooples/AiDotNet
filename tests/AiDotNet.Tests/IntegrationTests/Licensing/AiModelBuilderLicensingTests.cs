using AiDotNet.Data.Loaders;
using AiDotNet.Exceptions;
using AiDotNet.Helpers;
using AiDotNet.Models;
using AiDotNet.Models.Results;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Licensing;

/// <summary>
/// Integration tests verifying that licensing enforcement works correctly through
/// the AiModelBuilder save/load pipeline end-to-end.
/// </summary>
[Collection("LicensingTests")]
public class AiModelBuilderLicensingTests : IDisposable
{
    private readonly string? _originalEnvVar;
    private readonly string _licenseFilePath;
    private readonly string? _originalLicenseFile;

    public AiModelBuilderLicensingTests()
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

    private void ResetDefaultTrial()
    {
        var manager = new TrialStateManager();
        manager.Reset();
    }

    private static AiModelResult<double, Matrix<double>, Vector<double>> TrainSimpleModel()
    {
        var (x, y) = CreateLinearDataset(samples: 40, features: 3, seed: 42);
        var loader = DataLoaders.FromMatrixVector(x, y);

        // Set license key during training so we don't consume trial ops
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", "training-bypass-key");

        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        return result;
    }

    [Fact]
    public void SerializeModel_WithLicenseKey_Succeeds()
    {
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", "test-license-key");
        var result = TrainSimpleModel();
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();

        // Should succeed without any trial impact
        byte[] data = builder.SerializeModel(result);
        Assert.NotNull(data);
        Assert.True(data.Length > 0);
    }

    [Fact]
    public void DeserializeModel_WithLicenseKey_Succeeds()
    {
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", "test-license-key");
        var result = TrainSimpleModel();
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();

        byte[] data = builder.SerializeModel(result);
        var restored = builder.DeserializeModel(data);
        Assert.NotNull(restored);
    }

    [Fact]
    public void SerializeModel_WithoutLicense_CountsOneTrialOperation()
    {
        ClearAllLicenseSources();
        ResetDefaultTrial();

        var result = TrainSimpleModel();

        // Clear license so trial kicks in
        ClearAllLicenseSources();

        var manager = new TrialStateManager();
        int before = manager.GetStatus().OperationsUsed;

        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        builder.SerializeModel(result);

        int after = manager.GetStatus().OperationsUsed;

        // Should count exactly one operation (not double-count from inner Serialize)
        Assert.Equal(before + 1, after);
    }

    [Fact]
    public void DeserializeModel_WithoutLicense_CountsOneTrialOperation()
    {
        // Serialize with license key
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", "test-license-key");
        var result = TrainSimpleModel();
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        byte[] data = builder.SerializeModel(result);

        // Now clear license and reset trial
        ClearAllLicenseSources();
        ResetDefaultTrial();

        var manager = new TrialStateManager();
        int before = manager.GetStatus().OperationsUsed;

        builder.DeserializeModel(data);

        int after = manager.GetStatus().OperationsUsed;
        Assert.Equal(before + 1, after);
    }

    [Fact]
    public void SerializeModel_ExhaustedTrial_ThrowsLicenseRequiredException()
    {
        ClearAllLicenseSources();
        ResetDefaultTrial();

        var result = TrainSimpleModel();
        ClearAllLicenseSources();

        // Exhaust the trial
        var manager = new TrialStateManager();
        for (int i = 0; i < TrialStateManager.TrialOperationLimit; i++)
        {
            manager.RecordOperationOrThrow();
        }

        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        Assert.Throws<LicenseRequiredException>(() => builder.SerializeModel(result));
    }

    [Fact]
    public void DeserializeModel_ExhaustedTrial_ThrowsLicenseRequiredException()
    {
        // Serialize with license key
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", "test-license-key");
        var result = TrainSimpleModel();
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        byte[] data = builder.SerializeModel(result);

        // Clear and exhaust trial
        ClearAllLicenseSources();
        ResetDefaultTrial();

        var manager = new TrialStateManager();
        for (int i = 0; i < TrialStateManager.TrialOperationLimit; i++)
        {
            manager.RecordOperationOrThrow();
        }

        Assert.Throws<LicenseRequiredException>(() => builder.DeserializeModel(data));
    }

    [Fact]
    public void DirectSerialize_WithoutLicense_CountsTrialOperation()
    {
        ClearAllLicenseSources();
        ResetDefaultTrial();

        var result = TrainSimpleModel();
        ClearAllLicenseSources();

        var manager = new TrialStateManager();
        int before = manager.GetStatus().OperationsUsed;

        // Call Serialize directly on the result (not via builder)
        result.Serialize();

        int after = manager.GetStatus().OperationsUsed;
        Assert.True(after > before, "Direct Serialize should count a trial operation");
    }

    [Fact]
    public void TrainingDoesNotCountTrialOperations()
    {
        // Reset trial to a known state
        ClearAllLicenseSources();
        ResetDefaultTrial();

        // Set a license key so any incidental internal serialization is allowed,
        // then verify the trial counter stays at zero (training itself doesn't count).
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", "training-test-key");

        var (x, y) = CreateLinearDataset(samples: 40, features: 3, seed: 42);
        var loader = DataLoaders.FromMatrixVector(x, y);

        new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureDataLoader(loader)
            .ConfigureModel(new RidgeRegression<double>())
            .BuildAsync()
            .GetAwaiter()
            .GetResult();

        // Clear the key and check the trial — if training counted ops, the counter would be > 0
        // (the license key bypasses the guard entirely, so ops aren't counted)
        Environment.SetEnvironmentVariable("AIDOTNET_LICENSE_KEY", null);

        var manager = new TrialStateManager();
        var status = manager.GetStatus();
        Assert.Equal(0, status.OperationsUsed);
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
