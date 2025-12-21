using AiDotNet.AdversarialRobustness.CertifiedRobustness;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.AdversarialRobustness;

/// <summary>
/// Comprehensive tests for CROWN (Convex Relaxation based perturbation analysis Of Neural networks)
/// verification implementation. Tests cover mathematical correctness of linear relaxation bounds
/// and API functionality.
/// </summary>
public class CROWNVerificationTests
{
    private const double Tolerance = 1e-6;

    #region Constructor Tests

    [Fact]
    public void Constructor_Default_CreatesInstance()
    {
        // Act
        var crown = new CROWNVerification<double>();

        // Assert
        Assert.NotNull(crown);
        var options = crown.GetOptions();
        Assert.NotNull(options);
        Assert.Equal("CROWN", options.CertificationMethod);
        Assert.True(options.UseTightBounds);
    }

    [Fact]
    public void Constructor_WithOptions_UsesOptions()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.1,
            NumSamples = 100,
            ConfidenceLevel = 0.95
        };

        // Act
        var crown = new CROWNVerification<double>(options);

        // Assert
        Assert.NotNull(crown);
        var returnedOptions = crown.GetOptions();
        Assert.Equal(0.1, returnedOptions.NoiseSigma, Tolerance);
        Assert.Equal(100, returnedOptions.NumSamples);
        Assert.Equal(0.95, returnedOptions.ConfidenceLevel, Tolerance);
        Assert.Equal("CROWN", returnedOptions.CertificationMethod);
        Assert.True(returnedOptions.UseTightBounds);
    }

    [Fact]
    public void Constructor_WithNullOptions_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new CROWNVerification<double>(null!));
    }

    #endregion

    #region CertifyPrediction Tests

    [Fact]
    public void CertifyPrediction_WithNullInput_ThrowsArgumentNullException()
    {
        // Arrange
        var crown = new CROWNVerification<double>();
        var mockModel = new MockPredictiveModel(3, 2);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => crown.CertifyPrediction(null!, mockModel));
    }

    [Fact]
    public void CertifyPrediction_WithNullModel_ThrowsArgumentNullException()
    {
        // Arrange
        var crown = new CROWNVerification<double>();
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => crown.CertifyPrediction(input, null!));
    }

    [Fact]
    public void CertifyPrediction_WithValidInput_ReturnsCertification()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.01,
            NumSamples = 50
        };
        var crown = new CROWNVerification<double>(options);
        var mockModel = new MockPredictiveModel(3, 2);
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        // Act
        var result = crown.CertifyPrediction(input, mockModel);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.PredictedClass >= 0);
        Assert.True(result.PredictedClass < 2);
        Assert.True(result.Confidence >= 0.0 && result.Confidence <= 1.0);
    }

    [Fact]
    public void CertifyPrediction_WithSmallEpsilon_MayCertify()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.001, // Very small perturbation
            NumSamples = 100
        };
        var crown = new CROWNVerification<double>(options);
        var mockModel = new MockPredictiveModelWithMargin(3, 2, 1.0); // Large margin
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        // Act
        var result = crown.CertifyPrediction(input, mockModel);

        // Assert
        Assert.NotNull(result);
        // With small epsilon and large margin, should be certified
        Assert.True(result.IsCertified);
    }

    [Fact]
    public void CertifyPrediction_CROWN_ShouldProduceTighterBounds_ThanIBP()
    {
        // Arrange - CROWN should produce tighter bounds than IBP
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.05,
            NumSamples = 100
        };

        var crown = new CROWNVerification<double>(options);
        var ibp = new IntervalBoundPropagation<double>(options);
        var mockModel = new MockPredictiveModelWithMargin(3, 2, 0.3);
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        // Act
        var crownResult = crown.CertifyPrediction(input, mockModel);
        var ibpResult = ibp.CertifyPrediction(input, mockModel);

        // Assert
        // CROWN should have tighter bounds (larger lower bound, smaller upper bound)
        // or at least equal to IBP bounds
        Assert.True(crownResult.LowerBound >= ibpResult.LowerBound - Tolerance);
        Assert.True(crownResult.UpperBound <= ibpResult.UpperBound + Tolerance);
    }

    [Fact]
    public void CertifyPrediction_ReturnsBoundsForPredictedClass()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.05,
            NumSamples = 50
        };
        var crown = new CROWNVerification<double>(options);
        var mockModel = new MockPredictiveModel(3, 2);
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        // Act
        var result = crown.CertifyPrediction(input, mockModel);

        // Assert
        // LowerBound should be <= UpperBound
        Assert.True(result.LowerBound <= result.UpperBound);
    }

    #endregion

    #region CertifyBatch Tests

    [Fact]
    public void CertifyBatch_WithNullInputs_ThrowsArgumentNullException()
    {
        // Arrange
        var crown = new CROWNVerification<double>();
        var mockModel = new MockPredictiveModel(3, 2);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => crown.CertifyBatch(null!, mockModel));
    }

    [Fact]
    public void CertifyBatch_WithNullModel_ThrowsArgumentNullException()
    {
        // Arrange
        var crown = new CROWNVerification<double>();
        var inputs = new Matrix<double>(2, 3);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => crown.CertifyBatch(inputs, null!));
    }

    [Fact]
    public void CertifyBatch_ReturnsCorrectNumberOfResults()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.01,
            NumSamples = 20
        };
        var crown = new CROWNVerification<double>(options);
        var mockModel = new MockPredictiveModel(3, 2);

        // Create batch of 5 samples
        var inputs = new Matrix<double>(5, 3);
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                inputs[i, j] = 0.1 * (i + 1) + 0.05 * j;
            }
        }

        // Act
        var results = crown.CertifyBatch(inputs, mockModel);

        // Assert
        Assert.NotNull(results);
        Assert.Equal(5, results.Length);
        foreach (var result in results)
        {
            Assert.NotNull(result);
        }
    }

    #endregion

    #region ComputeCertifiedRadius Tests

    [Fact]
    public void ComputeCertifiedRadius_WithNullInput_ThrowsArgumentNullException()
    {
        // Arrange
        var crown = new CROWNVerification<double>();
        var mockModel = new MockPredictiveModel(3, 2);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => crown.ComputeCertifiedRadius(null!, mockModel));
    }

    [Fact]
    public void ComputeCertifiedRadius_WithNullModel_ThrowsArgumentNullException()
    {
        // Arrange
        var crown = new CROWNVerification<double>();
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => crown.ComputeCertifiedRadius(input, null!));
    }

    [Fact]
    public void ComputeCertifiedRadius_ReturnsNonNegativeRadius()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.05,
            NumSamples = 50
        };
        var crown = new CROWNVerification<double>(options);
        var mockModel = new MockPredictiveModelWithMargin(3, 2, 0.5);
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        // Act
        var radius = crown.ComputeCertifiedRadius(input, mockModel);

        // Assert
        Assert.True(radius >= 0.0);
    }

    [Fact]
    public void ComputeCertifiedRadius_CROWN_ShouldGiveTighterRadius_ThanIBP()
    {
        // Arrange - CROWN with tighter bounds should give larger certified radius
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.1,
            NumSamples = 100
        };

        var crown = new CROWNVerification<double>(options);
        var ibp = new IntervalBoundPropagation<double>(options);
        var mockModel = new MockPredictiveModelWithMargin(3, 2, 0.5);
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        // Act
        var crownRadius = crown.ComputeCertifiedRadius(input, mockModel);
        var ibpRadius = ibp.ComputeCertifiedRadius(input, mockModel);

        // Assert
        // CROWN should produce >= radius compared to IBP due to tighter bounds
        Assert.True(crownRadius >= ibpRadius - Tolerance);
    }

    #endregion

    #region EvaluateCertifiedAccuracy Tests

    [Fact]
    public void EvaluateCertifiedAccuracy_WithNullTestData_ThrowsArgumentNullException()
    {
        // Arrange
        var crown = new CROWNVerification<double>();
        var mockModel = new MockPredictiveModel(3, 2);
        var labels = new Vector<int>(new[] { 0, 1, 0 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            crown.EvaluateCertifiedAccuracy(null!, labels, mockModel, 0.1));
    }

    [Fact]
    public void EvaluateCertifiedAccuracy_WithNullLabels_ThrowsArgumentNullException()
    {
        // Arrange
        var crown = new CROWNVerification<double>();
        var mockModel = new MockPredictiveModel(3, 2);
        var testData = new Matrix<double>(3, 3);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            crown.EvaluateCertifiedAccuracy(testData, null!, mockModel, 0.1));
    }

    [Fact]
    public void EvaluateCertifiedAccuracy_WithNullModel_ThrowsArgumentNullException()
    {
        // Arrange
        var crown = new CROWNVerification<double>();
        var testData = new Matrix<double>(3, 3);
        var labels = new Vector<int>(new[] { 0, 1, 0 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            crown.EvaluateCertifiedAccuracy(testData, labels, null!, 0.1));
    }

    [Fact]
    public void EvaluateCertifiedAccuracy_WithMismatchedDimensions_ThrowsArgumentException()
    {
        // Arrange
        var crown = new CROWNVerification<double>();
        var mockModel = new MockPredictiveModel(3, 2);
        var testData = new Matrix<double>(5, 3);
        var labels = new Vector<int>(new[] { 0, 1, 0 }); // Only 3 labels for 5 samples

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            crown.EvaluateCertifiedAccuracy(testData, labels, mockModel, 0.1));
    }

    [Fact]
    public void EvaluateCertifiedAccuracy_ReturnsValidMetrics()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.01,
            NumSamples = 20
        };
        var crown = new CROWNVerification<double>(options);
        var mockModel = new MockPredictiveModelWithMargin(3, 2, 0.5);

        // Create test data
        var testData = new Matrix<double>(4, 3);
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                testData[i, j] = 0.5;
            }
        }

        // Labels matching the model's predictions
        var labels = new Vector<int>(new[] { 0, 0, 0, 0 });

        // Act
        var metrics = crown.EvaluateCertifiedAccuracy(testData, labels, mockModel, 0.01);

        // Assert
        Assert.NotNull(metrics);
        Assert.True(metrics.CleanAccuracy >= 0.0 && metrics.CleanAccuracy <= 1.0);
        Assert.True(metrics.CertifiedAccuracy >= 0.0 && metrics.CertifiedAccuracy <= 1.0);
        Assert.True(metrics.CertificationRate >= 0.0 && metrics.CertificationRate <= 1.0);
        Assert.True(metrics.CertifiedAccuracy <= metrics.CleanAccuracy);
    }

    #endregion

    #region Serialization Tests

    [Fact]
    public void Serialize_ReturnsNonEmptyBytes()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.1,
            NumSamples = 100,
            ConfidenceLevel = 0.99
        };
        var crown = new CROWNVerification<double>(options);

        // Act
        var bytes = crown.Serialize();

        // Assert
        Assert.NotNull(bytes);
        Assert.True(bytes.Length > 0);
    }

    [Fact]
    public void Deserialize_WithNullData_ThrowsArgumentNullException()
    {
        // Arrange
        var crown = new CROWNVerification<double>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => crown.Deserialize(null!));
    }

    [Fact]
    public void SerializeDeserialize_PreservesOptions()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.15,
            NumSamples = 75,
            ConfidenceLevel = 0.9,
            NormType = "L2",
            UseTightBounds = true,
            BatchSize = 32,
            RandomSeed = 42
        };
        var crown = new CROWNVerification<double>(options);

        // Act
        var bytes = crown.Serialize();
        var crown2 = new CROWNVerification<double>();
        crown2.Deserialize(bytes);
        var restoredOptions = crown2.GetOptions();

        // Assert
        Assert.Equal(0.15, restoredOptions.NoiseSigma, Tolerance);
        Assert.Equal(75, restoredOptions.NumSamples);
        Assert.Equal(0.9, restoredOptions.ConfidenceLevel, Tolerance);
        Assert.Equal("L2", restoredOptions.NormType);
        Assert.True(restoredOptions.UseTightBounds);
        Assert.Equal(32, restoredOptions.BatchSize);
        Assert.Equal(42, restoredOptions.RandomSeed);
        Assert.Equal("CROWN", restoredOptions.CertificationMethod);
    }

    #endregion

    #region SaveModel/LoadModel Tests

    [Fact]
    public void SaveModel_WithNullPath_ThrowsArgumentException()
    {
        // Arrange
        var crown = new CROWNVerification<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => crown.SaveModel(null!));
    }

    [Fact]
    public void SaveModel_WithEmptyPath_ThrowsArgumentException()
    {
        // Arrange
        var crown = new CROWNVerification<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => crown.SaveModel(string.Empty));
    }

    [Fact]
    public void LoadModel_WithNullPath_ThrowsArgumentException()
    {
        // Arrange
        var crown = new CROWNVerification<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => crown.LoadModel(null!));
    }

    [Fact]
    public void LoadModel_WithNonExistentFile_ThrowsFileNotFoundException()
    {
        // Arrange
        var crown = new CROWNVerification<double>();

        // Act & Assert
        Assert.Throws<FileNotFoundException>(() => crown.LoadModel("nonexistent_crown_model_12345.json"));
    }

    [Fact]
    public void SaveAndLoadModel_RoundTrip_PreservesOptions()
    {
        // Arrange
        var tempPath = Path.Combine(Path.GetTempPath(), $"crown_test_{Guid.NewGuid()}.json");
        try
        {
            var options = new CertifiedDefenseOptions<double>
            {
                NoiseSigma = 0.2,
                NumSamples = 150,
                ConfidenceLevel = 0.95,
                UseTightBounds = true
            };
            var crown = new CROWNVerification<double>(options);

            // Act
            crown.SaveModel(tempPath);
            var crown2 = new CROWNVerification<double>();
            crown2.LoadModel(tempPath);
            var loadedOptions = crown2.GetOptions();

            // Assert
            Assert.Equal(0.2, loadedOptions.NoiseSigma, Tolerance);
            Assert.Equal(150, loadedOptions.NumSamples);
            Assert.Equal(0.95, loadedOptions.ConfidenceLevel, Tolerance);
            Assert.True(loadedOptions.UseTightBounds);
            Assert.Equal("CROWN", loadedOptions.CertificationMethod);
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
        }
    }

    #endregion

    #region Reset Tests

    [Fact]
    public void Reset_ResetsToDefaultOptions()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.5,
            NumSamples = 200,
            UseTightBounds = false
        };
        var crown = new CROWNVerification<double>(options);

        // Act
        crown.Reset();
        var resetOptions = crown.GetOptions();

        // Assert
        Assert.Equal("CROWN", resetOptions.CertificationMethod);
        Assert.True(resetOptions.UseTightBounds);
    }

    #endregion

    #region CROWN-Specific Linear Relaxation Tests

    [Fact]
    public void CROWN_WithIdenticalInputs_ProducesSameCertification()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.01,
            NumSamples = 50,
            RandomSeed = 42
        };
        var crown1 = new CROWNVerification<double>(options);
        var crown2 = new CROWNVerification<double>(options);
        var mockModel = new MockPredictiveModel(3, 2);
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        // Act
        var result1 = crown1.CertifyPrediction(input, mockModel);
        var result2 = crown2.CertifyPrediction(input, mockModel);

        // Assert - Same seed and input should produce same results
        Assert.Equal(result1.PredictedClass, result2.PredictedClass);
        Assert.Equal(result1.IsCertified, result2.IsCertified);
        Assert.Equal(result1.LowerBound, result2.LowerBound, Tolerance);
        Assert.Equal(result1.UpperBound, result2.UpperBound, Tolerance);
    }

    [Fact]
    public void CROWN_WithLargerEpsilon_ProducesWiderBounds()
    {
        // Arrange
        var smallEpsOptions = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.01,
            NumSamples = 50
        };
        var largeEpsOptions = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.1,
            NumSamples = 50
        };
        var crown1 = new CROWNVerification<double>(smallEpsOptions);
        var crown2 = new CROWNVerification<double>(largeEpsOptions);
        var mockModel = new MockPredictiveModel(3, 2);
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        // Act
        var smallResult = crown1.CertifyPrediction(input, mockModel);
        var largeResult = crown2.CertifyPrediction(input, mockModel);

        // Assert - Larger epsilon should produce wider bounds
        double smallWidth = smallResult.UpperBound - smallResult.LowerBound;
        double largeWidth = largeResult.UpperBound - largeResult.LowerBound;
        Assert.True(largeWidth >= smallWidth - Tolerance);
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void CertifyPrediction_FloatType_WorksCorrectly()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<float>
        {
            NoiseSigma = 0.01f,
            NumSamples = 20
        };
        var crown = new CROWNVerification<float>(options);
        var mockModel = new MockPredictiveModelFloat(3, 2);
        var input = new Vector<float>(new[] { 0.5f, 0.5f, 0.5f });

        // Act
        var result = crown.CertifyPrediction(input, mockModel);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.PredictedClass >= 0);
    }

    #endregion

    #region Mock Models

    /// <summary>
    /// Simple mock predictive model for testing.
    /// </summary>
    private class MockPredictiveModel : IPredictiveModel<double, Vector<double>, Vector<double>>
    {
        private readonly int _inputDim;
        private readonly int _outputDim;

        public MockPredictiveModel(int inputDim, int outputDim)
        {
            _inputDim = inputDim;
            _outputDim = outputDim;
        }

        public Vector<double> Predict(Vector<double> input)
        {
            var output = new Vector<double>(_outputDim);
            // Simple linear transformation
            double sum = 0;
            for (int i = 0; i < Math.Min(input.Length, _inputDim); i++)
            {
                sum += input[i];
            }
            output[0] = sum / _inputDim + 0.5;
            output[_outputDim - 1] = 1.0 - output[0];
            return output;
        }

        public ModelMetadata<double> GetModelMetadata() => new ModelMetadata<double>();
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
    }

    /// <summary>
    /// Mock model with configurable margin between class scores.
    /// </summary>
    private class MockPredictiveModelWithMargin : IPredictiveModel<double, Vector<double>, Vector<double>>
    {
        private readonly int _inputDim;
        private readonly int _outputDim;
        private readonly double _margin;

        public MockPredictiveModelWithMargin(int inputDim, int outputDim, double margin)
        {
            _inputDim = inputDim;
            _outputDim = outputDim;
            _margin = margin;
        }

        public Vector<double> Predict(Vector<double> input)
        {
            var output = new Vector<double>(_outputDim);
            // First class always wins with specified margin
            output[0] = 0.5 + _margin;
            for (int i = 1; i < _outputDim; i++)
            {
                output[i] = 0.5 - _margin / (_outputDim - 1);
            }
            return output;
        }

        public ModelMetadata<double> GetModelMetadata() => new ModelMetadata<double>();
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
    }

    /// <summary>
    /// Float version of mock model.
    /// </summary>
    private class MockPredictiveModelFloat : IPredictiveModel<float, Vector<float>, Vector<float>>
    {
        private readonly int _inputDim;
        private readonly int _outputDim;

        public MockPredictiveModelFloat(int inputDim, int outputDim)
        {
            _inputDim = inputDim;
            _outputDim = outputDim;
        }

        public Vector<float> Predict(Vector<float> input)
        {
            var output = new Vector<float>(_outputDim);
            float sum = 0;
            for (int i = 0; i < Math.Min(input.Length, _inputDim); i++)
            {
                sum += input[i];
            }
            output[0] = sum / _inputDim + 0.5f;
            output[_outputDim - 1] = 1.0f - output[0];
            return output;
        }

        public ModelMetadata<float> GetModelMetadata() => new ModelMetadata<float>();
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
    }

    #endregion
}
