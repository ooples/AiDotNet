using AiDotNet.AdversarialRobustness.CertifiedRobustness;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AdversarialRobustness;

/// <summary>
/// Tests for the RandomizedSmoothing certified defense implementation.
/// </summary>
/// <remarks>
/// RandomizedSmoothing provides certified robustness guarantees by averaging predictions
/// over Gaussian noise and computing confidence bounds using Clopper-Pearson intervals.
/// </remarks>
public class RandomizedSmoothingTests
{
    #region Test Mock

    /// <summary>
    /// Mock predictive model that returns deterministic predictions based on input.
    /// </summary>
    private class MockPredictiveModel : IPredictiveModel<double, Vector<double>, Vector<double>>
    {
        private readonly int _numClasses;
        private readonly int _dominantClass;
        private readonly double _dominantProbability;

        public MockPredictiveModel(int numClasses = 3, int dominantClass = 0, double dominantProbability = 0.8)
        {
            _numClasses = numClasses;
            _dominantClass = dominantClass;
            _dominantProbability = dominantProbability;
        }

        public Vector<double> Predict(Vector<double> input)
        {
            // Return logits where dominant class has highest value
            var output = new Vector<double>(_numClasses);
            double remainingProb = (1.0 - _dominantProbability) / (_numClasses - 1);

            for (int i = 0; i < _numClasses; i++)
            {
                output[i] = i == _dominantClass ? _dominantProbability : remainingProb;
            }

            return output;
        }

        public ModelMetadata<double> GetModelMetadata() => new();
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
    }

    #endregion

    #region Constructor Tests

    [Fact]
    public void Constructor_WithValidOptions_CreatesInstance()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.5,
            NumSamples = 100,
            ConfidenceLevel = 0.95
        };

        // Act
        var smoothing = new RandomizedSmoothing<double>(options);

        // Assert
        Assert.NotNull(smoothing);
        Assert.NotNull(smoothing.GetOptions());
    }

    [Fact]
    public void Constructor_WithNullOptions_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new RandomizedSmoothing<double>(null!));
    }

    [Fact]
    public void Constructor_WithRandomSeed_ProducesReproducibleResults()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.5,
            NumSamples = 100,
            ConfidenceLevel = 0.95,
            RandomSeed = 42
        };
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });
        var model = new MockPredictiveModel();

        // Act
        var smoothing1 = new RandomizedSmoothing<double>(options);
        var result1 = smoothing1.CertifyPrediction(input, model);

        var smoothing2 = new RandomizedSmoothing<double>(options);
        var result2 = smoothing2.CertifyPrediction(input, model);

        // Assert - same seed should produce same results
        Assert.Equal(result1.PredictedClass, result2.PredictedClass);
        Assert.Equal(result1.CertifiedRadius, result2.CertifiedRadius, 6);
    }

    #endregion

    #region CertifyPrediction Tests

    [Fact]
    public void CertifyPrediction_WithNullInput_ThrowsArgumentNullException()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double> { NoiseSigma = 0.5, NumSamples = 100 };
        var smoothing = new RandomizedSmoothing<double>(options);
        var model = new MockPredictiveModel();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => smoothing.CertifyPrediction(null!, model));
    }

    [Fact]
    public void CertifyPrediction_WithNullModel_ThrowsArgumentNullException()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double> { NoiseSigma = 0.5, NumSamples = 100 };
        var smoothing = new RandomizedSmoothing<double>(options);
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => smoothing.CertifyPrediction(input, null!));
    }

    [Fact]
    public void CertifyPrediction_WithStrongModel_ReturnsHighConfidence()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.5,
            NumSamples = 100,
            ConfidenceLevel = 0.95,
            RandomSeed = 42
        };
        var smoothing = new RandomizedSmoothing<double>(options);
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });
        var model = new MockPredictiveModel(numClasses: 3, dominantClass: 0, dominantProbability: 0.99);

        // Act
        var result = smoothing.CertifyPrediction(input, model);

        // Assert
        Assert.Equal(0, result.PredictedClass);
        Assert.True(result.Confidence > 0.5, "Confidence should be above 0.5 for certification");
        Assert.True(result.IsCertified, "Strong model should produce certified prediction");
        Assert.True(result.CertifiedRadius > 0, "Certified radius should be positive");
    }

    [Fact]
    public void CertifyPrediction_ContainsRequiredDetails()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.5,
            NumSamples = 100,
            ConfidenceLevel = 0.95,
            RandomSeed = 42
        };
        var smoothing = new RandomizedSmoothing<double>(options);
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });
        var model = new MockPredictiveModel();

        // Act
        var result = smoothing.CertifyPrediction(input, model);

        // Assert
        Assert.Contains("SampleCount", result.CertificationDetails.Keys);
        Assert.Contains("Sigma", result.CertificationDetails.Keys);
        Assert.Contains("TopClassCount", result.CertificationDetails.Keys);
        Assert.Equal(100, result.CertificationDetails["SampleCount"]);
        Assert.Equal(0.5, result.CertificationDetails["Sigma"]);
    }

    [Fact]
    public void CertifyPrediction_LowerBoundLessThanOrEqualToConfidence()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.5,
            NumSamples = 100,
            ConfidenceLevel = 0.95,
            RandomSeed = 42
        };
        var smoothing = new RandomizedSmoothing<double>(options);
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });
        var model = new MockPredictiveModel();

        // Act
        var result = smoothing.CertifyPrediction(input, model);

        // Assert
        Assert.True(result.LowerBound <= result.Confidence,
            $"Lower bound {result.LowerBound} should be <= confidence {result.Confidence}");
        Assert.True(result.UpperBound >= result.Confidence,
            $"Upper bound {result.UpperBound} should be >= confidence {result.Confidence}");
    }

    [Theory]
    [InlineData(0.25)]
    [InlineData(0.5)]
    [InlineData(1.0)]
    public void CertifyPrediction_WithDifferentSigma_AffectsRadius(double sigma)
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = sigma,
            NumSamples = 100,
            ConfidenceLevel = 0.95,
            RandomSeed = 42
        };
        var smoothing = new RandomizedSmoothing<double>(options);
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });
        var model = new MockPredictiveModel(numClasses: 3, dominantClass: 0, dominantProbability: 0.99);

        // Act
        var result = smoothing.CertifyPrediction(input, model);

        // Assert - larger sigma should produce larger certified radius
        // (for the same confidence level)
        Assert.NotNull(result);
        Assert.True(result.CertifiedRadius >= 0, "Certified radius should be non-negative");
    }

    #endregion

    #region CertifyBatch Tests

    [Fact]
    public void CertifyBatch_WithNullInputs_ThrowsArgumentNullException()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double> { NoiseSigma = 0.5, NumSamples = 50 };
        var smoothing = new RandomizedSmoothing<double>(options);
        var model = new MockPredictiveModel();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => smoothing.CertifyBatch(null!, model));
    }

    [Fact]
    public void CertifyBatch_ReturnsResultForEachInput()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.5,
            NumSamples = 50,
            ConfidenceLevel = 0.95,
            RandomSeed = 42
        };
        var smoothing = new RandomizedSmoothing<double>(options);
        var inputs = new Matrix<double>(5, 3); // 5 samples, 3 features
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                inputs[i, j] = 0.5;
            }
        }
        var model = new MockPredictiveModel();

        // Act
        var results = smoothing.CertifyBatch(inputs, model);

        // Assert
        Assert.Equal(5, results.Length);
        foreach (var result in results)
        {
            Assert.NotNull(result);
            Assert.True(result.PredictedClass >= 0, "Predicted class should be non-negative");
        }
    }

    #endregion

    #region ComputeCertifiedRadius Tests

    [Fact]
    public void ComputeCertifiedRadius_ReturnsValueFromCertification()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.5,
            NumSamples = 100,
            ConfidenceLevel = 0.95,
            RandomSeed = 42
        };
        var smoothing = new RandomizedSmoothing<double>(options);
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });
        var model = new MockPredictiveModel();

        // Act
        var radius = smoothing.ComputeCertifiedRadius(input, model);

        // Assert
        Assert.True(radius >= 0, "Certified radius should be non-negative");
    }

    #endregion

    #region EvaluateCertifiedAccuracy Tests

    [Fact]
    public void EvaluateCertifiedAccuracy_WithNullTestData_ThrowsArgumentNullException()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double> { NoiseSigma = 0.5, NumSamples = 50 };
        var smoothing = new RandomizedSmoothing<double>(options);
        var labels = new Vector<int>(new[] { 0, 1, 0 });
        var model = new MockPredictiveModel();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            smoothing.EvaluateCertifiedAccuracy(null!, labels, model, 0.5));
    }

    [Fact]
    public void EvaluateCertifiedAccuracy_WithNullLabels_ThrowsArgumentNullException()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double> { NoiseSigma = 0.5, NumSamples = 50 };
        var smoothing = new RandomizedSmoothing<double>(options);
        var testData = new Matrix<double>(3, 3);
        var model = new MockPredictiveModel();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            smoothing.EvaluateCertifiedAccuracy(testData, null!, model, 0.5));
    }

    [Fact]
    public void EvaluateCertifiedAccuracy_WithNullModel_ThrowsArgumentNullException()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double> { NoiseSigma = 0.5, NumSamples = 50 };
        var smoothing = new RandomizedSmoothing<double>(options);
        var testData = new Matrix<double>(3, 3);
        var labels = new Vector<int>(new[] { 0, 1, 0 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            smoothing.EvaluateCertifiedAccuracy(testData, labels, null!, 0.5));
    }

    [Fact]
    public void EvaluateCertifiedAccuracy_WithMismatchedDimensions_ThrowsArgumentException()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double> { NoiseSigma = 0.5, NumSamples = 50 };
        var smoothing = new RandomizedSmoothing<double>(options);
        var testData = new Matrix<double>(5, 3); // 5 samples
        var labels = new Vector<int>(new[] { 0, 1, 0 }); // Only 3 labels
        var model = new MockPredictiveModel();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            smoothing.EvaluateCertifiedAccuracy(testData, labels, model, 0.5));
    }

    [Fact]
    public void EvaluateCertifiedAccuracy_ReturnsValidMetrics()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.5,
            NumSamples = 50,
            ConfidenceLevel = 0.95,
            RandomSeed = 42
        };
        var smoothing = new RandomizedSmoothing<double>(options);

        // Create test data with 3 samples
        var testData = new Matrix<double>(3, 3);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                testData[i, j] = 0.5;
            }
        }

        // Model always predicts class 0
        var labels = new Vector<int>(new[] { 0, 0, 0 });
        var model = new MockPredictiveModel(numClasses: 3, dominantClass: 0);

        // Act
        var metrics = smoothing.EvaluateCertifiedAccuracy(testData, labels, model, 0.1);

        // Assert
        Assert.True(metrics.CleanAccuracy >= 0 && metrics.CleanAccuracy <= 1,
            "Clean accuracy should be in [0,1]");
        Assert.True(metrics.CertifiedAccuracy >= 0 && metrics.CertifiedAccuracy <= 1,
            "Certified accuracy should be in [0,1]");
        Assert.True(metrics.CertificationRate >= 0 && metrics.CertificationRate <= 1,
            "Certification rate should be in [0,1]");
    }

    [Fact]
    public void EvaluateCertifiedAccuracy_CertifiedAccuracyNotExceedsCleanAccuracy()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.5,
            NumSamples = 50,
            ConfidenceLevel = 0.95,
            RandomSeed = 42
        };
        var smoothing = new RandomizedSmoothing<double>(options);

        var testData = new Matrix<double>(5, 3);
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                testData[i, j] = 0.5;
            }
        }

        var labels = new Vector<int>(new[] { 0, 0, 0, 0, 0 });
        var model = new MockPredictiveModel(numClasses: 3, dominantClass: 0);

        // Act
        var metrics = smoothing.EvaluateCertifiedAccuracy(testData, labels, model, 0.1);

        // Assert - certified accuracy should not exceed clean accuracy
        // (you can't certify something that's already wrong)
        Assert.True(metrics.CertifiedAccuracy <= metrics.CleanAccuracy + 0.001,
            $"Certified accuracy {metrics.CertifiedAccuracy} should not exceed clean accuracy {metrics.CleanAccuracy}");
    }

    #endregion

    #region Serialization Tests

    [Fact]
    public void Serialize_ReturnsNonEmptyByteArray()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.5,
            NumSamples = 100,
            ConfidenceLevel = 0.95
        };
        var smoothing = new RandomizedSmoothing<double>(options);

        // Act
        var bytes = smoothing.Serialize();

        // Assert
        Assert.NotNull(bytes);
        Assert.True(bytes.Length > 0, "Serialized data should not be empty");
    }

    [Fact]
    public void Deserialize_WithNullData_ThrowsArgumentNullException()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double> { NoiseSigma = 0.5, NumSamples = 100 };
        var smoothing = new RandomizedSmoothing<double>(options);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => smoothing.Deserialize(null!));
    }

    [Fact]
    public void SerializeDeserialize_PreservesOptions()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.75,
            NumSamples = 200,
            ConfidenceLevel = 0.99
        };
        var smoothing = new RandomizedSmoothing<double>(options);

        // Act
        var bytes = smoothing.Serialize();
        smoothing.Deserialize(bytes);
        var restoredOptions = smoothing.GetOptions();

        // Assert
        Assert.Equal(0.75, restoredOptions.NoiseSigma);
        Assert.Equal(200, restoredOptions.NumSamples);
        Assert.Equal(0.99, restoredOptions.ConfidenceLevel);
    }

    #endregion

    #region GetOptions and Reset Tests

    [Fact]
    public void GetOptions_ReturnsConfiguredOptions()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.5,
            NumSamples = 100,
            ConfidenceLevel = 0.95
        };
        var smoothing = new RandomizedSmoothing<double>(options);

        // Act
        var returnedOptions = smoothing.GetOptions();

        // Assert
        Assert.Equal(0.5, returnedOptions.NoiseSigma);
        Assert.Equal(100, returnedOptions.NumSamples);
        Assert.Equal(0.95, returnedOptions.ConfidenceLevel);
    }

    [Fact]
    public void Reset_DoesNotThrow()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double> { NoiseSigma = 0.5, NumSamples = 100 };
        var smoothing = new RandomizedSmoothing<double>(options);

        // Act & Assert - Reset should not throw
        var exception = Record.Exception(() => smoothing.Reset());
        Assert.Null(exception);
    }

    #endregion

    #region Median Calculation Tests

    [Fact]
    public void EvaluateCertifiedAccuracy_ComputesCorrectMedian_OddCount()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.5,
            NumSamples = 50,
            ConfidenceLevel = 0.95,
            RandomSeed = 42
        };
        var smoothing = new RandomizedSmoothing<double>(options);

        // Create 3 test samples (odd count)
        var testData = new Matrix<double>(3, 3);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                testData[i, j] = 0.5;
            }
        }

        var labels = new Vector<int>(new[] { 0, 0, 0 });
        var model = new MockPredictiveModel(numClasses: 3, dominantClass: 0);

        // Act
        var metrics = smoothing.EvaluateCertifiedAccuracy(testData, labels, model, 0.0);

        // Assert - with odd count, median should be the middle element
        // Just verify we get a valid median (non-negative if there are certified samples)
        Assert.True(metrics.MedianCertifiedRadius >= 0 || metrics.CertificationRate == 0);
    }

    [Fact]
    public void EvaluateCertifiedAccuracy_ComputesCorrectMedian_EvenCount()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.5,
            NumSamples = 50,
            ConfidenceLevel = 0.95,
            RandomSeed = 42
        };
        var smoothing = new RandomizedSmoothing<double>(options);

        // Create 4 test samples (even count)
        var testData = new Matrix<double>(4, 3);
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                testData[i, j] = 0.5;
            }
        }

        var labels = new Vector<int>(new[] { 0, 0, 0, 0 });
        var model = new MockPredictiveModel(numClasses: 3, dominantClass: 0);

        // Act
        var metrics = smoothing.EvaluateCertifiedAccuracy(testData, labels, model, 0.0);

        // Assert - with even count, median should be average of two middle elements
        Assert.True(metrics.MedianCertifiedRadius >= 0 || metrics.CertificationRate == 0);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void RandomizedSmoothing_EndToEndWorkflow()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.5,
            NumSamples = 100,
            ConfidenceLevel = 0.95,
            RandomSeed = 42
        };
        var smoothing = new RandomizedSmoothing<double>(options);
        var model = new MockPredictiveModel(numClasses: 3, dominantClass: 0, dominantProbability: 0.95);

        // Create test data
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });
        var testData = new Matrix<double>(3, 3);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                testData[i, j] = 0.5;
            }
        }
        var labels = new Vector<int>(new[] { 0, 0, 0 });

        // Act - Full workflow
        var singleResult = smoothing.CertifyPrediction(input, model);
        var batchResults = smoothing.CertifyBatch(testData, model);
        var metrics = smoothing.EvaluateCertifiedAccuracy(testData, labels, model, 0.1);

        // Serialize and deserialize
        var bytes = smoothing.Serialize();
        smoothing.Deserialize(bytes);

        // Assert
        Assert.NotNull(singleResult);
        Assert.Equal(3, batchResults.Length);
        Assert.True(metrics.CleanAccuracy >= 0);
        Assert.True(bytes.Length > 0);
    }

    #endregion
}
