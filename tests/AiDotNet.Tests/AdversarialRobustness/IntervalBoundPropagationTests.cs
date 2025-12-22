using AiDotNet.AdversarialRobustness.CertifiedRobustness;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.AdversarialRobustness;

/// <summary>
/// Comprehensive tests for Interval Bound Propagation (IBP) implementation.
/// Tests cover mathematical correctness of bound propagation and API functionality.
/// </summary>
public class IntervalBoundPropagationTests
{
    private const double Tolerance = 1e-6;

    #region Constructor Tests

    [Fact]
    public void Constructor_Default_CreatesInstance()
    {
        // Act
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>();

        // Assert
        Assert.NotNull(ibp);
        var options = ibp.GetOptions();
        Assert.NotNull(options);
        Assert.Equal("IBP", options.CertificationMethod);
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
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(ibp);
        var returnedOptions = ibp.GetOptions();
        Assert.Equal(0.1, returnedOptions.NoiseSigma, Tolerance);
        Assert.Equal(100, returnedOptions.NumSamples);
        Assert.Equal(0.95, returnedOptions.ConfidenceLevel, Tolerance);
        Assert.Equal("IBP", returnedOptions.CertificationMethod);
    }

    [Fact]
    public void Constructor_WithNullOptions_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => new IntervalBoundPropagation<double, Vector<double>, Vector<double>>(null!));
    }

    #endregion

    #region CertifyPrediction Tests

    [Fact]
    public void CertifyPrediction_WithNullInput_ThrowsArgumentNullException()
    {
        // Arrange
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>();
        var mockModel = new MockFullModel(3, 2);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => ibp.CertifyPrediction(null!, mockModel));
    }

    [Fact]
    public void CertifyPrediction_WithNullModel_ThrowsArgumentNullException()
    {
        // Arrange
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>();
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => ibp.CertifyPrediction(input, null!));
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
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>(options);
        var mockModel = new MockFullModel(3, 2);
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        // Act
        var result = ibp.CertifyPrediction(input, mockModel);

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
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>(options);
        var mockModel = new MockFullModelWithMargin(3, 2, 1.0); // Large margin
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        // Act
        var result = ibp.CertifyPrediction(input, mockModel);

        // Assert
        Assert.NotNull(result);
        // With small epsilon and large margin, should be certified
        Assert.True(result.IsCertified);
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
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>(options);
        var mockModel = new MockFullModel(3, 2);
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        // Act
        var result = ibp.CertifyPrediction(input, mockModel);

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
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>();
        var mockModel = new MockFullModel(3, 2);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => ibp.CertifyBatch(null!, mockModel));
    }

    [Fact]
    public void CertifyBatch_WithNullModel_ThrowsArgumentNullException()
    {
        // Arrange
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>();
        var inputs = new Vector<double>[] { new Vector<double>(3), new Vector<double>(3) };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => ibp.CertifyBatch(inputs, null!));
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
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>(options);
        var mockModel = new MockFullModel(3, 2);

        // Create batch of 5 samples
        var inputs = new Vector<double>[5];
        for (int i = 0; i < 5; i++)
        {
            inputs[i] = new Vector<double>(new[] { 0.1 * (i + 1), 0.1 * (i + 1) + 0.05, 0.1 * (i + 1) + 0.1 });
        }

        // Act
        var results = ibp.CertifyBatch(inputs, mockModel);

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
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>();
        var mockModel = new MockFullModel(3, 2);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => ibp.ComputeCertifiedRadius(null!, mockModel));
    }

    [Fact]
    public void ComputeCertifiedRadius_WithNullModel_ThrowsArgumentNullException()
    {
        // Arrange
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>();
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => ibp.ComputeCertifiedRadius(input, null!));
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
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>(options);
        var mockModel = new MockFullModelWithMargin(3, 2, 0.5);
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        // Act
        var radius = ibp.ComputeCertifiedRadius(input, mockModel);

        // Assert
        Assert.True(radius >= 0.0);
    }

    [Fact]
    public void ComputeCertifiedRadius_WithLargeMarginModel_ReturnsLargerRadius()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.1,
            NumSamples = 50
        };

        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        var ibp1 = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>(options);
        var smallMarginModel = new MockFullModelWithMargin(3, 2, 0.1);

        var ibp2 = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>(options);
        var largeMarginModel = new MockFullModelWithMargin(3, 2, 1.0);

        // Act
        var smallRadius = ibp1.ComputeCertifiedRadius(input, smallMarginModel);
        var largeRadius = ibp2.ComputeCertifiedRadius(input, largeMarginModel);

        // Assert
        // Model with larger margin should have larger certified radius
        Assert.True(largeRadius >= smallRadius);
    }

    #endregion

    #region EvaluateCertifiedAccuracy Tests

    [Fact]
    public void EvaluateCertifiedAccuracy_WithNullTestData_ThrowsArgumentNullException()
    {
        // Arrange
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>();
        var mockModel = new MockFullModel(3, 2);
        var labels = new Vector<double>[] { new Vector<double>(new[] { 1.0, 0.0 }), new Vector<double>(new[] { 0.0, 1.0 }), new Vector<double>(new[] { 1.0, 0.0 }) };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            ibp.EvaluateCertifiedAccuracy(null!, labels, mockModel, 0.1));
    }

    [Fact]
    public void EvaluateCertifiedAccuracy_WithNullLabels_ThrowsArgumentNullException()
    {
        // Arrange
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>();
        var mockModel = new MockFullModel(3, 2);
        var testData = new Vector<double>[] { new Vector<double>(3), new Vector<double>(3), new Vector<double>(3) };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            ibp.EvaluateCertifiedAccuracy(testData, null!, mockModel, 0.1));
    }

    [Fact]
    public void EvaluateCertifiedAccuracy_WithNullModel_ThrowsArgumentNullException()
    {
        // Arrange
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>();
        var testData = new Vector<double>[] { new Vector<double>(3), new Vector<double>(3), new Vector<double>(3) };
        var labels = new Vector<double>[] { new Vector<double>(new[] { 1.0, 0.0 }), new Vector<double>(new[] { 0.0, 1.0 }), new Vector<double>(new[] { 1.0, 0.0 }) };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            ibp.EvaluateCertifiedAccuracy(testData, labels, null!, 0.1));
    }

    [Fact]
    public void EvaluateCertifiedAccuracy_WithMismatchedDimensions_ThrowsArgumentException()
    {
        // Arrange
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>();
        var mockModel = new MockFullModel(3, 2);
        var testData = new Vector<double>[5];
        for (int i = 0; i < 5; i++)
        {
            testData[i] = new Vector<double>(3);
        }
        var labels = new Vector<double>[] { new Vector<double>(new[] { 1.0, 0.0 }), new Vector<double>(new[] { 0.0, 1.0 }), new Vector<double>(new[] { 1.0, 0.0 }) }; // Only 3 labels for 5 samples

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            ibp.EvaluateCertifiedAccuracy(testData, labels, mockModel, 0.1));
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
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>(options);
        var mockModel = new MockFullModelWithMargin(3, 2, 0.5);

        // Create test data
        var testData = new Vector<double>[4];
        for (int i = 0; i < 4; i++)
        {
            testData[i] = new Vector<double>(new[] { 0.5, 0.5, 0.5 });
        }

        // Labels matching the model's predictions (class 0 wins with margin)
        var labels = new Vector<double>[4];
        for (int i = 0; i < 4; i++)
        {
            labels[i] = new Vector<double>(new[] { 1.0, 0.0 });
        }

        // Act
        var metrics = ibp.EvaluateCertifiedAccuracy(testData, labels, mockModel, 0.01);

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
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>(options);

        // Act
        var bytes = ibp.Serialize();

        // Assert
        Assert.NotNull(bytes);
        Assert.True(bytes.Length > 0);
    }

    [Fact]
    public void Deserialize_WithNullData_ThrowsArgumentNullException()
    {
        // Arrange
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => ibp.Deserialize(null!));
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
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>(options);

        // Act
        var bytes = ibp.Serialize();
        var ibp2 = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>();
        ibp2.Deserialize(bytes);
        var restoredOptions = ibp2.GetOptions();

        // Assert
        Assert.Equal(0.15, restoredOptions.NoiseSigma, Tolerance);
        Assert.Equal(75, restoredOptions.NumSamples);
        Assert.Equal(0.9, restoredOptions.ConfidenceLevel, Tolerance);
        Assert.Equal("L2", restoredOptions.NormType);
        Assert.True(restoredOptions.UseTightBounds);
        Assert.Equal(32, restoredOptions.BatchSize);
        Assert.Equal(42, restoredOptions.RandomSeed);
    }

    #endregion

    #region SaveModel/LoadModel Tests

    [Fact]
    public void SaveModel_WithNullPath_ThrowsArgumentException()
    {
        // Arrange
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => ibp.SaveModel(null!));
    }

    [Fact]
    public void SaveModel_WithEmptyPath_ThrowsArgumentException()
    {
        // Arrange
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => ibp.SaveModel(string.Empty));
    }

    [Fact]
    public void LoadModel_WithNullPath_ThrowsArgumentException()
    {
        // Arrange
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => ibp.LoadModel(null!));
    }

    [Fact]
    public void LoadModel_WithNonExistentFile_ThrowsFileNotFoundException()
    {
        // Arrange
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>();

        // Act & Assert
        Assert.Throws<FileNotFoundException>(() => ibp.LoadModel("nonexistent_ibp_model_12345.json"));
    }

    [Fact]
    public void SaveAndLoadModel_RoundTrip_PreservesOptions()
    {
        // Arrange
        var tempPath = Path.Combine(Path.GetTempPath(), $"ibp_test_{Guid.NewGuid()}.json");
        try
        {
            var options = new CertifiedDefenseOptions<double>
            {
                NoiseSigma = 0.2,
                NumSamples = 150,
                ConfidenceLevel = 0.95
            };
            var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>(options);

            // Act
            ibp.SaveModel(tempPath);
            var ibp2 = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>();
            ibp2.LoadModel(tempPath);
            var loadedOptions = ibp2.GetOptions();

            // Assert
            Assert.Equal(0.2, loadedOptions.NoiseSigma, Tolerance);
            Assert.Equal(150, loadedOptions.NumSamples);
            Assert.Equal(0.95, loadedOptions.ConfidenceLevel, Tolerance);
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
            NumSamples = 200
        };
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>(options);

        // Act
        ibp.Reset();
        var resetOptions = ibp.GetOptions();

        // Assert
        Assert.Equal("IBP", resetOptions.CertificationMethod);
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
        var ibp = new IntervalBoundPropagation<float, Vector<float>, Vector<float>>(options);
        var mockModel = new MockFullModelFloat(3, 2);
        var input = new Vector<float>(new[] { 0.5f, 0.5f, 0.5f });

        // Act
        var result = ibp.CertifyPrediction(input, mockModel);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.PredictedClass >= 0);
    }

    #endregion

    #region Mock Models

    /// <summary>
    /// Simple mock full model for testing.
    /// </summary>
    private class MockFullModel : IFullModel<double, Vector<double>, Vector<double>>
    {
        private readonly int _inputDim;
        private readonly int _outputDim;
        private List<int> _activeFeatures;

        public MockFullModel(int inputDim, int outputDim)
        {
            _inputDim = inputDim;
            _outputDim = outputDim;
            _activeFeatures = Enumerable.Range(0, inputDim).ToList();
        }

        public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();

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

        public void Train(Vector<double> input, Vector<double> expectedOutput) { }
        public ModelMetadata<double> GetModelMetadata() => new ModelMetadata<double>();
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
        public void SaveState(Stream stream) { }
        public void LoadState(Stream stream) { }
        public IFullModel<double, Vector<double>, Vector<double>> Clone() => new MockFullModel(_inputDim, _outputDim);
        public IFullModel<double, Vector<double>, Vector<double>> DeepCopy() => new MockFullModel(_inputDim, _outputDim);
        public Vector<double> ComputeGradients(Vector<double> input, Vector<double> target, ILossFunction<double>? lossFunction = null) => new Vector<double>(_inputDim);
        public void ApplyGradients(Vector<double> gradients, double learningRate) { }
        public Vector<double> GetParameters() => new Vector<double>(_inputDim * _outputDim);
        public void SetParameters(Vector<double> parameters) { }
        public IFullModel<double, Vector<double>, Vector<double>> WithParameters(Vector<double> parameters) => new MockFullModel(_inputDim, _outputDim);
        public int ParameterCount => _inputDim * _outputDim;
        public IEnumerable<int> GetActiveFeatureIndices() => _activeFeatures;
        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices) => _activeFeatures = featureIndices.ToList();
        public bool IsFeatureUsed(int featureIndex) => _activeFeatures.Contains(featureIndex);
        public Dictionary<string, double> GetFeatureImportance() => Enumerable.Range(0, _inputDim).ToDictionary(i => $"Feature{i}", i => 1.0 / _inputDim);
        public bool SupportsJitCompilation => false;
        public ComputationNode<double> ExportComputationGraph(List<ComputationNode<double>> inputNodes) => throw new NotSupportedException();
    }

    /// <summary>
    /// Mock model with configurable margin between class scores.
    /// </summary>
    private class MockFullModelWithMargin : IFullModel<double, Vector<double>, Vector<double>>
    {
        private readonly int _inputDim;
        private readonly int _outputDim;
        private readonly double _margin;
        private List<int> _activeFeatures;

        public MockFullModelWithMargin(int inputDim, int outputDim, double margin)
        {
            _inputDim = inputDim;
            _outputDim = outputDim;
            _margin = margin;
            _activeFeatures = Enumerable.Range(0, inputDim).ToList();
        }

        public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();

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

        public void Train(Vector<double> input, Vector<double> expectedOutput) { }
        public ModelMetadata<double> GetModelMetadata() => new ModelMetadata<double>();
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
        public void SaveState(Stream stream) { }
        public void LoadState(Stream stream) { }
        public IFullModel<double, Vector<double>, Vector<double>> Clone() => new MockFullModelWithMargin(_inputDim, _outputDim, _margin);
        public IFullModel<double, Vector<double>, Vector<double>> DeepCopy() => new MockFullModelWithMargin(_inputDim, _outputDim, _margin);
        public Vector<double> ComputeGradients(Vector<double> input, Vector<double> target, ILossFunction<double>? lossFunction = null) => new Vector<double>(_inputDim);
        public void ApplyGradients(Vector<double> gradients, double learningRate) { }
        public Vector<double> GetParameters() => new Vector<double>(_inputDim * _outputDim);
        public void SetParameters(Vector<double> parameters) { }
        public IFullModel<double, Vector<double>, Vector<double>> WithParameters(Vector<double> parameters) => new MockFullModelWithMargin(_inputDim, _outputDim, _margin);
        public int ParameterCount => _inputDim * _outputDim;
        public IEnumerable<int> GetActiveFeatureIndices() => _activeFeatures;
        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices) => _activeFeatures = featureIndices.ToList();
        public bool IsFeatureUsed(int featureIndex) => _activeFeatures.Contains(featureIndex);
        public Dictionary<string, double> GetFeatureImportance() => Enumerable.Range(0, _inputDim).ToDictionary(i => $"Feature{i}", i => 1.0 / _inputDim);
        public bool SupportsJitCompilation => false;
        public ComputationNode<double> ExportComputationGraph(List<ComputationNode<double>> inputNodes) => throw new NotSupportedException();
    }

    /// <summary>
    /// Float version of mock model.
    /// </summary>
    private class MockFullModelFloat : IFullModel<float, Vector<float>, Vector<float>>
    {
        private readonly int _inputDim;
        private readonly int _outputDim;
        private List<int> _activeFeatures;

        public MockFullModelFloat(int inputDim, int outputDim)
        {
            _inputDim = inputDim;
            _outputDim = outputDim;
            _activeFeatures = Enumerable.Range(0, inputDim).ToList();
        }

        public ILossFunction<float> DefaultLossFunction => new MeanSquaredErrorLoss<float>();

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

        public void Train(Vector<float> input, Vector<float> expectedOutput) { }
        public ModelMetadata<float> GetModelMetadata() => new ModelMetadata<float>();
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
        public void SaveState(Stream stream) { }
        public void LoadState(Stream stream) { }
        public IFullModel<float, Vector<float>, Vector<float>> Clone() => new MockFullModelFloat(_inputDim, _outputDim);
        public IFullModel<float, Vector<float>, Vector<float>> DeepCopy() => new MockFullModelFloat(_inputDim, _outputDim);
        public Vector<float> ComputeGradients(Vector<float> input, Vector<float> target, ILossFunction<float>? lossFunction = null) => new Vector<float>(_inputDim);
        public void ApplyGradients(Vector<float> gradients, float learningRate) { }
        public Vector<float> GetParameters() => new Vector<float>(_inputDim * _outputDim);
        public void SetParameters(Vector<float> parameters) { }
        public IFullModel<float, Vector<float>, Vector<float>> WithParameters(Vector<float> parameters) => new MockFullModelFloat(_inputDim, _outputDim);
        public int ParameterCount => _inputDim * _outputDim;
        public IEnumerable<int> GetActiveFeatureIndices() => _activeFeatures;
        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices) => _activeFeatures = featureIndices.ToList();
        public bool IsFeatureUsed(int featureIndex) => _activeFeatures.Contains(featureIndex);
        public Dictionary<string, float> GetFeatureImportance() => Enumerable.Range(0, _inputDim).ToDictionary(i => $"Feature{i}", i => 1.0f / _inputDim);
        public bool SupportsJitCompilation => false;
        public ComputationNode<float> ExportComputationGraph(List<ComputationNode<float>> inputNodes) => throw new NotSupportedException();
    }

    #endregion
}
