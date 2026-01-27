using AiDotNet.OutlierRemoval;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.OutlierRemoval;

/// <summary>
/// Unit tests for algorithmic outlier detectors.
/// Tests IsolationForestOutlierDetector, LocalOutlierFactorDetector,
/// OneClassSVMOutlierDetector, and AutoencoderOutlierDetector.
/// </summary>
public class OutlierDetectorTests
{
    private const double Tolerance = 1e-6;

    #region IsolationForest Tests

    [Fact]
    public void IsolationForest_Fit_SetsIsFittedToTrue()
    {
        // Arrange
        var detector = new IsolationForestOutlierDetector<double>();
        var X = CreateNormalData(100, 3);

        // Act
        detector.Fit(X);

        // Assert
        Assert.True(detector.IsFitted);
    }

    [Fact]
    public void IsolationForest_Predict_BeforeFit_ThrowsException()
    {
        // Arrange
        var detector = new IsolationForestOutlierDetector<double>();
        var X = CreateNormalData(10, 2);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => detector.Predict(X));
    }

    [Fact]
    public void IsolationForest_Predict_ReturnsCorrectDimensions()
    {
        // Arrange
        var detector = new IsolationForestOutlierDetector<double>();
        var X = CreateNormalData(100, 3);
        var XTest = CreateNormalData(20, 3);

        // Act
        detector.Fit(X);
        var predictions = detector.Predict(XTest);

        // Assert
        Assert.Equal(20, predictions.Length);
    }

    [Fact]
    public void IsolationForest_Predict_ReturnsValidLabels()
    {
        // Arrange
        var detector = new IsolationForestOutlierDetector<double>();
        var X = CreateNormalData(100, 3);

        // Act
        detector.Fit(X);
        var predictions = detector.Predict(X);

        // Assert - All predictions should be 1 or -1
        foreach (var pred in predictions.ToArray())
        {
            Assert.True(Math.Abs(pred - 1.0) < Tolerance || Math.Abs(pred + 1.0) < Tolerance,
                $"Prediction {pred} is not 1 or -1");
        }
    }

    [Fact]
    public void IsolationForest_DetectsObviousOutliers()
    {
        // Arrange
        var detector = new IsolationForestOutlierDetector<double>(contamination: 0.1, randomSeed: 42);
        var X = CreateDataWithOutliers(100, 2, 10); // 100 normal + 10 outliers

        // Act
        detector.Fit(X);
        var predictions = detector.Predict(X);

        // Assert - Count outliers (predictions = -1)
        int outlierCount = predictions.ToArray().Count(p => p < 0);
        // Should detect roughly the contamination proportion
        Assert.True(outlierCount >= 5 && outlierCount <= 20,
            $"Expected 5-20 outliers, got {outlierCount}");
    }

    [Fact]
    public void IsolationForest_SameSeed_ProducesSameResults()
    {
        // Arrange
        var detector1 = new IsolationForestOutlierDetector<double>(randomSeed: 123);
        var detector2 = new IsolationForestOutlierDetector<double>(randomSeed: 123);
        var X = CreateNormalData(50, 2);

        // Act
        detector1.Fit(X);
        detector2.Fit(X);
        var scores1 = detector1.DecisionFunction(X);
        var scores2 = detector2.DecisionFunction(X);

        // Assert
        for (int i = 0; i < scores1.Length; i++)
        {
            Assert.Equal(scores1[i], scores2[i], Tolerance);
        }
    }

    [Fact]
    public void IsolationForest_InvalidNumTrees_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new IsolationForestOutlierDetector<double>(numTrees: 0));
    }

    [Fact]
    public void IsolationForest_InvalidMaxSamples_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new IsolationForestOutlierDetector<double>(maxSamples: 0));
    }

    [Fact]
    public void IsolationForest_NullInput_ThrowsException()
    {
        // Arrange
        var detector = new IsolationForestOutlierDetector<double>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => detector.Fit(null!));
    }

    #endregion

    #region LocalOutlierFactor Tests

    [Fact]
    public void LOF_Fit_SetsIsFittedToTrue()
    {
        // Arrange
        var detector = new LocalOutlierFactorDetector<double>(numNeighbors: 5);
        var X = CreateNormalData(50, 3);

        // Act
        detector.Fit(X);

        // Assert
        Assert.True(detector.IsFitted);
    }

    [Fact]
    public void LOF_Predict_BeforeFit_ThrowsException()
    {
        // Arrange
        var detector = new LocalOutlierFactorDetector<double>();
        var X = CreateNormalData(10, 2);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => detector.Predict(X));
    }

    [Fact]
    public void LOF_Predict_ReturnsCorrectDimensions()
    {
        // Arrange
        var detector = new LocalOutlierFactorDetector<double>(numNeighbors: 5);
        var X = CreateNormalData(50, 3);
        var XTest = CreateNormalData(10, 3);

        // Act
        detector.Fit(X);
        var predictions = detector.Predict(XTest);

        // Assert
        Assert.Equal(10, predictions.Length);
    }

    [Fact]
    public void LOF_DetectsLocalOutliers()
    {
        // Arrange - Create a simple cluster with one obvious outlier
        var detector = new LocalOutlierFactorDetector<double>(numNeighbors: 5, contamination: 0.2);

        // Create tight cluster of normal points
        var X = new Matrix<double>(new double[,]
        {
            { 0.0, 0.0 },
            { 0.1, 0.1 },
            { 0.2, 0.0 },
            { 0.0, 0.2 },
            { 0.1, 0.2 },
            { 0.2, 0.1 },
            { 0.15, 0.15 },
            { 0.05, 0.05 },
            { 0.25, 0.25 },
            { 100.0, 100.0 }  // Obvious outlier far from cluster
        });

        // Act
        detector.Fit(X);
        var predictions = detector.Predict(X);
        var scores = detector.DecisionFunction(X);

        // Assert - The outlier (last point) should be detected
        // With contamination 0.2, roughly 2 points should be marked as outliers
        int outlierCount = predictions.ToArray().Count(p => p < 0);
        Assert.True(outlierCount >= 1 && outlierCount <= 3,
            $"Expected 1-3 outliers with 20% contamination on 10 points, got {outlierCount}");

        // The last point (obvious outlier) should have a lower score than the cluster
        double clusterAvgScore = 0;
        for (int i = 0; i < 9; i++)
        {
            clusterAvgScore += scores[i];
        }
        clusterAvgScore /= 9;
        double outlierScore = scores[9];

        // Outlier score should be more negative (indicating anomaly)
        Assert.True(outlierScore <= clusterAvgScore,
            $"Outlier score ({outlierScore:F4}) should be <= cluster avg ({clusterAvgScore:F4})");
    }

    [Fact]
    public void LOF_TooFewSamples_ThrowsException()
    {
        // Arrange
        var detector = new LocalOutlierFactorDetector<double>(numNeighbors: 20);
        var X = CreateNormalData(10, 2); // Only 10 samples, but need > 20 neighbors

        // Act & Assert
        Assert.Throws<ArgumentException>(() => detector.Fit(X));
    }

    [Fact]
    public void LOF_InvalidNumNeighbors_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new LocalOutlierFactorDetector<double>(numNeighbors: 0));
    }

    [Fact]
    public void LOF_NullInput_ThrowsException()
    {
        // Arrange
        var detector = new LocalOutlierFactorDetector<double>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => detector.Fit(null!));
    }

    #endregion

    #region OneClassSVM Tests

    [Fact]
    public void OneClassSVM_Fit_SetsIsFittedToTrue()
    {
        // Arrange
        var detector = new OneClassSVMOutlierDetector<double>();
        var X = CreateNormalData(50, 3);

        // Act
        detector.Fit(X);

        // Assert
        Assert.True(detector.IsFitted);
    }

    [Fact]
    public void OneClassSVM_Predict_BeforeFit_ThrowsException()
    {
        // Arrange
        var detector = new OneClassSVMOutlierDetector<double>();
        var X = CreateNormalData(10, 2);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => detector.Predict(X));
    }

    [Fact]
    public void OneClassSVM_Predict_ReturnsCorrectDimensions()
    {
        // Arrange
        var detector = new OneClassSVMOutlierDetector<double>();
        var X = CreateNormalData(50, 3);
        var XTest = CreateNormalData(10, 3);

        // Act
        detector.Fit(X);
        var predictions = detector.Predict(XTest);

        // Assert
        Assert.Equal(10, predictions.Length);
    }

    [Fact]
    public void OneClassSVM_Predict_ReturnsValidLabels()
    {
        // Arrange
        var detector = new OneClassSVMOutlierDetector<double>();
        var X = CreateNormalData(50, 2);

        // Act
        detector.Fit(X);
        var predictions = detector.Predict(X);

        // Assert
        foreach (var pred in predictions.ToArray())
        {
            Assert.True(Math.Abs(pred - 1.0) < Tolerance || Math.Abs(pred + 1.0) < Tolerance,
                $"Prediction {pred} is not 1 or -1");
        }
    }

    [Fact]
    public void OneClassSVM_InvalidNu_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new OneClassSVMOutlierDetector<double>(nu: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new OneClassSVMOutlierDetector<double>(nu: 1.5));
    }

    [Fact]
    public void OneClassSVM_NullInput_ThrowsException()
    {
        // Arrange
        var detector = new OneClassSVMOutlierDetector<double>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => detector.Fit(null!));
    }

    #endregion

    #region Autoencoder Tests

    [Fact]
    public void Autoencoder_Fit_SetsIsFittedToTrue()
    {
        // Arrange
        var detector = new AutoencoderOutlierDetector<double>(epochs: 10);
        var X = CreateNormalData(50, 4);

        // Act
        detector.Fit(X);

        // Assert
        Assert.True(detector.IsFitted);
    }

    [Fact]
    public void Autoencoder_Predict_BeforeFit_ThrowsException()
    {
        // Arrange
        var detector = new AutoencoderOutlierDetector<double>();
        var X = CreateNormalData(10, 2);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => detector.Predict(X));
    }

    [Fact]
    public void Autoencoder_Predict_ReturnsCorrectDimensions()
    {
        // Arrange
        var detector = new AutoencoderOutlierDetector<double>(epochs: 10);
        var X = CreateNormalData(50, 4);
        var XTest = CreateNormalData(10, 4);

        // Act
        detector.Fit(X);
        var predictions = detector.Predict(XTest);

        // Assert
        Assert.Equal(10, predictions.Length);
    }

    [Fact]
    public void Autoencoder_DetectsOutliers()
    {
        // Arrange
        var detector = new AutoencoderOutlierDetector<double>(
            epochs: 30,
            contamination: 0.1,
            randomSeed: 42);
        var X = CreateDataWithOutliers(50, 4, 5);

        // Act
        detector.Fit(X);
        var predictions = detector.Predict(X);

        // Assert
        int outlierCount = predictions.ToArray().Count(p => p < 0);
        Assert.True(outlierCount >= 2 && outlierCount <= 15,
            $"Expected 2-15 outliers, got {outlierCount}");
    }

    [Fact]
    public void Autoencoder_InvalidEpochs_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new AutoencoderOutlierDetector<double>(epochs: 0));
    }

    [Fact]
    public void Autoencoder_InvalidLearningRate_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new AutoencoderOutlierDetector<double>(learningRate: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new AutoencoderOutlierDetector<double>(learningRate: -0.1));
    }

    [Fact]
    public void Autoencoder_NullInput_ThrowsException()
    {
        // Arrange
        var detector = new AutoencoderOutlierDetector<double>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => detector.Fit(null!));
    }

    #endregion

    #region Common Tests

    [Theory]
    [InlineData(typeof(IsolationForestOutlierDetector<double>))]
    [InlineData(typeof(LocalOutlierFactorDetector<double>))]
    [InlineData(typeof(OneClassSVMOutlierDetector<double>))]
    [InlineData(typeof(AutoencoderOutlierDetector<double>))]
    public void AllDetectors_InvalidContamination_ThrowsException(Type detectorType)
    {
        // Act & Assert
        // Contamination <= 0 should throw
        var ex1 = Assert.ThrowsAny<Exception>(() =>
        {
            if (detectorType == typeof(IsolationForestOutlierDetector<double>))
                new IsolationForestOutlierDetector<double>(contamination: 0);
            else if (detectorType == typeof(LocalOutlierFactorDetector<double>))
                new LocalOutlierFactorDetector<double>(contamination: 0);
            else if (detectorType == typeof(OneClassSVMOutlierDetector<double>))
                new OneClassSVMOutlierDetector<double>(contamination: 0);
            else if (detectorType == typeof(AutoencoderOutlierDetector<double>))
                new AutoencoderOutlierDetector<double>(contamination: 0);
        });
        Assert.IsType<ArgumentOutOfRangeException>(ex1);

        // Contamination > 0.5 should throw
        var ex2 = Assert.ThrowsAny<Exception>(() =>
        {
            if (detectorType == typeof(IsolationForestOutlierDetector<double>))
                new IsolationForestOutlierDetector<double>(contamination: 0.6);
            else if (detectorType == typeof(LocalOutlierFactorDetector<double>))
                new LocalOutlierFactorDetector<double>(contamination: 0.6);
            else if (detectorType == typeof(OneClassSVMOutlierDetector<double>))
                new OneClassSVMOutlierDetector<double>(contamination: 0.6);
            else if (detectorType == typeof(AutoencoderOutlierDetector<double>))
                new AutoencoderOutlierDetector<double>(contamination: 0.6);
        });
        Assert.IsType<ArgumentOutOfRangeException>(ex2);
    }

    [Fact]
    public void AllDetectors_DecisionFunction_ReturnsScores()
    {
        // Arrange
        var X = CreateNormalData(50, 3);

        var isolationForest = new IsolationForestOutlierDetector<double>();
        var lof = new LocalOutlierFactorDetector<double>(numNeighbors: 5);
        var svm = new OneClassSVMOutlierDetector<double>();
        var autoencoder = new AutoencoderOutlierDetector<double>(epochs: 10);

        // Act
        isolationForest.Fit(X);
        lof.Fit(X);
        svm.Fit(X);
        autoencoder.Fit(X);

        var scores1 = isolationForest.DecisionFunction(X);
        var scores2 = lof.DecisionFunction(X);
        var scores3 = svm.DecisionFunction(X);
        var scores4 = autoencoder.DecisionFunction(X);

        // Assert - All should return scores with correct dimensions
        Assert.Equal(50, scores1.Length);
        Assert.Equal(50, scores2.Length);
        Assert.Equal(50, scores3.Length);
        Assert.Equal(50, scores4.Length);
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Creates a matrix with normally distributed data centered at origin.
    /// </summary>
    private static Matrix<double> CreateNormalData(int numSamples, int numFeatures, int seed = 42)
    {
        var random = new Random(seed);
        var matrix = new Matrix<double>(numSamples, numFeatures);

        for (int i = 0; i < numSamples; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                // Box-Muller transform for normal distribution
                double u1 = 1.0 - random.NextDouble();
                double u2 = 1.0 - random.NextDouble();
                matrix[i, j] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            }
        }

        return matrix;
    }

    /// <summary>
    /// Creates a matrix with normal data plus some obvious outliers far from the main cluster.
    /// </summary>
    private static Matrix<double> CreateDataWithOutliers(int numNormal, int numFeatures, int numOutliers, int seed = 42)
    {
        var random = new Random(seed);
        int total = numNormal + numOutliers;
        var matrix = new Matrix<double>(total, numFeatures);

        // Generate normal data
        for (int i = 0; i < numNormal; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                double u1 = 1.0 - random.NextDouble();
                double u2 = 1.0 - random.NextDouble();
                matrix[i, j] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            }
        }

        // Generate outliers (far from origin)
        for (int i = numNormal; i < total; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                // Outliers are 10-15 units away from center (normal data has std ~ 1)
                matrix[i, j] = 10 + random.NextDouble() * 5;
            }
        }

        return matrix;
    }

    #endregion
}
