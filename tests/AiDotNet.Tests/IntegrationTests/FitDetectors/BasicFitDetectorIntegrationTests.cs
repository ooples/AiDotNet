using AiDotNet.Enums;
using AiDotNet.FitDetectors;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.FitDetectors;

/// <summary>
/// Integration tests for Basic Fit Detectors including DefaultFitDetector,
/// HoldoutValidationFitDetector, KFoldCrossValidationFitDetector, CrossValidationFitDetector,
/// BootstrapFitDetector, JackknifeFitDetector, and StratifiedKFoldCrossValidationFitDetector.
/// These tests verify the complete workflow from input data to fit detection results.
/// </summary>
public class BasicFitDetectorIntegrationTests
{
    #region Test Data Helpers

    /// <summary>
    /// Creates evaluation data simulating a good fit scenario where training, validation,
    /// and test performance are all high and similar.
    /// </summary>
    private static ModelEvaluationData<double, Matrix<double>, Vector<double>> CreateGoodFitData()
    {
        // Low MSE across all sets indicates good predictions
        var (trainActual, trainPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(0.01);
        var (validActual, validPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(0.012);
        var (testActual, testPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(0.011);

        var features = FitDetectorTestHelper.CreateFeatureMatrix(trainActual.Length, 3);

        return FitDetectorTestHelper.CreateEvaluationData(
            trainActual, trainPredicted,
            validActual, validPredicted,
            testActual, testPredicted,
            features: features);
    }

    /// <summary>
    /// Creates evaluation data simulating an overfit scenario where training performance
    /// is excellent but validation/test performance is poor.
    /// </summary>
    private static ModelEvaluationData<double, Matrix<double>, Vector<double>> CreateOverfitData()
    {
        // Very low training MSE but high validation/test MSE
        var (trainActual, trainPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(0.001);
        var (validActual, validPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(0.5);
        var (testActual, testPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(0.6);

        var features = FitDetectorTestHelper.CreateFeatureMatrix(trainActual.Length, 3);

        return FitDetectorTestHelper.CreateEvaluationData(
            trainActual, trainPredicted,
            validActual, validPredicted,
            testActual, testPredicted,
            features: features);
    }

    /// <summary>
    /// Creates evaluation data simulating an underfit scenario where all performance metrics are poor.
    /// </summary>
    private static ModelEvaluationData<double, Matrix<double>, Vector<double>> CreateUnderfitData()
    {
        // High MSE across all sets indicates poor model performance
        var (trainActual, trainPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(0.8);
        var (validActual, validPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(0.85);
        var (testActual, testPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(0.82);

        var features = FitDetectorTestHelper.CreateFeatureMatrix(trainActual.Length, 3);

        return FitDetectorTestHelper.CreateEvaluationData(
            trainActual, trainPredicted,
            validActual, validPredicted,
            testActual, testPredicted,
            features: features);
    }

    /// <summary>
    /// Creates evaluation data simulating high variance scenario where validation
    /// and test performance differ significantly.
    /// </summary>
    private static ModelEvaluationData<double, Matrix<double>, Vector<double>> CreateHighVarianceData()
    {
        var (trainActual, trainPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(0.1);
        var (validActual, validPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(0.15);
        var (testActual, testPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(0.45);

        var features = FitDetectorTestHelper.CreateFeatureMatrix(trainActual.Length, 3);

        return FitDetectorTestHelper.CreateEvaluationData(
            trainActual, trainPredicted,
            validActual, validPredicted,
            testActual, testPredicted,
            features: features);
    }

    #endregion

    #region DefaultFitDetector Tests

    [Fact]
    public void DefaultFitDetector_Constructor_InitializesSuccessfully()
    {
        // Act
        var detector = new DefaultFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void DefaultFitDetector_DetectFit_WithGoodFitData_ReturnsGoodFitOrValidResult()
    {
        // Arrange
        var detector = new DefaultFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
        Assert.NotNull(result.Recommendations);
    }

    [Fact]
    public void DefaultFitDetector_DetectFit_WithOverfitData_DetectsOverfitOrHighVariance()
    {
        // Arrange
        var detector = new DefaultFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateOverfitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(
            result.FitType == FitType.Overfit ||
            result.FitType == FitType.HighVariance ||
            result.FitType == FitType.Unstable,
            $"Expected Overfit, HighVariance, or Unstable but got {result.FitType}");
    }

    [Fact]
    public void DefaultFitDetector_DetectFit_WithUnderfitData_DetectsUnderfitOrHighBias()
    {
        // Arrange
        var detector = new DefaultFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateUnderfitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(
            result.FitType == FitType.Underfit ||
            result.FitType == FitType.HighBias ||
            result.FitType == FitType.PoorFit ||
            result.FitType == FitType.VeryPoorFit,
            $"Expected Underfit, HighBias, PoorFit, or VeryPoorFit but got {result.FitType}");
    }

    [Fact]
    public void DefaultFitDetector_DetectFit_ReturnsConfidenceLevelBetweenZeroAndOne()
    {
        // Arrange
        var detector = new DefaultFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.True(result.ConfidenceLevel >= 0.0, "Confidence level should be >= 0");
        Assert.True(result.ConfidenceLevel <= 1.0, "Confidence level should be <= 1");
    }

    [Fact]
    public void DefaultFitDetector_DetectFit_ReturnsRecommendationsList()
    {
        // Arrange
        var detector = new DefaultFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert - Recommendations list should be initialized (may be empty for good fit)
        Assert.NotNull(result.Recommendations);
    }

    [Fact]
    public void DefaultFitDetector_DetectFit_IsConsistentAcrossMultipleCalls()
    {
        // Arrange
        var detector = new DefaultFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result1 = detector.DetectFit(evaluationData);
        var result2 = detector.DetectFit(evaluationData);

        // Assert - Same data should produce same results
        Assert.Equal(result1.FitType, result2.FitType);
    }

    #endregion

    #region HoldoutValidationFitDetector Tests

    [Fact]
    public void HoldoutValidationFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new HoldoutValidationFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void HoldoutValidationFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new HoldoutValidationFitDetectorOptions
        {
            OverfitThreshold = 0.15,
            UnderfitThreshold = 0.4
        };

        // Act
        var detector = new HoldoutValidationFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void HoldoutValidationFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var detector = new HoldoutValidationFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
        Assert.NotNull(result.Recommendations);
    }

    [Fact]
    public void HoldoutValidationFitDetector_DetectFit_WithOverfitData_DetectsIssue()
    {
        // Arrange
        var detector = new HoldoutValidationFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateOverfitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(
            result.FitType == FitType.Overfit ||
            result.FitType == FitType.HighVariance ||
            result.FitType == FitType.Unstable,
            $"Expected problematic fit type for overfit data, got {result.FitType}");
    }

    [Fact]
    public void HoldoutValidationFitDetector_DetectFit_ReturnsValidConfidenceLevel()
    {
        // Arrange
        var detector = new HoldoutValidationFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.True(result.ConfidenceLevel >= 0.0);
        Assert.True(result.ConfidenceLevel <= 1.0);
    }

    #endregion

    #region KFoldCrossValidationFitDetector Tests

    [Fact]
    public void KFoldCrossValidationFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new KFoldCrossValidationFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void KFoldCrossValidationFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new KFoldCrossValidationFitDetectorOptions
        {
            OverfitThreshold = 0.15,
            UnderfitThreshold = 0.6,
            HighVarianceThreshold = 0.12,
            GoodFitThreshold = 0.75,
            StabilityThreshold = 0.08
        };

        // Act
        var detector = new KFoldCrossValidationFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void KFoldCrossValidationFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var detector = new KFoldCrossValidationFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
        Assert.NotNull(result.Recommendations);
        Assert.NotEmpty(result.Recommendations);
    }

    [Fact]
    public void KFoldCrossValidationFitDetector_DetectFit_IncludesMetricsInRecommendations()
    {
        // Arrange
        var detector = new KFoldCrossValidationFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.Contains(result.Recommendations, r => r.Contains("R2") || r.Contains("Validation"));
    }

    [Fact]
    public void KFoldCrossValidationFitDetector_DetectFit_WithHighVarianceData_DetectsVariance()
    {
        // Arrange
        var detector = new KFoldCrossValidationFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateHighVarianceData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        // Should detect some form of instability or variance
        Assert.True(
            result.FitType == FitType.HighVariance ||
            result.FitType == FitType.Unstable ||
            result.FitType == FitType.Overfit,
            $"Expected variance-related fit type, got {result.FitType}");
    }

    #endregion

    #region CrossValidationFitDetector Tests

    [Fact]
    public void CrossValidationFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new CrossValidationFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void CrossValidationFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new CrossValidationFitDetectorOptions
        {
            OverfitThreshold = 0.12,
            UnderfitThreshold = 0.55
        };

        // Act
        var detector = new CrossValidationFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void CrossValidationFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var detector = new CrossValidationFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
    }

    [Fact]
    public void CrossValidationFitDetector_DetectFit_ReturnsValidConfidenceAndRecommendations()
    {
        // Arrange
        var detector = new CrossValidationFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.True(result.ConfidenceLevel >= 0.0);
        Assert.True(result.ConfidenceLevel <= 1.0);
        Assert.NotNull(result.Recommendations);
    }

    #endregion

    #region BootstrapFitDetector Tests

    [Fact]
    public void BootstrapFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new BootstrapFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void BootstrapFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new BootstrapFitDetectorOptions
        {
            NumberOfBootstraps = 50,
            ConfidenceInterval = 0.90
        };

        // Act
        var detector = new BootstrapFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void BootstrapFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var detector = new BootstrapFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
        Assert.NotNull(result.Recommendations);
    }

    [Fact]
    public void BootstrapFitDetector_DetectFit_WithUnderfitData_DetectsIssue()
    {
        // Arrange
        var detector = new BootstrapFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateUnderfitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(
            result.FitType == FitType.Underfit ||
            result.FitType == FitType.HighBias ||
            result.FitType == FitType.PoorFit ||
            result.FitType == FitType.VeryPoorFit ||
            result.FitType == FitType.Unstable,
            $"Expected problematic fit type for underfit data, got {result.FitType}");
    }

    #endregion

    #region JackknifeFitDetector Tests

    [Fact]
    public void JackknifeFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new JackknifeFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void JackknifeFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new JackknifeFitDetectorOptions
        {
            MinSampleSize = 20,
            OverfitThreshold = 0.15
        };

        // Act
        var detector = new JackknifeFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void JackknifeFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var detector = new JackknifeFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
        Assert.NotNull(result.Recommendations);
    }

    [Fact]
    public void JackknifeFitDetector_DetectFit_ReturnsValidConfidenceLevel()
    {
        // Arrange
        var detector = new JackknifeFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.True(result.ConfidenceLevel >= 0.0);
        Assert.True(result.ConfidenceLevel <= 1.0);
    }

    #endregion

    #region StratifiedKFoldCrossValidationFitDetector Tests

    [Fact]
    public void StratifiedKFoldCrossValidationFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new StratifiedKFoldCrossValidationFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void StratifiedKFoldCrossValidationFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new StratifiedKFoldCrossValidationFitDetectorOptions
        {
            OverfitThreshold = 0.12,
            UnderfitThreshold = 0.55
        };

        // Act
        var detector = new StratifiedKFoldCrossValidationFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void StratifiedKFoldCrossValidationFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var detector = new StratifiedKFoldCrossValidationFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
        Assert.NotNull(result.Recommendations);
    }

    [Fact]
    public void StratifiedKFoldCrossValidationFitDetector_DetectFit_ReturnsNonEmptyRecommendations()
    {
        // Arrange
        var detector = new StratifiedKFoldCrossValidationFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotEmpty(result.Recommendations);
    }

    #endregion

    #region TimeSeriesCrossValidationFitDetector Tests

    [Fact]
    public void TimeSeriesCrossValidationFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new TimeSeriesCrossValidationFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void TimeSeriesCrossValidationFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new TimeSeriesCrossValidationFitDetectorOptions
        {
            OverfitThreshold = 0.15
        };

        // Act
        var detector = new TimeSeriesCrossValidationFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void TimeSeriesCrossValidationFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var detector = new TimeSeriesCrossValidationFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
    }

    #endregion

    #region Cross-Detector Consistency Tests

    [Fact]
    public void AllBasicDetectors_WithSameData_ProduceValidResults()
    {
        // Arrange
        var evaluationData = CreateGoodFitData();
        var detectors = new IFitDetector<double, Matrix<double>, Vector<double>>[]
        {
            new DefaultFitDetector<double, Matrix<double>, Vector<double>>(),
            new HoldoutValidationFitDetector<double, Matrix<double>, Vector<double>>(),
            new KFoldCrossValidationFitDetector<double, Matrix<double>, Vector<double>>(),
            new CrossValidationFitDetector<double, Matrix<double>, Vector<double>>(),
            new BootstrapFitDetector<double, Matrix<double>, Vector<double>>(),
            new JackknifeFitDetector<double, Matrix<double>, Vector<double>>()
        };

        // Act & Assert
        foreach (var detector in detectors)
        {
            var result = detector.DetectFit(evaluationData);

            Assert.NotNull(result);
            Assert.True(Enum.IsDefined(typeof(FitType), result.FitType),
                $"Detector {detector.GetType().Name} returned invalid FitType");
            Assert.NotNull(result.Recommendations);
        }
    }

    [Fact]
    public void AllBasicDetectors_WithOverfitData_DetectProblems()
    {
        // Arrange
        var evaluationData = CreateOverfitData();
        var detectors = new IFitDetector<double, Matrix<double>, Vector<double>>[]
        {
            new DefaultFitDetector<double, Matrix<double>, Vector<double>>(),
            new HoldoutValidationFitDetector<double, Matrix<double>, Vector<double>>(),
            new KFoldCrossValidationFitDetector<double, Matrix<double>, Vector<double>>(),
            new CrossValidationFitDetector<double, Matrix<double>, Vector<double>>()
        };

        var problematicFitTypes = new[]
        {
            FitType.Overfit, FitType.HighVariance, FitType.Unstable,
            FitType.PoorFit, FitType.VeryPoorFit
        };

        // Act & Assert
        foreach (var detector in detectors)
        {
            var result = detector.DetectFit(evaluationData);

            Assert.NotNull(result);
            Assert.True(
                Array.Exists(problematicFitTypes, t => t == result.FitType),
                $"Detector {detector.GetType().Name} should detect a problem with overfit data, but returned {result.FitType}");
        }
    }

    #endregion

    #region AdditionalInfo Dictionary Tests

    [Fact]
    public void DefaultFitDetector_DetectFit_AdditionalInfoIsNotNull()
    {
        // Arrange
        var detector = new DefaultFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result.AdditionalInfo);
    }

    [Fact]
    public void KFoldCrossValidationFitDetector_DetectFit_AdditionalInfoIsNotNull()
    {
        // Arrange
        var detector = new KFoldCrossValidationFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result.AdditionalInfo);
    }

    #endregion
}
