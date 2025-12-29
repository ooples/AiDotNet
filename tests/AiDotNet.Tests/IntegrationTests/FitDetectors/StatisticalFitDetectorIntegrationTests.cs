using AiDotNet.Enums;
using AiDotNet.Exceptions;
using AiDotNet.FitDetectors;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.FitDetectors;

/// <summary>
/// Integration tests for Statistical Fit Detectors including ResidualAnalysisFitDetector,
/// AutocorrelationFitDetector, HeteroscedasticityFitDetector, VIFFitDetector, CookDistanceFitDetector,
/// InformationCriteriaFitDetector, ROCCurveFitDetector, PrecisionRecallCurveFitDetector,
/// ConfusionMatrixFitDetector, PermutationTestFitDetector, and others.
/// These tests verify statistical analysis methods work correctly and produce meaningful results.
/// </summary>
public class StatisticalFitDetectorIntegrationTests
{
    #region Test Data Helpers

    /// <summary>
    /// Creates evaluation data with good fit characteristics for statistical analysis.
    /// </summary>
    private static ModelEvaluationData<double, Matrix<double>, Vector<double>> CreateGoodFitData()
    {
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
    /// Creates evaluation data with poor fit characteristics.
    /// </summary>
    private static ModelEvaluationData<double, Matrix<double>, Vector<double>> CreatePoorFitData()
    {
        var (trainActual, trainPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(0.7);
        var (validActual, validPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(0.75);
        var (testActual, testPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(0.72);

        var features = FitDetectorTestHelper.CreateFeatureMatrix(trainActual.Length, 3);

        return FitDetectorTestHelper.CreateEvaluationData(
            trainActual, trainPredicted,
            validActual, validPredicted,
            testActual, testPredicted,
            features: features);
    }

    /// <summary>
    /// Creates evaluation data with overfit characteristics.
    /// </summary>
    private static ModelEvaluationData<double, Matrix<double>, Vector<double>> CreateOverfitData()
    {
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
    /// Creates evaluation data with single feature for detectors that require simple regression
    /// (like HeteroscedasticityFitDetector which internally uses SimpleRegression).
    /// </summary>
    private static ModelEvaluationData<double, Matrix<double>, Vector<double>> CreateSingleFeatureData()
    {
        var (trainActual, trainPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(0.01);
        var (validActual, validPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(0.012);
        var (testActual, testPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(0.011);

        // Single column feature matrix for SimpleRegression compatibility
        var features = FitDetectorTestHelper.CreateFeatureMatrix(trainActual.Length, 1);

        return FitDetectorTestHelper.CreateEvaluationData(
            trainActual, trainPredicted,
            validActual, validPredicted,
            testActual, testPredicted,
            features: features);
    }

    #endregion

    #region ResidualAnalysisFitDetector Tests

    [Fact]
    public void ResidualAnalysisFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void ResidualAnalysisFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new ResidualAnalysisFitDetectorOptions
        {
            MeanThreshold = 0.15,
            MapeThreshold = 0.12
        };

        // Act
        var detector = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void ResidualAnalysisFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var detector = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
        Assert.NotNull(result.Recommendations);
    }

    [Fact]
    public void ResidualAnalysisFitDetector_DetectFit_WithPoorFitData_DetectsIssue()
    {
        // Arrange
        var detector = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreatePoorFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.NotEqual(FitType.GoodFit, result.FitType);
    }

    [Fact]
    public void ResidualAnalysisFitDetector_DetectFit_ReturnsValidConfidenceLevel()
    {
        // Arrange
        var detector = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.True(result.ConfidenceLevel >= 0.0);
        Assert.True(result.ConfidenceLevel <= 1.0);
    }

    #endregion

    #region AutocorrelationFitDetector Tests

    [Fact]
    public void AutocorrelationFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new AutocorrelationFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void AutocorrelationFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new AutocorrelationFitDetectorOptions
        {
            StrongPositiveAutocorrelationThreshold = 0.8,
            NoAutocorrelationLowerBound = 1.6
        };

        // Act
        var detector = new AutocorrelationFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void AutocorrelationFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var detector = new AutocorrelationFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
        Assert.NotNull(result.Recommendations);
    }

    [Fact]
    public void AutocorrelationFitDetector_DetectFit_ReturnsAutocorrelationRelatedFitType()
    {
        // Arrange
        var detector = new AutocorrelationFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert - Should return an autocorrelation-related fit type or indicate no issues
        var validAutocorrelationTypes = new[]
        {
            FitType.NoAutocorrelation,
            FitType.WeakAutocorrelation,
            FitType.StrongPositiveAutocorrelation,
            FitType.StrongNegativeAutocorrelation,
            FitType.GoodFit,
            FitType.Moderate
        };

        Assert.Contains(result.FitType, validAutocorrelationTypes);
    }

    #endregion

    #region HeteroscedasticityFitDetector Tests

    [Fact]
    public void HeteroscedasticityFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new HeteroscedasticityFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void HeteroscedasticityFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new HeteroscedasticityFitDetectorOptions
        {
            HeteroscedasticityThreshold = 0.01
        };

        // Act
        var detector = new HeteroscedasticityFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void HeteroscedasticityFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var detector = new HeteroscedasticityFitDetector<double, Matrix<double>, Vector<double>>();
        // HeteroscedasticityFitDetector uses SimpleRegression internally which requires 1 feature column
        var evaluationData = CreateSingleFeatureData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
        Assert.NotNull(result.Recommendations);
    }

    [Fact]
    public void HeteroscedasticityFitDetector_DetectFit_ReturnsValidConfidenceLevel()
    {
        // Arrange
        var detector = new HeteroscedasticityFitDetector<double, Matrix<double>, Vector<double>>();
        // HeteroscedasticityFitDetector uses SimpleRegression internally which requires 1 feature column
        var evaluationData = CreateSingleFeatureData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.True(result.ConfidenceLevel >= 0.0);
        Assert.True(result.ConfidenceLevel <= 1.0);
    }

    #endregion

    #region VIFFitDetector Tests

    [Fact]
    public void VIFFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new VIFFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void VIFFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new VIFFitDetectorOptions
        {
            ModerateMulticollinearityThreshold = 3.0,
            SevereMulticollinearityThreshold = 8.0
        };

        // Act
        var detector = new VIFFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void VIFFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var detector = new VIFFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
        Assert.NotNull(result.Recommendations);
    }

    [Fact]
    public void VIFFitDetector_DetectFit_ReturnsMulticollinearityRelatedFitType()
    {
        // Arrange
        var detector = new VIFFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert - Should return multicollinearity-related fit type
        var validVifTypes = new[]
        {
            FitType.GoodFit,
            FitType.ModerateMulticollinearity,
            FitType.SevereMulticollinearity,
            FitType.Moderate
        };

        Assert.Contains(result.FitType, validVifTypes);
    }

    #endregion

    #region CookDistanceFitDetector Tests

    [Fact]
    public void CookDistanceFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new CookDistanceFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void CookDistanceFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new CookDistanceFitDetectorOptions
        {
            InfluentialThreshold = 0.8
        };

        // Act
        var detector = new CookDistanceFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void CookDistanceFitDetector_DetectFit_WithoutModel_ThrowsInvalidOperationException()
    {
        // Arrange
        var detector = new CookDistanceFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act & Assert - InvalidOperationException is thrown when Model is not set,
        // as this is a required dependency for Cook's distance calculation
        var exception = Assert.Throws<InvalidOperationException>(() => detector.DetectFit(evaluationData));
        Assert.Contains("Model is null", exception.Message);
        Assert.Contains("CookDistanceFitDetector requires Model", exception.Message);
    }

    #endregion

    #region InformationCriteriaFitDetector Tests

    [Fact]
    public void InformationCriteriaFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void InformationCriteriaFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new InformationCriteriaFitDetectorOptions
        {
            AicThreshold = 2.5,
            OverfitThreshold = 0.15
        };

        // Act
        var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void InformationCriteriaFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
        Assert.NotNull(result.Recommendations);
    }

    #endregion

    #region ROCCurveFitDetector Tests

    [Fact]
    public void ROCCurveFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new ROCCurveFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void ROCCurveFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new ROCCurveFitDetectorOptions
        {
            GoodFitThreshold = 0.85,
            ModerateFitThreshold = 0.75,
            PoorFitThreshold = 0.65
        };

        // Act
        var detector = new ROCCurveFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void ROCCurveFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var detector = new ROCCurveFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
    }

    [Fact]
    public void ROCCurveFitDetector_DetectFit_ReturnsValidConfidenceLevel()
    {
        // Arrange
        var detector = new ROCCurveFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert - Confidence level is AUC * scaling factor, clamped to [0, 1] range
        // The implementation ensures confidence never goes negative or exceeds 1.0
        Assert.NotNull(result);
        Assert.True(result.ConfidenceLevel >= 0.0,
            $"ConfidenceLevel should be >= 0, actual: {result.ConfidenceLevel}");
        Assert.True(result.ConfidenceLevel <= 1.0,
            $"ConfidenceLevel should be <= 1, actual: {result.ConfidenceLevel}");
    }

    #endregion

    #region PrecisionRecallCurveFitDetector Tests

    [Fact]
    public void PrecisionRecallCurveFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new PrecisionRecallCurveFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void PrecisionRecallCurveFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new PrecisionRecallCurveFitDetectorOptions
        {
            AreaUnderCurveThreshold = 0.8,
            F1ScoreThreshold = 0.7
        };

        // Act
        var detector = new PrecisionRecallCurveFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void PrecisionRecallCurveFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var detector = new PrecisionRecallCurveFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
        Assert.NotNull(result.Recommendations);
    }

    #endregion

    #region ConfusionMatrixFitDetector Tests

    [Fact]
    public void ConfusionMatrixFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new ConfusionMatrixFitDetectorOptions();

        // Act
        var detector = new ConfusionMatrixFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void ConfusionMatrixFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new ConfusionMatrixFitDetectorOptions
        {
            GoodFitThreshold = 0.88,
            ModerateFitThreshold = 0.68
        };

        // Act
        var detector = new ConfusionMatrixFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void ConfusionMatrixFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var options = new ConfusionMatrixFitDetectorOptions();
        var detector = new ConfusionMatrixFitDetector<double, Matrix<double>, Vector<double>>(options);
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
    }

    #endregion

    #region PermutationTestFitDetector Tests

    [Fact]
    public void PermutationTestFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new PermutationTestFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void PermutationTestFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new PermutationTestFitDetectorOptions
        {
            NumberOfPermutations = 500,
            SignificanceLevel = 0.01
        };

        // Act
        var detector = new PermutationTestFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void PermutationTestFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var detector = new PermutationTestFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
        Assert.NotNull(result.Recommendations);
    }

    #endregion

    #region ResidualBootstrapFitDetector Tests

    [Fact]
    public void ResidualBootstrapFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new ResidualBootstrapFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void ResidualBootstrapFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new ResidualBootstrapFitDetectorOptions
        {
            NumBootstrapSamples = 200,
            MinSampleSize = 25
        };

        // Act
        var detector = new ResidualBootstrapFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void ResidualBootstrapFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var detector = new ResidualBootstrapFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
    }

    #endregion

    #region ShapleyValueFitDetector Tests

    [Fact]
    public void ShapleyValueFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new ShapleyValueFitDetectorOptions();

        // Act
        var detector = new ShapleyValueFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void ShapleyValueFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new ShapleyValueFitDetectorOptions
        {
            MonteCarloSamples = 50
        };

        // Act
        var detector = new ShapleyValueFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void ShapleyValueFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var options = new ShapleyValueFitDetectorOptions();
        var detector = new ShapleyValueFitDetector<double, Matrix<double>, Vector<double>>(options);
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
        Assert.NotNull(result.Recommendations);
    }

    #endregion

    #region PartialDependencePlotFitDetector Tests

    [Fact]
    public void PartialDependencePlotFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new PartialDependencePlotFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void PartialDependencePlotFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new PartialDependencePlotFitDetectorOptions
        {
            NumPoints = 50
        };

        // Act
        var detector = new PartialDependencePlotFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void PartialDependencePlotFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var detector = new PartialDependencePlotFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
    }

    #endregion

    #region LearningCurveFitDetector Tests

    [Fact]
    public void LearningCurveFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void LearningCurveFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new LearningCurveFitDetectorOptions
        {
            ConvergenceThreshold = 0.02,
            MinDataPoints = 8
        };

        // Act
        var detector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void LearningCurveFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var detector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
        Assert.NotNull(result.Recommendations);
    }

    [Fact]
    public void LearningCurveFitDetector_DetectFit_ReturnsNonEmptyRecommendations()
    {
        // Arrange
        var detector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotEmpty(result.Recommendations);
    }

    #endregion

    #region Cross-Detector Consistency Tests

    [Fact]
    public void AllStatisticalDetectors_WithSameData_ProduceValidResults()
    {
        // Arrange
        var evaluationData = CreateGoodFitData();
        // Note: HeteroscedasticityFitDetector excluded - requires 1-column feature matrix for SimpleRegression
        // Note: CookDistanceFitDetector excluded - requires Model to be set on ModelStats
        var detectors = new IFitDetector<double, Matrix<double>, Vector<double>>[]
        {
            new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>(),
            new AutocorrelationFitDetector<double, Matrix<double>, Vector<double>>(),
            new VIFFitDetector<double, Matrix<double>, Vector<double>>(),
            new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>(),
            new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>()
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
    public void AllStatisticalDetectors_ReturnValidConfidenceLevels()
    {
        // Arrange
        var evaluationData = CreateGoodFitData();
        // Note: HeteroscedasticityFitDetector excluded - requires 1-column feature matrix for SimpleRegression
        var detectors = new IFitDetector<double, Matrix<double>, Vector<double>>[]
        {
            new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>(),
            new AutocorrelationFitDetector<double, Matrix<double>, Vector<double>>(),
            new VIFFitDetector<double, Matrix<double>, Vector<double>>(),
            new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>()
        };

        // Act & Assert
        foreach (var detector in detectors)
        {
            var result = detector.DetectFit(evaluationData);
            Assert.True(result.ConfidenceLevel >= 0.0,
                $"Detector {detector.GetType().Name} returned confidence < 0");
            Assert.True(result.ConfidenceLevel <= 1.0,
                $"Detector {detector.GetType().Name} returned confidence > 1");
        }
    }

    [Fact]
    public void AllStatisticalDetectors_WithPoorFitData_DetectProblems()
    {
        // Arrange
        var evaluationData = CreatePoorFitData();
        var detectors = new IFitDetector<double, Matrix<double>, Vector<double>>[]
        {
            new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>(),
            new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>()
        };

        // Act & Assert
        foreach (var detector in detectors)
        {
            var result = detector.DetectFit(evaluationData);

            Assert.NotNull(result);
            // Should not indicate a perfect fit for poor data
            Assert.True(
                result.FitType != FitType.GoodFit || result.ConfidenceLevel < 0.9,
                $"Detector {detector.GetType().Name} should detect issues with poor fit data");
        }
    }

    #endregion

    #region AdditionalInfo Dictionary Tests

    [Fact]
    public void ResidualAnalysisFitDetector_DetectFit_AdditionalInfoIsNotNull()
    {
        // Arrange
        var detector = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result.AdditionalInfo);
    }

    [Fact]
    public void VIFFitDetector_DetectFit_AdditionalInfoIsNotNull()
    {
        // Arrange
        var detector = new VIFFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result.AdditionalInfo);
    }

    #endregion
}
