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
/// Integration tests for Advanced Fit Detectors including AdaptiveFitDetector,
/// BayesianFitDetector, CalibratedProbabilityFitDetector, EnsembleFitDetector,
/// FeatureImportanceFitDetector, GaussianProcessFitDetector, GradientBoostingFitDetector,
/// HybridFitDetector, and NeuralNetworkFitDetector.
/// These tests verify advanced analysis methods work correctly and produce meaningful results.
/// </summary>
public class AdvancedFitDetectorIntegrationTests
{
    #region Test Data Helpers

    /// <summary>
    /// Creates evaluation data with good fit characteristics for testing.
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

    #endregion

    #region AdaptiveFitDetector Tests

    [Fact]
    public void AdaptiveFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void AdaptiveFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new AdaptiveFitDetectorOptions();

        // Act
        var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void AdaptiveFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
        Assert.NotNull(result.Recommendations);
    }

    [Fact]
    public void AdaptiveFitDetector_DetectFit_ReturnsValidConfidenceLevel()
    {
        // Arrange
        var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.True(result.ConfidenceLevel >= 0.0);
        Assert.True(result.ConfidenceLevel <= 1.0);
    }

    #endregion

    #region BayesianFitDetector Tests

    [Fact]
    public void BayesianFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void BayesianFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new BayesianFitDetectorOptions();

        // Act
        var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void BayesianFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
        Assert.NotNull(result.Recommendations);
    }

    [Fact]
    public void BayesianFitDetector_DetectFit_ReturnsValidConfidenceLevel()
    {
        // Arrange
        var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.True(result.ConfidenceLevel >= 0.0);
        Assert.True(result.ConfidenceLevel <= 1.0);
    }

    #endregion

    #region CalibratedProbabilityFitDetector Tests

    [Fact]
    public void CalibratedProbabilityFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void CalibratedProbabilityFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new CalibratedProbabilityFitDetectorOptions();

        // Act
        var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void CalibratedProbabilityFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
        Assert.NotNull(result.Recommendations);
    }

    #endregion

    #region EnsembleFitDetector Tests

    private List<IFitDetector<double, Matrix<double>, Vector<double>>> CreateBaseDetectors()
    {
        return new List<IFitDetector<double, Matrix<double>, Vector<double>>>
        {
            new DefaultFitDetector<double, Matrix<double>, Vector<double>>(),
            new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>(),
            new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>()
        };
    }

    [Fact]
    public void EnsembleFitDetector_Constructor_WithDetectors_InitializesSuccessfully()
    {
        // Arrange
        var detectors = CreateBaseDetectors();

        // Act
        var detector = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void EnsembleFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var detectors = CreateBaseDetectors();
        var options = new EnsembleFitDetectorOptions();

        // Act
        var detector = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors, options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void EnsembleFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var detectors = CreateBaseDetectors();
        var detector = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors);
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
        Assert.NotNull(result.Recommendations);
    }

    [Fact]
    public void EnsembleFitDetector_DetectFit_ReturnsValidConfidenceLevel()
    {
        // Arrange
        var detectors = CreateBaseDetectors();
        var detector = new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(detectors);
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.True(result.ConfidenceLevel >= 0.0);
        Assert.True(result.ConfidenceLevel <= 1.0);
    }

    #endregion

    #region FeatureImportanceFitDetector Tests

    [Fact]
    public void FeatureImportanceFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new FeatureImportanceFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void FeatureImportanceFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new FeatureImportanceFitDetectorOptions();

        // Act
        var detector = new FeatureImportanceFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void FeatureImportanceFitDetector_DetectFit_WithoutModel_ThrowsException()
    {
        // Arrange
        var detector = new FeatureImportanceFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();
        // Note: FeatureImportanceFitDetector requires Model to be set on ModelStats

        // Act & Assert
        var exception = Assert.Throws<AiDotNetException>(() => detector.DetectFit(evaluationData));
        Assert.Contains("Model is null", exception.Message);
    }

    #endregion

    #region GaussianProcessFitDetector Tests

    [Fact]
    public void GaussianProcessFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void GaussianProcessFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new GaussianProcessFitDetectorOptions();

        // Act
        var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void GaussianProcessFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
        Assert.NotNull(result.Recommendations);
    }

    [Fact]
    public void GaussianProcessFitDetector_DetectFit_ReturnsValidConfidenceLevel()
    {
        // Arrange
        var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.True(result.ConfidenceLevel >= 0.0);
        Assert.True(result.ConfidenceLevel <= 1.0);
    }

    #endregion

    #region GradientBoostingFitDetector Tests

    [Fact]
    public void GradientBoostingFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void GradientBoostingFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new GradientBoostingFitDetectorOptions();

        // Act
        var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void GradientBoostingFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
        Assert.NotNull(result.Recommendations);
    }

    [Fact]
    public void GradientBoostingFitDetector_DetectFit_ReturnsValidConfidenceLevel()
    {
        // Arrange
        var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.True(result.ConfidenceLevel >= 0.0);
        Assert.True(result.ConfidenceLevel <= 1.0);
    }

    #endregion

    #region HybridFitDetector Tests

    [Fact]
    public void HybridFitDetector_Constructor_WithDetectors_InitializesSuccessfully()
    {
        // Arrange
        var residualDetector = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
        var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();

        // Act
        var detector = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
            residualDetector, learningCurveDetector);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void HybridFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var residualDetector = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
        var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
        var options = new HybridFitDetectorOptions();

        // Act
        var detector = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
            residualDetector, learningCurveDetector, options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void HybridFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var residualDetector = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
        var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
        var detector = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
            residualDetector, learningCurveDetector);
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
        Assert.NotNull(result.Recommendations);
    }

    [Fact]
    public void HybridFitDetector_DetectFit_ReturnsValidConfidenceLevel()
    {
        // Arrange
        var residualDetector = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
        var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
        var detector = new HybridFitDetector<double, Matrix<double>, Vector<double>>(
            residualDetector, learningCurveDetector);
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.True(result.ConfidenceLevel >= 0.0);
        Assert.True(result.ConfidenceLevel <= 1.0);
    }

    #endregion

    #region NeuralNetworkFitDetector Tests

    [Fact]
    public void NeuralNetworkFitDetector_Constructor_WithDefaultOptions_InitializesSuccessfully()
    {
        // Act
        var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void NeuralNetworkFitDetector_Constructor_WithCustomOptions_InitializesSuccessfully()
    {
        // Arrange
        var options = new NeuralNetworkFitDetectorOptions();

        // Act
        var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>(options);

        // Assert
        Assert.NotNull(detector);
    }

    [Fact]
    public void NeuralNetworkFitDetector_DetectFit_WithValidData_ReturnsValidResult()
    {
        // Arrange
        var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.NotNull(result);
        Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
        Assert.NotNull(result.Recommendations);
    }

    [Fact]
    public void NeuralNetworkFitDetector_DetectFit_ReturnsValidConfidenceLevel()
    {
        // Arrange
        var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>();
        var evaluationData = CreateGoodFitData();

        // Act
        var result = detector.DetectFit(evaluationData);

        // Assert
        Assert.True(result.ConfidenceLevel >= 0.0);
        Assert.True(result.ConfidenceLevel <= 1.0);
    }

    #endregion

    #region Cross-Detector Consistency Tests

    [Fact]
    public void AllAdvancedDetectors_WithSameData_ProduceValidResults()
    {
        // Arrange
        var evaluationData = CreateGoodFitData();
        var baseDetectors = CreateBaseDetectors();
        var residualDetector = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
        var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
        // Note: FeatureImportanceFitDetector excluded - requires Model to be set on ModelStats
        var detectors = new IFitDetector<double, Matrix<double>, Vector<double>>[]
        {
            new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>(),
            new BayesianFitDetector<double, Matrix<double>, Vector<double>>(),
            new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>(),
            new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(baseDetectors),
            new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>(),
            new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>(),
            new HybridFitDetector<double, Matrix<double>, Vector<double>>(residualDetector, learningCurveDetector),
            new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>()
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
    public void AllAdvancedDetectors_ReturnValidConfidenceLevels()
    {
        // Arrange
        var evaluationData = CreateGoodFitData();
        var baseDetectors = CreateBaseDetectors();
        var residualDetector = new ResidualAnalysisFitDetector<double, Matrix<double>, Vector<double>>();
        var learningCurveDetector = new LearningCurveFitDetector<double, Matrix<double>, Vector<double>>();
        // Note: FeatureImportanceFitDetector excluded - requires Model to be set on ModelStats
        var detectors = new IFitDetector<double, Matrix<double>, Vector<double>>[]
        {
            new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>(),
            new BayesianFitDetector<double, Matrix<double>, Vector<double>>(),
            new EnsembleFitDetector<double, Matrix<double>, Vector<double>>(baseDetectors),
            new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>(),
            new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>(),
            new HybridFitDetector<double, Matrix<double>, Vector<double>>(residualDetector, learningCurveDetector),
            new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>()
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

    #endregion
}
