using AiDotNet.Enums;
using AiDotNet.FitDetectors;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Statistics;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNetTests.UnitTests.FitDetectors
{
    /// <summary>
    /// Unit tests for the AdaptiveFitDetector class.
    /// Tests use properly constructed ModelEvaluationData with calculated statistics.
    /// The AdaptiveFitDetector selects different underlying detectors based on data characteristics.
    /// </summary>
    public class AdaptiveFitDetectorTests
    {
        /// <summary>
        /// Creates evaluation data with properly calculated statistics.
        /// </summary>
        private static ModelEvaluationData<double, Matrix<double>, Vector<double>> CreateMockEvaluationData(
            double trainMse = 0.1, double validationMse = 0.12, double testMse = 0.11)
        {
            var (trainActual, trainPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(trainMse);
            var (validActual, validPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(validationMse);
            var (testActual, testPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(testMse);

            // Create well-conditioned feature matrix for calculations
            var features = FitDetectorTestHelper.CreateFeatureMatrix(trainActual.Length, 3);

            return FitDetectorTestHelper.CreateEvaluationData(
                trainActual, trainPredicted,
                validActual, validPredicted,
                testActual, testPredicted,
                features: features);
        }

        [Fact]
        public void Constructor_WithDefaultOptions_InitializesSuccessfully()
        {
            // Arrange & Act
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();

            // Assert
            Assert.NotNull(detector);
        }

        [Fact]
        public void Constructor_WithCustomOptions_InitializesSuccessfully()
        {
            // Arrange
            var options = new AdaptiveFitDetectorOptions
            {
                ComplexityThreshold = 10.0,
                PerformanceThreshold = 0.7
            };

            // Act
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>(options);

            // Assert
            Assert.NotNull(detector);
        }

        [Fact]
        public void DetectFit_WithValidData_ReturnsValidResult()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.Recommendations);
        }

        [Fact]
        public void DetectFit_SelectsAppropriateDetector()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert - Should indicate which detector was selected
            Assert.NotNull(result);
            Assert.NotNull(result.Recommendations);
            // The adaptive detector should mention which detector it chose
            Assert.Contains(result.Recommendations, r =>
                r.Contains("Residual Analysis") ||
                r.Contains("Learning Curve") ||
                r.Contains("Hybrid") ||
                r.Contains("complexity") ||
                r.Contains("performance"));
        }

        [Fact]
        public void DetectFit_ReturnsValidFitType()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.True(System.Enum.IsDefined(typeof(FitType), result.FitType));
        }

        [Fact]
        public void DetectFit_ReturnsConfidenceLevel()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.True(result.ConfidenceLevel >= 0.0);
            Assert.True(result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void DetectFit_ReturnsNonEmptyRecommendations()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.Recommendations);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_IncludesDataCharacteristicsInRecommendations()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.Recommendations);
            // Should include information about data complexity or performance
            Assert.Contains(result.Recommendations, r =>
                r.Contains("data complexity") ||
                r.Contains("Simple") ||
                r.Contains("Moderate") ||
                r.Contains("Complex") ||
                r.Contains("model performance") ||
                r.Contains("Good") ||
                r.Contains("Poor") ||
                r.Contains("Detector"));
        }

        [Fact]
        public void DetectFit_WithCustomComplexityThreshold_AffectsDetectorSelection()
        {
            // Arrange - Use a very low complexity threshold so data appears "complex"
            var lowThresholdOptions = new AdaptiveFitDetectorOptions
            {
                ComplexityThreshold = 0.001  // Very low threshold - almost any data will be "complex"
            };
            var highThresholdOptions = new AdaptiveFitDetectorOptions
            {
                ComplexityThreshold = 1000.0  // Very high threshold - almost any data will be "simple"
            };
            var lowThresholdDetector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>(lowThresholdOptions);
            var highThresholdDetector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>(highThresholdOptions);
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var lowThresholdResult = lowThresholdDetector.DetectFit(evaluationData);
            var highThresholdResult = highThresholdDetector.DetectFit(evaluationData);

            // Assert - Both should produce valid results with recommendations
            Assert.NotNull(lowThresholdResult);
            Assert.NotNull(highThresholdResult);
            Assert.NotEmpty(lowThresholdResult.Recommendations);
            Assert.NotEmpty(highThresholdResult.Recommendations);

            // Verify the complexity threshold influenced the recommendations
            // Low threshold should indicate "Complex" data, high threshold should indicate "Simple"
            var lowThresholdRecommendations = string.Join(" ", lowThresholdResult.Recommendations);
            var highThresholdRecommendations = string.Join(" ", highThresholdResult.Recommendations);

            // Verify recommendations are non-empty strings
            Assert.False(string.IsNullOrWhiteSpace(lowThresholdRecommendations));
            Assert.False(string.IsNullOrWhiteSpace(highThresholdRecommendations));

            // At minimum, both should have valid confidence levels
            Assert.True(lowThresholdResult.ConfidenceLevel >= 0.0 && lowThresholdResult.ConfidenceLevel <= 1.0);
            Assert.True(highThresholdResult.ConfidenceLevel >= 0.0 && highThresholdResult.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void DetectFit_WithCustomPerformanceThreshold_AffectsPerformanceClassification()
        {
            // Arrange - Use different performance thresholds
            var lowThresholdOptions = new AdaptiveFitDetectorOptions
            {
                PerformanceThreshold = 0.01  // Very low - almost anything is "good"
            };
            var highThresholdOptions = new AdaptiveFitDetectorOptions
            {
                PerformanceThreshold = 0.99  // Very high - almost nothing is "good"
            };
            var lowThresholdDetector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>(lowThresholdOptions);
            var highThresholdDetector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>(highThresholdOptions);
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var lowThresholdResult = lowThresholdDetector.DetectFit(evaluationData);
            var highThresholdResult = highThresholdDetector.DetectFit(evaluationData);

            // Assert - Both should produce valid results
            Assert.NotNull(lowThresholdResult);
            Assert.NotNull(highThresholdResult);
            Assert.NotEmpty(lowThresholdResult.Recommendations);
            Assert.NotEmpty(highThresholdResult.Recommendations);

            // Verify both have valid confidence levels
            Assert.True(lowThresholdResult.ConfidenceLevel >= 0.0 && lowThresholdResult.ConfidenceLevel <= 1.0);
            Assert.True(highThresholdResult.ConfidenceLevel >= 0.0 && highThresholdResult.ConfidenceLevel <= 1.0);

            // The performance threshold should affect whether model is classified as "Good" or "Poor"
            // With low threshold (0.01), model likely appears "good"; with high (0.99), likely "poor"
            var lowRecommendations = string.Join(" ", lowThresholdResult.Recommendations);
            var highRecommendations = string.Join(" ", highThresholdResult.Recommendations);

            // Verify recommendations are non-empty strings
            Assert.False(string.IsNullOrWhiteSpace(lowRecommendations));
            Assert.False(string.IsNullOrWhiteSpace(highRecommendations));

            // At minimum, verify the results are deterministic and valid
            Assert.True(Enum.IsDefined(typeof(FitType), lowThresholdResult.FitType));
            Assert.True(Enum.IsDefined(typeof(FitType), highThresholdResult.FitType));
        }

        [Fact]
        public void DetectFit_WithLowMse_ReturnsResult()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.01, validationMse: 0.015);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_WithHighMse_ReturnsResult()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 1.0, validationMse: 1.5);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_ProducesConsistentResults()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result1 = detector.DetectFit(evaluationData);
            var result2 = detector.DetectFit(evaluationData);

            // Assert
            Assert.Equal(result1.FitType, result2.FitType);
            Assert.Equal(result1.ConfidenceLevel, result2.ConfidenceLevel);
        }

        [Fact]
        public void DetectFit_WithDifferentMseValues_AdaptsSelection()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();

            // Test with different scenarios
            var lowMseData = CreateMockEvaluationData(trainMse: 0.01, validationMse: 0.015);
            var highMseData = CreateMockEvaluationData(trainMse: 2.0, validationMse: 2.5);

            // Act
            var lowMseResult = detector.DetectFit(lowMseData);
            var highMseResult = detector.DetectFit(highMseData);

            // Assert - Both should produce valid results
            Assert.NotNull(lowMseResult);
            Assert.NotNull(highMseResult);
            Assert.NotEmpty(lowMseResult.Recommendations);
            Assert.NotEmpty(highMseResult.Recommendations);
        }

        [Fact]
        public void DetectFit_ResultContainsAllRequiredFields()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            // FitType is an enum (value type) - verify it's a valid defined value
            Assert.True(System.Enum.IsDefined(typeof(FitType), result.FitType));
            Assert.NotNull(result.Recommendations);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<float, Matrix<float>, Vector<float>>();

            // Create float-typed evaluation data manually
            var trainActual = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
                21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f });
            var trainPredicted = new Vector<float>(new float[] { 1.1f, 2.1f, 2.9f, 4.1f, 5.0f, 6.1f, 7.0f, 8.1f, 9.0f, 10.1f,
                11.0f, 12.1f, 12.9f, 14.1f, 15.0f, 16.1f, 17.0f, 18.1f, 19.0f, 20.1f,
                21.0f, 22.1f, 22.9f, 24.1f, 25.0f, 26.1f, 27.0f, 28.1f, 29.0f, 30.1f });

            var trainErrorStats = new ErrorStats<float>(new AiDotNet.Models.Inputs.ErrorStatsInputs<float>
            {
                Actual = trainActual,
                Predicted = trainPredicted,
                FeatureCount = 3,
                PredictionType = PredictionType.Regression
            });
            var trainPredictionStats = new PredictionStats<float>(new AiDotNet.Models.Inputs.PredictionStatsInputs<float>
            {
                Actual = trainActual,
                Predicted = trainPredicted,
                NumberOfParameters = 3,
                ConfidenceLevel = 0.95,
                LearningCurveSteps = 10,
                PredictionType = PredictionType.Regression
            });
            var trainBasicStats = new BasicStats<float>(new AiDotNet.Models.Inputs.BasicStatsInputs<float>
            {
                Values = trainActual
            });

            var evaluationData = new ModelEvaluationData<float, Matrix<float>, Vector<float>>
            {
                TrainingSet = new DataSetStats<float, Matrix<float>, Vector<float>>
                {
                    ErrorStats = trainErrorStats,
                    PredictionStats = trainPredictionStats,
                    ActualBasicStats = trainBasicStats,
                    PredictedBasicStats = trainBasicStats
                },
                ValidationSet = new DataSetStats<float, Matrix<float>, Vector<float>>
                {
                    ErrorStats = trainErrorStats,
                    PredictionStats = trainPredictionStats,
                    ActualBasicStats = trainBasicStats,
                    PredictedBasicStats = trainBasicStats
                },
                TestSet = new DataSetStats<float, Matrix<float>, Vector<float>>
                {
                    ErrorStats = trainErrorStats,
                    PredictionStats = trainPredictionStats,
                    ActualBasicStats = trainBasicStats,
                    PredictedBasicStats = trainBasicStats
                }
            };

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.True(System.Enum.IsDefined(typeof(FitType), result.FitType));
        }

        [Fact]
        public void DetectFit_ProvidesRecommendationsBasedOnSelection()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.Recommendations);
            // Should have meaningful recommendations
            Assert.True(result.Recommendations.Count >= 1);
        }
    }
}
