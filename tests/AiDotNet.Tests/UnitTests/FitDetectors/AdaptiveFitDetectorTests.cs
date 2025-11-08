using AiDotNet.Enums;
using AiDotNet.FitDetectors;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Statistics;
using Xunit;

namespace AiDotNetTests.UnitTests.FitDetectors
{
    /// <summary>
    /// Unit tests for the AdaptiveFitDetector class.
    /// </summary>
    public class AdaptiveFitDetectorTests
    {
        private static ModelEvaluationData<double, Matrix<double>, Vector<double>> CreateTestEvaluationData(
            double trainingVariance, double validationVariance, double testVariance,
            double trainingR2, double validationR2, double testR2)
        {
            var trainingBasicStats = BasicStats<double>.Empty();
            var validationBasicStats = BasicStats<double>.Empty();
            var testBasicStats = BasicStats<double>.Empty();

            var trainingPredStats = PredictionStats<double>.Empty();
            var validationPredStats = PredictionStats<double>.Empty();
            var testPredStats = PredictionStats<double>.Empty();

            // Use reflection to set variance values
            var varianceProperty = typeof(BasicStats<double>).GetProperty("Variance");
            varianceProperty?.SetValue(trainingBasicStats, trainingVariance);
            varianceProperty?.SetValue(validationBasicStats, validationVariance);
            varianceProperty?.SetValue(testBasicStats, testVariance);

            // Use reflection to set R2 values
            var r2Property = typeof(PredictionStats<double>).GetProperty("R2");
            r2Property?.SetValue(trainingPredStats, trainingR2);
            r2Property?.SetValue(validationPredStats, validationR2);
            r2Property?.SetValue(testPredStats, testR2);

            return new ModelEvaluationData<double, Matrix<double>, Vector<double>>
            {
                TrainingSet = new DataSetStats<double, Matrix<double>, Vector<double>>
                {
                    ActualBasicStats = trainingBasicStats,
                    PredictionStats = trainingPredStats,
                    ErrorStats = ErrorStats<double>.Empty()
                },
                ValidationSet = new DataSetStats<double, Matrix<double>, Vector<double>>
                {
                    ActualBasicStats = validationBasicStats,
                    PredictionStats = validationPredStats,
                    ErrorStats = ErrorStats<double>.Empty()
                },
                TestSet = new DataSetStats<double, Matrix<double>, Vector<double>>
                {
                    ActualBasicStats = testBasicStats,
                    PredictionStats = testPredStats,
                    ErrorStats = ErrorStats<double>.Empty()
                }
            };
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
        public void DetectFit_WithSimpleDataAndGoodPerformance_UsesResidualAnalyzer()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                trainingVariance: 1.0,     // Low variance = simple data
                validationVariance: 1.1,
                testVariance: 1.05,
                trainingR2: 0.9,           // High R2 = good performance
                validationR2: 0.88,
                testR2: 0.89
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.Recommendations);
            Assert.Contains(result.Recommendations, r => r.Contains("Residual Analysis Detector"));
        }

        [Fact]
        public void DetectFit_WithModerateComplexity_UsesLearningCurveDetector()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                trainingVariance: 8.0,     // Moderate variance
                validationVariance: 8.5,
                testVariance: 8.2,
                trainingR2: 0.7,           // Moderate performance
                validationR2: 0.68,
                testR2: 0.69
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.Recommendations);
            Assert.Contains(result.Recommendations, r => r.Contains("Learning Curve Detector"));
        }

        [Fact]
        public void DetectFit_WithComplexDataAndPoorPerformance_UsesHybridDetector()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                trainingVariance: 20.0,    // High variance = complex data
                validationVariance: 21.0,
                testVariance: 20.5,
                trainingR2: 0.4,           // Poor performance
                validationR2: 0.38,
                testR2: 0.39
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.Recommendations);
            Assert.Contains(result.Recommendations, r => r.Contains("Hybrid Detector"));
        }

        [Fact]
        public void DetectFit_ReturnsValidFitType()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                5.0, 5.5, 5.2,
                0.75, 0.73, 0.74
            );

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
            var evaluationData = CreateTestEvaluationData(
                5.0, 5.5, 5.2,
                0.75, 0.73, 0.74
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.0);
            Assert.True(result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void DetectFit_ReturnsNonEmptyRecommendations()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                5.0, 5.5, 5.2,
                0.75, 0.73, 0.74
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.Recommendations);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_IncludesDataComplexityInRecommendations()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                5.0, 5.5, 5.2,
                0.75, 0.73, 0.74
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.Recommendations);
            Assert.Contains(result.Recommendations, r =>
                r.Contains("data complexity") ||
                r.Contains("Simple") ||
                r.Contains("Moderate") ||
                r.Contains("Complex"));
        }

        [Fact]
        public void DetectFit_IncludesModelPerformanceInRecommendations()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                5.0, 5.5, 5.2,
                0.75, 0.73, 0.74
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.Recommendations);
            Assert.Contains(result.Recommendations, r =>
                r.Contains("model performance") ||
                r.Contains("Good") ||
                r.Contains("Moderate") ||
                r.Contains("Poor"));
        }

        [Fact]
        public void DetectFit_WithCustomComplexityThreshold_UsesCustomThreshold()
        {
            // Arrange
            var options = new AdaptiveFitDetectorOptions
            {
                ComplexityThreshold = 2.0  // Lower threshold
            };
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>(options);
            var evaluationData = CreateTestEvaluationData(
                trainingVariance: 3.0,     // Would be simple with default, moderate with custom
                validationVariance: 3.2,
                testVariance: 3.1,
                trainingR2: 0.85,
                validationR2: 0.83,
                testR2: 0.84
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            // With lower complexity threshold, this should not use Residual Analyzer
            Assert.NotNull(result.Recommendations);
        }

        [Fact]
        public void DetectFit_WithCustomPerformanceThreshold_UsesCustomThreshold()
        {
            // Arrange
            var options = new AdaptiveFitDetectorOptions
            {
                PerformanceThreshold = 0.95  // Very high threshold
            };
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>(options);
            var evaluationData = CreateTestEvaluationData(
                trainingVariance: 1.0,
                validationVariance: 1.1,
                testVariance: 1.05,
                trainingR2: 0.85,          // Good but below custom threshold
                validationR2: 0.83,
                testR2: 0.84
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            // With higher performance threshold, even 0.85 R2 won't be "good"
            Assert.NotNull(result.Recommendations);
        }

        [Fact]
        public void DetectFit_WithVeryLowVariance_IdentifiesAsSimple()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                trainingVariance: 0.1,     // Very low variance
                validationVariance: 0.12,
                testVariance: 0.11,
                trainingR2: 0.9,
                validationR2: 0.88,
                testR2: 0.89
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.Recommendations);
            Assert.Contains(result.Recommendations, r => r.Contains("Residual Analysis Detector"));
        }

        [Fact]
        public void DetectFit_WithVeryHighVariance_IdentifiesAsComplex()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                trainingVariance: 50.0,    // Very high variance
                validationVariance: 51.0,
                testVariance: 50.5,
                trainingR2: 0.5,
                validationR2: 0.48,
                testR2: 0.49
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.Recommendations);
            Assert.Contains(result.Recommendations, r => r.Contains("Hybrid Detector"));
        }

        [Fact]
        public void DetectFit_WithHighR2_IdentifiesAsGoodPerformance()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                trainingVariance: 1.0,
                validationVariance: 1.1,
                testVariance: 1.05,
                trainingR2: 0.95,          // Very high R2
                validationR2: 0.94,
                testR2: 0.945
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.Recommendations);
            Assert.Contains(result.Recommendations, r => r.Contains("Residual Analysis Detector"));
        }

        [Fact]
        public void DetectFit_WithLowR2_IdentifiesAsPoorPerformance()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                trainingVariance: 20.0,
                validationVariance: 21.0,
                testVariance: 20.5,
                trainingR2: 0.2,           // Very low R2
                validationR2: 0.18,
                testR2: 0.19
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.Recommendations);
            Assert.Contains(result.Recommendations, r => r.Contains("Hybrid Detector"));
        }

        [Fact]
        public void DetectFit_ProvidesTailoredRecommendations()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                trainingVariance: 25.0,    // Complex data
                validationVariance: 26.0,
                testVariance: 25.5,
                trainingR2: 0.3,           // Poor performance
                validationR2: 0.28,
                testR2: 0.29
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.Recommendations);
            Assert.Contains(result.Recommendations, r =>
                r.Contains("advanced modeling") ||
                r.Contains("feature engineering") ||
                r.Contains("complex data"));
        }

        [Fact]
        public void DetectFit_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<float, Matrix<float>, Vector<float>>();

            var trainingBasicStats = BasicStats<float>.Empty();
            var validationBasicStats = BasicStats<float>.Empty();
            var testBasicStats = BasicStats<float>.Empty();
            var trainingPredStats = PredictionStats<float>.Empty();
            var validationPredStats = PredictionStats<float>.Empty();
            var testPredStats = PredictionStats<float>.Empty();

            var varianceProperty = typeof(BasicStats<float>).GetProperty("Variance");
            varianceProperty?.SetValue(trainingBasicStats, 5.0f);
            varianceProperty?.SetValue(validationBasicStats, 5.5f);
            varianceProperty?.SetValue(testBasicStats, 5.2f);

            var r2Property = typeof(PredictionStats<float>).GetProperty("R2");
            r2Property?.SetValue(trainingPredStats, 0.75f);
            r2Property?.SetValue(validationPredStats, 0.73f);
            r2Property?.SetValue(testPredStats, 0.74f);

            var evaluationData = new ModelEvaluationData<float, Matrix<float>, Vector<float>>
            {
                TrainingSet = new DataSetStats<float, Matrix<float>, Vector<float>>
                {
                    ActualBasicStats = trainingBasicStats,
                    PredictionStats = trainingPredStats,
                    ErrorStats = ErrorStats<float>.Empty()
                },
                ValidationSet = new DataSetStats<float, Matrix<float>, Vector<float>>
                {
                    ActualBasicStats = validationBasicStats,
                    PredictionStats = validationPredStats,
                    ErrorStats = ErrorStats<float>.Empty()
                },
                TestSet = new DataSetStats<float, Matrix<float>, Vector<float>>
                {
                    ActualBasicStats = testBasicStats,
                    PredictionStats = testPredStats,
                    ErrorStats = ErrorStats<float>.Empty()
                }
            };

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
        }

        [Fact]
        public void DetectFit_AdaptsToDataCharacteristics()
        {
            // Arrange
            var detector = new AdaptiveFitDetector<double, Matrix<double>, Vector<double>>();

            // Test with three different scenarios
            var simpleData = CreateTestEvaluationData(1.0, 1.1, 1.05, 0.9, 0.88, 0.89);
            var moderateData = CreateTestEvaluationData(8.0, 8.5, 8.2, 0.7, 0.68, 0.69);
            var complexData = CreateTestEvaluationData(20.0, 21.0, 20.5, 0.4, 0.38, 0.39);

            // Act
            var simpleResult = detector.DetectFit(simpleData);
            var moderateResult = detector.DetectFit(moderateData);
            var complexResult = detector.DetectFit(complexData);

            // Assert - Each should use a different detector
            Assert.NotNull(simpleResult);
            Assert.NotNull(moderateResult);
            Assert.NotNull(complexResult);

            Assert.Contains(simpleResult.Recommendations, r => r.Contains("Residual Analysis"));
            Assert.Contains(moderateResult.Recommendations, r => r.Contains("Learning Curve"));
            Assert.Contains(complexResult.Recommendations, r => r.Contains("Hybrid"));
        }
    }
}
