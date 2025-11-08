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
    /// Unit tests for the InformationCriteriaFitDetector class.
    /// </summary>
    public class InformationCriteriaFitDetectorTests
    {
        private static ModelEvaluationData<double, Matrix<double>, Vector<double>> CreateTestEvaluationData(
            double trainingAic, double validationAic, double testAic,
            double trainingBic, double validationBic, double testBic)
        {
            var trainingErrorStats = ErrorStats<double>.Empty();
            var validationErrorStats = ErrorStats<double>.Empty();
            var testErrorStats = ErrorStats<double>.Empty();

            // Use reflection to set AIC and BIC values
            var aicProperty = typeof(ErrorStats<double>).GetProperty("AIC");
            var bicProperty = typeof(ErrorStats<double>).GetProperty("BIC");

            aicProperty?.SetValue(trainingErrorStats, trainingAic);
            bicProperty?.SetValue(trainingErrorStats, trainingBic);
            aicProperty?.SetValue(validationErrorStats, validationAic);
            bicProperty?.SetValue(validationErrorStats, validationBic);
            aicProperty?.SetValue(testErrorStats, testAic);
            bicProperty?.SetValue(testErrorStats, testBic);

            return new ModelEvaluationData<double, Matrix<double>, Vector<double>>
            {
                TrainingSet = new DataSetStats<double, Matrix<double>, Vector<double>>
                {
                    ErrorStats = trainingErrorStats,
                    PredictionStats = PredictionStats<double>.Empty()
                },
                ValidationSet = new DataSetStats<double, Matrix<double>, Vector<double>>
                {
                    ErrorStats = validationErrorStats,
                    PredictionStats = PredictionStats<double>.Empty()
                },
                TestSet = new DataSetStats<double, Matrix<double>, Vector<double>>
                {
                    ErrorStats = testErrorStats,
                    PredictionStats = PredictionStats<double>.Empty()
                }
            };
        }

        [Fact]
        public void Constructor_WithDefaultOptions_InitializesSuccessfully()
        {
            // Arrange & Act
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();

            // Assert
            Assert.NotNull(detector);
        }

        [Fact]
        public void Constructor_WithCustomOptions_InitializesSuccessfully()
        {
            // Arrange
            var options = new InformationCriteriaFitDetectorOptions
            {
                AicThreshold = 3.0,
                BicThreshold = 3.0,
                OverfitThreshold = 8.0
            };

            // Act
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>(options);

            // Assert
            Assert.NotNull(detector);
        }

        [Fact]
        public void DetectFit_WithGoodFitMetrics_ReturnsGoodFit()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                trainingAic: 100.0,
                validationAic: 102.0,  // Small difference
                testAic: 101.0,
                trainingBic: 105.0,
                validationBic: 107.0,  // Small difference
                testBic: 106.0
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(FitType.GoodFit, result.FitType);
            Assert.NotNull(result.Recommendations);
            Assert.Contains(result.Recommendations, r => r.Contains("good fit"));
        }

        [Fact]
        public void DetectFit_WithOverfitMetrics_ReturnsOverfit()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                trainingAic: 50.0,
                validationAic: 100.0,  // Large difference indicating overfit
                testAic: 95.0,
                trainingBic: 55.0,
                validationBic: 105.0,  // Large difference indicating overfit
                testBic: 100.0
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(FitType.Overfit, result.FitType);
            Assert.NotNull(result.Recommendations);
            Assert.Contains(result.Recommendations, r => r.Contains("overfitting"));
        }

        [Fact]
        public void DetectFit_WithUnderfitMetrics_ReturnsUnderfit()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                trainingAic: 100.0,
                validationAic: 80.0,   // Validation better than training (unusual)
                testAic: 85.0,
                trainingBic: 105.0,
                validationBic: 85.0,   // Validation better than training (unusual)
                testBic: 90.0
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(FitType.Underfit, result.FitType);
            Assert.NotNull(result.Recommendations);
            Assert.Contains(result.Recommendations, r => r.Contains("underfitting"));
        }

        [Fact]
        public void DetectFit_WithHighVarianceMetrics_ReturnsHighVariance()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                trainingAic: 100.0,
                validationAic: 110.0,  // Moderate difference
                testAic: 150.0,        // Large difference between validation and test
                trainingBic: 105.0,
                validationBic: 115.0,
                testBic: 155.0
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(FitType.HighVariance, result.FitType);
            Assert.NotNull(result.Recommendations);
            Assert.Contains(result.Recommendations, r => r.Contains("high variance") || r.Contains("variance"));
        }

        [Fact]
        public void DetectFit_WithUnstableMetrics_ReturnsUnstable()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                trainingAic: 100.0,
                validationAic: 115.0,  // Some difference
                testAic: 108.0,        // Inconsistent patterns
                trainingBic: 105.0,
                validationBic: 112.0,
                testBic: 110.0
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(FitType.Unstable, result.FitType);
            Assert.NotNull(result.Recommendations);
            Assert.Contains(result.Recommendations, r => r.Contains("unstable"));
        }

        [Fact]
        public void DetectFit_ReturnsConfidenceLevel()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                100.0, 102.0, 101.0,
                105.0, 107.0, 106.0
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.0);
            Assert.True(result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void DetectFit_IncludesThresholdsInRecommendations()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                100.0, 102.0, 101.0,
                105.0, 107.0, 106.0
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.Recommendations);
            Assert.Contains(result.Recommendations, r => r.Contains("AIC threshold") || r.Contains("BIC threshold"));
        }

        [Fact]
        public void DetectFit_WithCustomThresholds_UsesCustomThresholds()
        {
            // Arrange
            var options = new InformationCriteriaFitDetectorOptions
            {
                AicThreshold = 10.0,   // Higher threshold
                BicThreshold = 10.0,
                OverfitThreshold = 20.0
            };
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>(options);
            var evaluationData = CreateTestEvaluationData(
                trainingAic: 100.0,
                validationAic: 108.0,  // Difference of 8, within custom threshold
                testAic: 107.0,
                trainingBic: 105.0,
                validationBic: 113.0,
                testBic: 112.0
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(FitType.GoodFit, result.FitType);
        }

        [Fact]
        public void DetectFit_OverfitRecommendations_ContainsRegularizationAdvice()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                trainingAic: 50.0,
                validationAic: 100.0,
                testAic: 95.0,
                trainingBic: 55.0,
                validationBic: 105.0,
                testBic: 100.0
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.Equal(FitType.Overfit, result.FitType);
            Assert.Contains(result.Recommendations, r => r.Contains("regularization") || r.Contains("complexity"));
        }

        [Fact]
        public void DetectFit_UnderfitRecommendations_ContainsComplexityAdvice()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                trainingAic: 100.0,
                validationAic: 80.0,
                testAic: 85.0,
                trainingBic: 105.0,
                validationBic: 85.0,
                testBic: 90.0
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.Equal(FitType.Underfit, result.FitType);
            Assert.Contains(result.Recommendations, r => r.Contains("complexity") || r.Contains("features"));
        }

        [Fact]
        public void DetectFit_HighVarianceRecommendations_ContainsDataAdvice()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                trainingAic: 100.0,
                validationAic: 110.0,
                testAic: 150.0,
                trainingBic: 105.0,
                validationBic: 115.0,
                testBic: 155.0
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.Equal(FitType.HighVariance, result.FitType);
            Assert.Contains(result.Recommendations, r => r.Contains("data") || r.Contains("ensemble") || r.Contains("cross-validation"));
        }

        [Fact]
        public void DetectFit_UnstableRecommendations_ContainsInvestigationAdvice()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                trainingAic: 100.0,
                validationAic: 115.0,
                testAic: 108.0,
                trainingBic: 105.0,
                validationBic: 112.0,
                testBic: 110.0
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.Equal(FitType.Unstable, result.FitType);
            Assert.Contains(result.Recommendations, r => r.Contains("quality") || r.Contains("architecture") || r.Contains("feature"));
        }

        [Fact]
        public void DetectFit_WithConsistentMetrics_ReturnsHighConfidence()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                trainingAic: 100.0,
                validationAic: 100.5,  // Very small difference
                testAic: 100.3,
                trainingBic: 105.0,
                validationBic: 105.4,
                testBic: 105.2
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.7);  // High confidence due to consistency
        }

        [Fact]
        public void DetectFit_WithInconsistentMetrics_ReturnsLowConfidence()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                trainingAic: 100.0,
                validationAic: 150.0,  // Large difference
                testAic: 120.0,
                trainingBic: 105.0,
                validationBic: 160.0,
                testBic: 130.0
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel <= 0.5);  // Low confidence due to large variations
        }

        [Fact]
        public void DetectFit_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<float, Matrix<float>, Vector<float>>();

            var trainingErrorStats = ErrorStats<float>.Empty();
            var validationErrorStats = ErrorStats<float>.Empty();
            var testErrorStats = ErrorStats<float>.Empty();

            var aicProperty = typeof(ErrorStats<float>).GetProperty("AIC");
            var bicProperty = typeof(ErrorStats<float>).GetProperty("BIC");

            aicProperty?.SetValue(trainingErrorStats, 100.0f);
            bicProperty?.SetValue(trainingErrorStats, 105.0f);
            aicProperty?.SetValue(validationErrorStats, 102.0f);
            bicProperty?.SetValue(validationErrorStats, 107.0f);
            aicProperty?.SetValue(testErrorStats, 101.0f);
            bicProperty?.SetValue(testErrorStats, 106.0f);

            var evaluationData = new ModelEvaluationData<float, Matrix<float>, Vector<float>>
            {
                TrainingSet = new DataSetStats<float, Matrix<float>, Vector<float>>
                {
                    ErrorStats = trainingErrorStats,
                    PredictionStats = PredictionStats<float>.Empty()
                },
                ValidationSet = new DataSetStats<float, Matrix<float>, Vector<float>>
                {
                    ErrorStats = validationErrorStats,
                    PredictionStats = PredictionStats<float>.Empty()
                },
                TestSet = new DataSetStats<float, Matrix<float>, Vector<float>>
                {
                    ErrorStats = testErrorStats,
                    PredictionStats = PredictionStats<float>.Empty()
                }
            };

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(FitType.GoodFit, result.FitType);
        }

        [Fact]
        public void DetectFit_ReturnsNonEmptyRecommendations()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                100.0, 102.0, 101.0,
                105.0, 107.0, 106.0
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.Recommendations);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_WithIdenticalAICandBIC_StillProducesResult()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                trainingAic: 100.0,
                validationAic: 100.0,
                testAic: 100.0,
                trainingBic: 100.0,
                validationBic: 100.0,
                testBic: 100.0
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
            Assert.NotNull(result.Recommendations);
        }
    }
}
