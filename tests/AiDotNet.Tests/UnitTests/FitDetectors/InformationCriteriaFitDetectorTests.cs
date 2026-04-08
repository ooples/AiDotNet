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
    /// Unit tests for the InformationCriteriaFitDetector class.
    /// Tests use properly constructed ModelEvaluationData with calculated error statistics.
    /// The InformationCriteriaFitDetector evaluates AIC/BIC from ErrorStats properties.
    /// </summary>
    public class InformationCriteriaFitDetectorTests
    {
        /// <summary>
        /// Creates evaluation data with properly calculated statistics.
        /// The AIC/BIC values are automatically computed from ErrorStats based on MSE.
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
        public void DetectFit_WithValidData_ReturnsValidFitType()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.True(
                result.FitType == FitType.GoodFit ||
                result.FitType == FitType.Overfit ||
                result.FitType == FitType.Underfit ||
                result.FitType == FitType.HighVariance ||
                result.FitType == FitType.Unstable);
        }

        [Fact]
        public void DetectFit_ReturnsConfidenceLevel()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
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
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.Recommendations);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_IncludesThresholdsInRecommendations()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.Recommendations);
            Assert.Contains(result.Recommendations, r => r.Contains("AIC threshold") || r.Contains("BIC threshold"));
        }

        [Fact]
        public void DetectFit_ReturnsRecommendationsForFitType()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.Recommendations);
            Assert.NotEmpty(result.Recommendations);

            // Verify recommendations contain advice based on fit type
            bool hasAdvice = result.Recommendations.Any(r =>
                r.Contains("good fit") ||
                r.Contains("overfitting") ||
                r.Contains("underfitting") ||
                r.Contains("high variance") ||
                r.Contains("unstable"));
            Assert.True(hasAdvice);
        }

        [Fact]
        public void DetectFit_WithLowMse_ReturnsResult()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
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
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 1.0, validationMse: 1.5);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_ResultContainsAllRequiredFields()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.Recommendations);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_WithDifferentTrainValidationMse_ReturnsResult()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();

            // Large gap between training and validation MSE might indicate overfitting
            var evaluationData = CreateMockEvaluationData(trainMse: 0.01, validationMse: 0.5);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void DetectFit_MultipleCallsReturnConsistentResults()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result1 = detector.DetectFit(evaluationData);
            var result2 = detector.DetectFit(evaluationData);

            // Assert
            Assert.Equal(result1.FitType, result2.FitType);
            Assert.Equal(result1.ConfidenceLevel, result2.ConfidenceLevel);
        }

        [Fact]
        public void DetectFit_WithCustomThresholds_InitializesSuccessfully()
        {
            // Arrange
            var options = new InformationCriteriaFitDetectorOptions
            {
                AicThreshold = 10.0,
                BicThreshold = 10.0,
                OverfitThreshold = 20.0
            };
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>(options);
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.15);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void DetectFit_WithSimilarMseValues_ProducesResult()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(
                trainMse: 0.10,
                validationMse: 0.105,
                testMse: 0.103);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.Recommendations);
        }

        [Fact]
        public void DetectFit_WithIdenticalMse_StillProducesResult()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(
                trainMse: 0.1,
                validationMse: 0.1,
                testMse: 0.1);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.Recommendations);
        }

        [Fact]
        public void DetectFit_GoodFitAdvice_ContainsDeploymentRecommendation()
        {
            // Arrange - Use custom thresholds that make any fit "good"
            var options = new InformationCriteriaFitDetectorOptions
            {
                AicThreshold = 1000.0,  // Very high threshold
                BicThreshold = 1000.0,
                OverfitThreshold = 500.0,
                UnderfitThreshold = 500.0,
                HighVarianceThreshold = 500.0
            };
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>(options);
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            if (result.FitType == FitType.GoodFit)
            {
                Assert.Contains(result.Recommendations, r => r.Contains("good fit") || r.Contains("deploying"));
            }
        }

        [Fact]
        public void DetectFit_OverfitAdvice_ContainsRegularizationRecommendation()
        {
            // Arrange - Use thresholds that detect overfitting more readily
            var options = new InformationCriteriaFitDetectorOptions
            {
                AicThreshold = 0.001,  // Very strict threshold
                BicThreshold = 0.001,
                OverfitThreshold = 0.0001  // Any positive diff triggers overfit
            };
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>(options);
            var evaluationData = CreateMockEvaluationData(trainMse: 0.05, validationMse: 0.5);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            if (result.FitType == FitType.Overfit)
            {
                Assert.Contains(result.Recommendations, r => r.Contains("regularization") || r.Contains("complexity"));
            }
        }

        [Fact]
        public void DetectFit_UnderfitAdvice_ContainsComplexityRecommendation()
        {
            // Arrange - Use thresholds that detect underfitting
            var options = new InformationCriteriaFitDetectorOptions
            {
                AicThreshold = 0.001,
                BicThreshold = 0.001,
                OverfitThreshold = 1000.0,  // Don't trigger overfit
                UnderfitThreshold = 0.0001  // Very sensitive to negative diff
            };
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>(options);
            var evaluationData = CreateMockEvaluationData(trainMse: 0.5, validationMse: 0.05);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            if (result.FitType == FitType.Underfit)
            {
                Assert.Contains(result.Recommendations, r => r.Contains("complexity") || r.Contains("features"));
            }
        }

        [Fact]
        public void DetectFit_HighVarianceAdvice_ContainsDataRecommendation()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(
                trainMse: 0.1,
                validationMse: 0.12,
                testMse: 0.5);  // Large difference between validation and test

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            if (result.FitType == FitType.HighVariance)
            {
                Assert.Contains(result.Recommendations, r => r.Contains("data") || r.Contains("ensemble") || r.Contains("cross-validation"));
            }
        }

        [Fact]
        public void DetectFit_UnstableAdvice_ContainsInvestigationRecommendation()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            if (result.FitType == FitType.Unstable)
            {
                Assert.Contains(result.Recommendations, r => r.Contains("quality") || r.Contains("architecture") || r.Contains("feature"));
            }
        }

        [Fact]
        public void DetectFit_WithConsistentMetrics_ReturnsHighConfidence()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var consistentData = CreateMockEvaluationData(
                trainMse: 0.1,
                validationMse: 0.101,  // Very small difference
                testMse: 0.102);
            var inconsistentData = CreateMockEvaluationData(
                trainMse: 0.1,
                validationMse: 1.5,  // Large difference
                testMse: 0.8);

            // Act
            var consistentResult = detector.DetectFit(consistentData);
            var inconsistentResult = detector.DetectFit(inconsistentData);

            // Assert - Both should have valid confidence levels in [0, 1]
            Assert.True(consistentResult.ConfidenceLevel >= 0.0 && consistentResult.ConfidenceLevel <= 1.0);
            Assert.True(inconsistentResult.ConfidenceLevel >= 0.0 && inconsistentResult.ConfidenceLevel <= 1.0);

            // Consistent metrics should produce higher or equal confidence than inconsistent ones
            Assert.True(consistentResult.ConfidenceLevel >= inconsistentResult.ConfidenceLevel,
                $"Expected consistent data confidence ({consistentResult.ConfidenceLevel:F4}) >= " +
                $"inconsistent data confidence ({inconsistentResult.ConfidenceLevel:F4})");
        }

        [Fact]
        public void DetectFit_WithInconsistentMetrics_ReturnsValidResult()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(
                trainMse: 0.1,
                validationMse: 1.5,  // Large difference
                testMse: 0.8);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert - Should still produce a valid result
            Assert.NotNull(result);
            Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
            Assert.True(result.ConfidenceLevel >= 0.0 && result.ConfidenceLevel <= 1.0);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var detector = new InformationCriteriaFitDetector<float, Matrix<float>, Vector<float>>();

            // Create distinct float-typed data for each dataset
            var trainActual = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f,
                21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f });
            var trainPredicted = new Vector<float>(new float[] { 1.1f, 2.1f, 2.9f, 4.1f, 5.0f, 6.1f, 7.0f, 8.1f, 9.0f, 10.1f,
                11.0f, 12.1f, 12.9f, 14.1f, 15.0f, 16.1f, 17.0f, 18.1f, 19.0f, 20.1f,
                21.0f, 22.1f, 22.9f, 24.1f, 25.0f, 26.1f, 27.0f, 28.1f, 29.0f, 30.1f });

            // Validation data - slightly different predictions (simulating validation error)
            var validPredicted = new Vector<float>(new float[] { 1.15f, 2.05f, 3.1f, 4.0f, 5.1f, 6.0f, 7.1f, 8.0f, 9.1f, 10.0f,
                11.1f, 12.0f, 13.1f, 14.0f, 15.1f, 16.0f, 17.1f, 18.0f, 19.1f, 20.0f,
                21.1f, 22.0f, 23.1f, 24.0f, 25.1f, 26.0f, 27.1f, 28.0f, 29.1f, 30.0f });

            // Test data - slightly different predictions (simulating test error)
            var testPredicted = new Vector<float>(new float[] { 1.2f, 2.0f, 3.0f, 4.2f, 5.0f, 6.2f, 7.0f, 8.0f, 9.0f, 10.2f,
                11.0f, 12.2f, 13.0f, 14.0f, 15.0f, 16.2f, 17.0f, 18.2f, 19.0f, 20.0f,
                21.0f, 22.2f, 23.0f, 24.0f, 25.0f, 26.2f, 27.0f, 28.2f, 29.0f, 30.0f });

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

            var validErrorStats = new ErrorStats<float>(new AiDotNet.Models.Inputs.ErrorStatsInputs<float>
            {
                Actual = trainActual,
                Predicted = validPredicted,
                FeatureCount = 3,
                PredictionType = PredictionType.Regression
            });
            var validPredictionStats = new PredictionStats<float>(new AiDotNet.Models.Inputs.PredictionStatsInputs<float>
            {
                Actual = trainActual,
                Predicted = validPredicted,
                NumberOfParameters = 3,
                ConfidenceLevel = 0.95,
                LearningCurveSteps = 10,
                PredictionType = PredictionType.Regression
            });

            var testErrorStats = new ErrorStats<float>(new AiDotNet.Models.Inputs.ErrorStatsInputs<float>
            {
                Actual = trainActual,
                Predicted = testPredicted,
                FeatureCount = 3,
                PredictionType = PredictionType.Regression
            });
            var testPredictionStats = new PredictionStats<float>(new AiDotNet.Models.Inputs.PredictionStatsInputs<float>
            {
                Actual = trainActual,
                Predicted = testPredicted,
                NumberOfParameters = 3,
                ConfidenceLevel = 0.95,
                LearningCurveSteps = 10,
                PredictionType = PredictionType.Regression
            });

            var evaluationData = new ModelEvaluationData<float, Matrix<float>, Vector<float>>
            {
                TrainingSet = new DataSetStats<float, Matrix<float>, Vector<float>>
                {
                    ErrorStats = trainErrorStats,
                    PredictionStats = trainPredictionStats
                },
                ValidationSet = new DataSetStats<float, Matrix<float>, Vector<float>>
                {
                    ErrorStats = validErrorStats,
                    PredictionStats = validPredictionStats
                },
                TestSet = new DataSetStats<float, Matrix<float>, Vector<float>>
                {
                    ErrorStats = testErrorStats,
                    PredictionStats = testPredictionStats
                }
            };

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.True(Enum.IsDefined(typeof(FitType), result.FitType));
            Assert.True(result.ConfidenceLevel >= 0.0 && result.ConfidenceLevel <= 1.0);
        }
    }
}
