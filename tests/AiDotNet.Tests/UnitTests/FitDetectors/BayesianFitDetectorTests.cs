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
    /// Unit tests for the BayesianFitDetector class.
    /// </summary>
    public class BayesianFitDetectorTests
    {
        private static ModelEvaluationData<double, Matrix<double>, Vector<double>> CreateTestEvaluationData(
            double dic, double waic, double loo, double posteriorCheck, double bayesFactor)
        {
            var modelStats = ModelStats<double, Matrix<double>, Vector<double>>.Empty();

            // Use reflection to set the internal calculated values for testing
            var dicField = typeof(ModelStats<double, Matrix<double>, Vector<double>>).GetField("_dic",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var waicField = typeof(ModelStats<double, Matrix<double>, Vector<double>>).GetField("_waic",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var looField = typeof(ModelStats<double, Matrix<double>, Vector<double>>).GetField("_loo",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var posteriorField = typeof(ModelStats<double, Matrix<double>, Vector<double>>).GetField("_posteriorPredictiveCheck",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var bayesFactorField = typeof(ModelStats<double, Matrix<double>, Vector<double>>).GetField("_bayesFactor",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            if (dicField == null || waicField == null || looField == null || posteriorField == null || bayesFactorField == null)
            {
                throw new InvalidOperationException("One or more private fields not found on ModelStats. This test needs to be updated to match the current implementation.");
            }

            dicField.SetValue(modelStats, dic);
            waicField.SetValue(modelStats, waic);
            looField.SetValue(modelStats, loo);
            posteriorField.SetValue(modelStats, posteriorCheck);
            bayesFactorField.SetValue(modelStats, bayesFactor);

            return new ModelEvaluationData<double, Matrix<double>, Vector<double>>
            {
                ModelStats = modelStats,
                TrainingSet = new DataSetStats<double, Matrix<double>, Vector<double>>
                {
                    ErrorStats = ErrorStats<double>.Empty(),
                    PredictionStats = PredictionStats<double>.Empty()
                },
                ValidationSet = new DataSetStats<double, Matrix<double>, Vector<double>>
                {
                    ErrorStats = ErrorStats<double>.Empty(),
                    PredictionStats = PredictionStats<double>.Empty()
                },
                TestSet = new DataSetStats<double, Matrix<double>, Vector<double>>
                {
                    ErrorStats = ErrorStats<double>.Empty(),
                    PredictionStats = PredictionStats<double>.Empty()
                }
            };
        }

        [Fact]
        public void Constructor_WithDefaultOptions_InitializesSuccessfully()
        {
            // Arrange & Act
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();

            // Assert
            Assert.NotNull(detector);
        }

        [Fact]
        public void Constructor_WithCustomOptions_InitializesSuccessfully()
        {
            // Arrange
            var options = new BayesianFitDetectorOptions
            {
                GoodFitThreshold = 3.0,
                OverfitThreshold = 8.0,
                UnderfitThreshold = 1.5
            };

            // Act
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>(options);

            // Assert
            Assert.NotNull(detector);
        }

        [Fact]
        public void DetectFit_WithGoodFitMetrics_ReturnsGoodFit()
        {
            // Arrange
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                dic: 3.0,    // < GoodFitThreshold (5.0)
                waic: 4.0,   // < GoodFitThreshold (5.0)
                loo: 3.5,    // < GoodFitThreshold (5.0)
                posteriorCheck: 0.8,
                bayesFactor: 0.9
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
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                dic: 12.0,   // > OverfitThreshold (10.0)
                waic: 8.0,
                loo: 7.0,
                posteriorCheck: 0.5,
                bayesFactor: 0.4
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
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                dic: 1.0,    // < UnderfitThreshold (2.0)
                waic: 1.5,   // < UnderfitThreshold (2.0)
                loo: 1.2,    // < UnderfitThreshold (2.0)
                posteriorCheck: 0.3,
                bayesFactor: 0.2
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
        public void DetectFit_WithUnstableMetrics_ReturnsUnstable()
        {
            // Arrange
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                dic: 6.0,    // Between thresholds but inconsistent
                waic: 3.0,
                loo: 11.0,   // Mixed signals
                posteriorCheck: 0.5,
                bayesFactor: 0.5
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
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                dic: 3.0,
                waic: 4.0,
                loo: 3.5,
                posteriorCheck: 0.8,
                bayesFactor: 0.9
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.0);
            Assert.True(result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void DetectFit_IncludesBayesianMetricsInRecommendations()
        {
            // Arrange
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                dic: 3.0,
                waic: 4.0,
                loo: 3.5,
                posteriorCheck: 0.8,
                bayesFactor: 0.9
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.Recommendations);
            Assert.Contains(result.Recommendations, r => r.Contains("DIC:"));
            Assert.Contains(result.Recommendations, r => r.Contains("WAIC:"));
            Assert.Contains(result.Recommendations, r => r.Contains("LOO:"));
            Assert.Contains(result.Recommendations, r => r.Contains("Posterior Predictive Check:"));
            Assert.Contains(result.Recommendations, r => r.Contains("Bayes Factor:"));
        }

        [Fact]
        public void DetectFit_WithCustomThresholds_UsesCustomThresholds()
        {
            // Arrange
            var options = new BayesianFitDetectorOptions
            {
                GoodFitThreshold = 6.0,  // Higher threshold
                OverfitThreshold = 12.0,
                UnderfitThreshold = 3.0
            };
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>(options);
            var evaluationData = CreateTestEvaluationData(
                dic: 5.5,   // Would be above default GoodFitThreshold (5.0) but below custom (6.0)
                waic: 5.8,
                loo: 5.2,
                posteriorCheck: 0.7,
                bayesFactor: 0.8
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(FitType.GoodFit, result.FitType);
        }

        [Fact]
        public void DetectFit_OverfitRecommendations_ContainsPriorAdvice()
        {
            // Arrange
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                dic: 15.0,
                waic: 12.0,
                loo: 11.0,
                posteriorCheck: 0.3,
                bayesFactor: 0.2
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.Equal(FitType.Overfit, result.FitType);
            Assert.Contains(result.Recommendations, r => r.Contains("priors") || r.Contains("prior"));
        }

        [Fact]
        public void DetectFit_UnderfitRecommendations_ContainsComplexityAdvice()
        {
            // Arrange
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                dic: 0.5,
                waic: 0.8,
                loo: 0.6,
                posteriorCheck: 0.2,
                bayesFactor: 0.1
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.Equal(FitType.Underfit, result.FitType);
            Assert.Contains(result.Recommendations, r => r.Contains("complexity") || r.Contains("features"));
        }

        [Fact]
        public void DetectFit_UnstableRecommendations_ContainsMCMCAdvice()
        {
            // Arrange
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                dic: 7.0,
                waic: 3.0,
                loo: 12.0,
                posteriorCheck: 0.4,
                bayesFactor: 0.3
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.Equal(FitType.Unstable, result.FitType);
            Assert.Contains(result.Recommendations, r => r.Contains("MCMC") || r.Contains("convergence") || r.Contains("multimodality"));
        }

        [Fact]
        public void DetectFit_WithHighConfidenceMetrics_ReturnsHighConfidence()
        {
            // Arrange
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                dic: 3.0,
                waic: 4.0,
                loo: 3.5,
                posteriorCheck: 0.95,  // High
                bayesFactor: 0.98      // High
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.8);
        }

        [Fact]
        public void DetectFit_WithLowConfidenceMetrics_ReturnsLowConfidence()
        {
            // Arrange
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(
                dic: 3.0,
                waic: 4.0,
                loo: 3.5,
                posteriorCheck: 0.1,   // Low
                bayesFactor: 0.05      // Low
            );

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel <= 0.2);
        }

        [Fact]
        public void DetectFit_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var detector = new BayesianFitDetector<float, Matrix<float>, Vector<float>>();
            var modelStats = ModelStats<float, Matrix<float>, Vector<float>>.Empty();

            var dicField = typeof(ModelStats<float, Matrix<float>, Vector<float>>).GetField("_dic",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var waicField = typeof(ModelStats<float, Matrix<float>, Vector<float>>).GetField("_waic",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var looField = typeof(ModelStats<float, Matrix<float>, Vector<float>>).GetField("_loo",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var posteriorField = typeof(ModelStats<float, Matrix<float>, Vector<float>>).GetField("_posteriorPredictiveCheck",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var bayesFactorField = typeof(ModelStats<float, Matrix<float>, Vector<float>>).GetField("_bayesFactor",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            dicField?.SetValue(modelStats, 3.0f);
            waicField?.SetValue(modelStats, 4.0f);
            looField?.SetValue(modelStats, 3.5f);
            posteriorField?.SetValue(modelStats, 0.8f);
            bayesFactorField?.SetValue(modelStats, 0.9f);

            var evaluationData = new ModelEvaluationData<float, Matrix<float>, Vector<float>>
            {
                ModelStats = modelStats,
                TrainingSet = new DataSetStats<float, Matrix<float>, Vector<float>>
                {
                    ErrorStats = ErrorStats<float>.Empty(),
                    PredictionStats = PredictionStats<float>.Empty()
                },
                ValidationSet = new DataSetStats<float, Matrix<float>, Vector<float>>
                {
                    ErrorStats = ErrorStats<float>.Empty(),
                    PredictionStats = PredictionStats<float>.Empty()
                },
                TestSet = new DataSetStats<float, Matrix<float>, Vector<float>>
                {
                    ErrorStats = ErrorStats<float>.Empty(),
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
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateTestEvaluationData(3.0, 4.0, 3.5, 0.7, 0.8);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.Recommendations);
            Assert.NotEmpty(result.Recommendations);
            Assert.True(result.Recommendations.Count > 5); // At least fit advice + 5 metrics
        }
    }
}
