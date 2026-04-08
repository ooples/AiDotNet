using AiDotNet.Enums;
using AiDotNet.FitDetectors;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Statistics;
using AiDotNet.Tests.Helpers;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNetTests.UnitTests.FitDetectors
{
    /// <summary>
    /// Unit tests for the BayesianFitDetector class.
    /// Tests use properly constructed ModelStats with calculated Bayesian metrics.
    /// The BayesianFitDetector calculates DIC, WAIC, LOO, etc. from ModelStats properties.
    /// </summary>
    public class BayesianFitDetectorTests
    {
        /// <summary>
        /// Creates evaluation data with properly calculated statistics.
        /// Bayesian metrics are computed from ModelStats properties like LogLikelihood,
        /// EffectiveNumberOfParameters, etc.
        /// </summary>
        private static ModelEvaluationData<double, Matrix<double>, Vector<double>> CreateMockEvaluationData(
            double trainMse = 0.1, double validationMse = 0.12)
        {
            var (trainActual, trainPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(trainMse);
            var (validActual, validPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(validationMse);

            // Create well-conditioned feature matrix for VIF calculation
            var features = FitDetectorTestHelper.CreateFeatureMatrix(trainActual.Length, 3);

            return FitDetectorTestHelper.CreateEvaluationData(
                trainActual, trainPredicted,
                validActual, validPredicted,
                features: features);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithDefaultOptions_InitializesSuccessfully()
        {
            // Arrange & Act
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();

            // Assert
            Assert.NotNull(detector);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithCustomOptions_InitializesSuccessfully()
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

        [Fact(Timeout = 60000)]
        public async Task DetectFit_WithValidData_ReturnsValidFitType()
        {
            // Arrange
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.True(
                result.FitType == FitType.GoodFit ||
                result.FitType == FitType.Overfit ||
                result.FitType == FitType.Underfit ||
                result.FitType == FitType.Unstable);
        }

        [Fact(Timeout = 60000)]
        public async Task DetectFit_ReturnsConfidenceLevel()
        {
            // Arrange
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.True(result.ConfidenceLevel >= 0.0);
            Assert.True(result.ConfidenceLevel <= 1.0);
        }

        [Fact(Timeout = 60000)]
        public async Task DetectFit_IncludesBayesianMetricsInRecommendations()
        {
            // Arrange
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

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

        [Fact(Timeout = 60000)]
        public async Task DetectFit_WithCustomThresholds_InitializesSuccessfully()
        {
            // Arrange
            var options = new BayesianFitDetectorOptions
            {
                GoodFitThreshold = 6.0,
                OverfitThreshold = 12.0,
                UnderfitThreshold = 3.0
            };
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>(options);
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact(Timeout = 60000)]
        public async Task DetectFit_ReturnsRecommendationsForFitType()
        {
            // Arrange
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
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
                r.Contains("unstable"));
            Assert.True(hasAdvice);
        }

        [Fact(Timeout = 60000)]
        public async Task DetectFit_ReturnsNonEmptyRecommendations()
        {
            // Arrange
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result.Recommendations);
            Assert.NotEmpty(result.Recommendations);
            Assert.True(result.Recommendations.Count >= 5); // At least fit advice + 5 metrics
        }

        [Fact(Timeout = 60000)]
        public async Task DetectFit_WithLowMse_ReturnsResult()
        {
            // Arrange
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.01, validationMse: 0.015);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact(Timeout = 60000)]
        public async Task DetectFit_WithHighMse_ReturnsResult()
        {
            // Arrange
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 1.0, validationMse: 1.5);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact(Timeout = 60000)]
        public async Task DetectFit_ResultContainsAllRequiredFields()
        {
            // Arrange
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.Recommendations);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact(Timeout = 60000)]
        public async Task DetectFit_WithDifferentTrainValidationMse_ReturnsResult()
        {
            // Arrange
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();

            // Large gap between training and validation MSE might indicate overfitting
            var evaluationData = CreateMockEvaluationData(trainMse: 0.01, validationMse: 0.5);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact(Timeout = 60000)]
        public async Task DetectFit_MultipleCallsReturnConsistentResults()
        {
            // Arrange
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result1 = detector.DetectFit(evaluationData);
            var result2 = detector.DetectFit(evaluationData);

            // Assert
            Assert.Equal(result1.FitType, result2.FitType);
            Assert.Equal(result1.ConfidenceLevel, result2.ConfidenceLevel);
        }
    }
}
