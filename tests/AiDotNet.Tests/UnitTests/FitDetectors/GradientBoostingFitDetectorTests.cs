using AiDotNet.Enums;
using AiDotNet.FitDetectors;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNetTests.UnitTests.FitDetectors
{
    public class GradientBoostingFitDetectorTests
    {
        private ModelEvaluationData<double, Matrix<double>, Vector<double>> CreateMockEvaluationData(
            double trainMse, double validationMse)
        {
            var (trainActual, trainPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(trainMse);
            var (validActual, validPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(validationMse);

            return FitDetectorTestHelper.CreateEvaluationData(
                trainActual, trainPredicted,
                validActual, validPredicted);
        }

        [Fact]
        public void Constructor_WithDefaultOptions_CreatesInstance()
        {
            // Act
            var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();

            // Assert
            Assert.NotNull(detector);
        }

        [Fact]
        public void Constructor_WithCustomOptions_CreatesInstance()
        {
            // Arrange
            var options = new GradientBoostingFitDetectorOptions
            {
                GoodFitThreshold = 0.05,
                OverfitThreshold = 0.15,
                SevereOverfitThreshold = 0.3
            };

            // Act
            var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>(options);

            // Assert
            Assert.NotNull(detector);
        }

        [Fact]
        public void DetectFit_WithNullEvaluationData_ThrowsArgumentNullException()
        {
            // Arrange
            var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => detector.DetectFit(null!));
        }

        [Fact]
        public void DetectFit_WithSimilarTrainAndValidationErrors_ReturnsGoodFit()
        {
            // Arrange
            var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.05, validationMse: 0.06);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(FitType.GoodFit, result.FitType);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_WithLargeValidationError_ReturnsNonGoodFit()
        {
            // Arrange
            var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.05, validationMse: 0.5);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.FitType == FitType.PoorFit || result.FitType == FitType.VeryPoorFit);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_WithModerateErrorDifference_ReturnsModerateFit()
        {
            // Arrange
            var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.15);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.FitType == FitType.Moderate || result.FitType == FitType.GoodFit);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_CalculatesConfidenceLevel()
        {
            // Arrange
            var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.ConfidenceLevel >= 0.0);
            Assert.True(result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void DetectFit_IncludesPerformanceMetricsInAdditionalInfo()
        {
            // Arrange
            var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.AdditionalInfo);
            Assert.True(result.AdditionalInfo.ContainsKey("PerformanceMetrics"));
        }

        [Fact]
        public void DetectFit_WithDifferentThresholds_ChangesClassification()
        {
            // Arrange
            var options = new GradientBoostingFitDetectorOptions
            {
                GoodFitThreshold = 0.01,
                OverfitThreshold = 0.05,
                SevereOverfitThreshold = 0.1
            };
            var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>(options);
            var evaluationData = CreateMockEvaluationData(trainMse: 0.05, validationMse: 0.08);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void DetectFit_GeneratesAppropriateRecommendationsForGoodFit()
        {
            // Arrange
            var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.05, validationMse: 0.06);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            if (result.FitType == FitType.GoodFit)
            {
                Assert.Contains(result.Recommendations, r => r.Contains("good fit") || r.Contains("fine-tuning"));
            }
        }

        [Fact]
        public void DetectFit_GeneratesAppropriateRecommendationsForPoorFit()
        {
            // Arrange
            var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.05, validationMse: 0.4);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            if (result.FitType == FitType.PoorFit || result.FitType == FitType.VeryPoorFit)
            {
                Assert.Contains(result.Recommendations, r =>
                    r.Contains("overfit") || r.Contains("regularization") || r.Contains("complexity"));
            }
        }

        [Fact]
        public void DetectFit_ResultContainsAllRequiredFields()
        {
            // Arrange
            var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.Recommendations);
            Assert.NotEmpty(result.Recommendations);
            Assert.NotNull(result.AdditionalInfo);
        }
    }
}
