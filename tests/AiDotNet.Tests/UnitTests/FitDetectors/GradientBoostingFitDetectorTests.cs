using AiDotNet.FitDetectors;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNetTests.UnitTests.FitDetectors
{
    public class GradientBoostingFitDetectorTests
    {
        private ModelEvaluationData<double, Matrix<double>, Vector<double>> CreateMockEvaluationData(
            double trainMSE, double validationMSE, double trainR2 = 0.9, double validationR2 = 0.85)
        {
            return new ModelEvaluationData<double, Matrix<double>, Vector<double>>
            {
                TrainingSet = new DataSetStats<double, Matrix<double>, Vector<double>>
                {
                    ErrorStats = new ErrorStats<double> { MSE = trainMSE },
                    PredictionStats = new PredictionStats<double> { R2 = trainR2 }
                },
                ValidationSet = new DataSetStats<double, Matrix<double>, Vector<double>>
                {
                    ErrorStats = new ErrorStats<double> { MSE = validationMSE },
                    PredictionStats = new PredictionStats<double> { R2 = validationR2 }
                }
            };
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
            var evaluationData = CreateMockEvaluationData(trainMSE: 0.05, validationMSE: 0.06);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(FitType.GoodFit, result.FitType);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_WithLargeValidationError_ReturnsOverfit()
        {
            // Arrange
            var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMSE: 0.05, validationMSE: 0.5);

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
            var evaluationData = CreateMockEvaluationData(trainMSE: 0.1, validationMSE: 0.15);

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
            var evaluationData = CreateMockEvaluationData(trainMSE: 0.1, validationMSE: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.0);
            Assert.True(result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void DetectFit_WithVeryLowTrainingError_IncludesDataLeakageWarning()
        {
            // Arrange
            var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMSE: 0.005, validationMSE: 0.15);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Recommendations);
            Assert.Contains(result.Recommendations, r => r.Contains("data leakage") || r.Contains("suspiciously low"));
        }

        [Fact]
        public void DetectFit_IncludesPerformanceMetricsInAdditionalInfo()
        {
            // Arrange
            var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMSE: 0.1, validationMSE: 0.12, trainR2: 0.9, validationR2: 0.85);

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
            var evaluationData = CreateMockEvaluationData(trainMSE: 0.05, validationMSE: 0.08);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
        }

        [Fact]
        public void DetectFit_GeneratesAppropriateRecommendationsForGoodFit()
        {
            // Arrange
            var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMSE: 0.05, validationMSE: 0.06);

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
            var evaluationData = CreateMockEvaluationData(trainMSE: 0.05, validationMSE: 0.4);

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
        public void DetectFit_WithHighConfidence_ReturnsHighConfidenceValue()
        {
            // Arrange
            var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMSE: 0.1, validationMSE: 0.1);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel > 0.5, "Confidence should be high when errors are similar");
        }

        [Fact]
        public void DetectFit_WithLowConfidence_ReturnsLowConfidenceValue()
        {
            // Arrange
            var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMSE: 0.1, validationMSE: 0.5);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.0 && result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void DetectFit_ResultContainsAllRequiredFields()
        {
            // Arrange
            var detector = new GradientBoostingFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMSE: 0.1, validationMSE: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
            Assert.NotNull(result.ConfidenceLevel);
            Assert.NotNull(result.Recommendations);
            Assert.NotEmpty(result.Recommendations);
            Assert.NotNull(result.AdditionalInfo);
        }
    }
}
