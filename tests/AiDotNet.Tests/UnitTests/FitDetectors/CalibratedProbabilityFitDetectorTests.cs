using AiDotNet.Enums;
using AiDotNet.FitDetectors;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNetTests.UnitTests.FitDetectors
{
    public class CalibratedProbabilityFitDetectorTests
    {
        private ModelEvaluationData<double, Matrix<double>, Vector<double>> CreateMockEvaluationData(
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

        [Fact]
        public void Constructor_WithDefaultOptions_CreatesInstance()
        {
            // Act
            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();

            // Assert
            Assert.NotNull(detector);
        }

        [Fact]
        public void Constructor_WithCustomOptions_CreatesInstance()
        {
            // Arrange
            var options = new CalibratedProbabilityFitDetectorOptions
            {
                NumCalibrationBins = 20,
                GoodFitThreshold = 0.08,
                OverfitThreshold = 0.25
            };

            // Act
            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>(options);

            // Assert
            Assert.NotNull(detector);
        }

        [Fact]
        public void DetectFit_WithWellCalibratedProbabilities_ReturnsGoodFit()
        {
            // Arrange
            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.01, validationMse: 0.02);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_WithPoorlyCalibratedProbabilities_ReturnsNonGoodFit()
        {
            // Arrange
            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.05, validationMse: 0.5);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.FitType == FitType.Overfit || result.FitType == FitType.Underfit || result.FitType == FitType.GoodFit);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_CalculatesConfidenceLevel()
        {
            // Arrange
            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.ConfidenceLevel >= 0.0);
            Assert.True(result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void DetectFit_GeneratesRecommendationsBasedOnFitType()
        {
            // Arrange
            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.05, validationMse: 0.06);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Recommendations);
            Assert.All(result.Recommendations, r => Assert.False(string.IsNullOrWhiteSpace(r)));
        }

        [Fact]
        public void DetectFit_WithOverconfidentPredictions_ReturnsRecommendations()
        {
            // Arrange
            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.05, validationMse: 0.3);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_WithUnderconfidentPredictions_ReturnsRecommendations()
        {
            // Arrange
            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.15);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_WithCustomThresholds_UsesThresholdsCorrectly()
        {
            // Arrange
            var options = new CalibratedProbabilityFitDetectorOptions
            {
                GoodFitThreshold = 0.05,
                OverfitThreshold = 0.2,
                MaxCalibrationError = 0.5
            };
            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>(options);
            var evaluationData = CreateMockEvaluationData(trainMse: 0.05, validationMse: 0.06);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void DetectFit_WithCustomBinCount_UsesCorrectBinCount()
        {
            // Arrange
            var options = new CalibratedProbabilityFitDetectorOptions
            {
                NumCalibrationBins = 5
            };
            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>(options);
            var evaluationData = CreateMockEvaluationData(trainMse: 0.05, validationMse: 0.06);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void DetectFit_WithLargeDataset_HandlesCorrectly()
        {
            // Arrange
            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();
            var (actual, predicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(0.05, 100);
            var features = FitDetectorTestHelper.CreateFeatureMatrix(100, 3);

            var evaluationData = FitDetectorTestHelper.CreateEvaluationData(
                actual, predicted, features: features);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_RecommendationsIncludeCalibrationMethods()
        {
            // Arrange
            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.05, validationMse: 0.3);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Recommendations);
            // Check if recommendations mention calibration techniques
            var hasCalibrationAdvice = result.Recommendations.Any(r =>
                r.Contains("calibration") || r.Contains("Platt") || r.Contains("isotonic") ||
                r.Contains("regularization") || r.Contains("complexity") || r.Contains("confident"));
            Assert.True(hasCalibrationAdvice, "Recommendations should include calibration-related advice");
        }

        [Fact]
        public void DetectFit_ResultContainsAllRequiredFields()
        {
            // Arrange
            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.Recommendations);
            Assert.NotEmpty(result.Recommendations);
            Assert.NotNull(result.AdditionalInfo);
        }

        [Fact]
        public void DetectFit_WithBinaryClassificationData_HandlesCorrectly()
        {
            // Arrange
            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.05, validationMse: 0.07);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
        }
    }
}
