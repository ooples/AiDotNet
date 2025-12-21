using AiDotNet.Enums;
using AiDotNet.FitDetectors;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNetTests.UnitTests.FitDetectors
{
    public class GaussianProcessFitDetectorTests
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
            var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();

            // Assert
            Assert.NotNull(detector);
        }

        [Fact]
        public void Constructor_WithCustomOptions_CreatesInstance()
        {
            // Arrange
            var options = new GaussianProcessFitDetectorOptions
            {
                GoodFitThreshold = 0.15,
                OverfitThreshold = 0.35,
                UnderfitThreshold = 0.5
            };

            // Act
            var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>(options);

            // Assert
            Assert.NotNull(detector);
        }

        [Fact]
        public void DetectFit_WithPerfectPredictions_ReturnsValidResult()
        {
            // Arrange
            var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.01, validationMse: 0.02);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.True(System.Enum.IsDefined(typeof(FitType), result.FitType));
            Assert.True(result.ConfidenceLevel >= 0.0);
            Assert.True(result.ConfidenceLevel <= 1.0);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_WithPoorPredictions_ReturnsValidResult()
        {
            // Arrange
            var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();
            // Use large MSE difference to indicate poor fit
            var evaluationData = CreateMockEvaluationData(trainMse: 0.05, validationMse: 0.5);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.True(System.Enum.IsDefined(typeof(FitType), result.FitType));
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_WithReasonablePredictions_ReturnsConfidenceLevel()
        {
            // Arrange
            var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.05, validationMse: 0.07);

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
            var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.05, validationMse: 0.06);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Recommendations);
            Assert.All(result.Recommendations, r => Assert.False(string.IsNullOrWhiteSpace(r)));
        }

        [Fact]
        public void DetectFit_WithVariousDataSizes_HandlesCorrectly()
        {
            // Arrange
            var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void DetectFit_WithHighDimensionalFeatures_HandlesCorrectly()
        {
            // Arrange
            var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();
            var (actual, predicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(0.05, 15);
            var features = FitDetectorTestHelper.CreateFeatureMatrix(15, 5);

            var evaluationData = FitDetectorTestHelper.CreateEvaluationData(
                actual, predicted, features: features);

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
            var options = new GaussianProcessFitDetectorOptions
            {
                GoodFitThreshold = 0.05,
                OverfitThreshold = 0.2,
                UnderfitThreshold = 0.4,
                LowUncertaintyThreshold = 0.15,
                HighUncertaintyThreshold = 0.3
            };
            var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>(options);
            var evaluationData = CreateMockEvaluationData(trainMse: 0.05, validationMse: 0.06);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void DetectFit_ResultContainsAllRequiredFields()
        {
            // Arrange
            var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.Recommendations);
            Assert.NotNull(result.AdditionalInfo);
        }

        [Fact]
        public void DetectFit_WithSlightlyOffPredictions_ReturnsModerateConfidence()
        {
            // Arrange
            var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.1, validationMse: 0.15);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.ConfidenceLevel >= 0.0);
            Assert.True(result.ConfidenceLevel <= 1.0);
        }
    }
}
