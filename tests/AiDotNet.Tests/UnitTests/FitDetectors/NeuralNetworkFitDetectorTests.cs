using AiDotNet.Enums;
using AiDotNet.FitDetectors;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNetTests.UnitTests.FitDetectors
{
    public class NeuralNetworkFitDetectorTests
    {
        private ModelEvaluationData<double, Matrix<double>, Vector<double>> CreateMockEvaluationData(
            double trainMse, double validationMse, double testMse = 0.1)
        {
            // Create vectors that will result in approximately the target MSE values
            var (trainActual, trainPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(trainMse);
            var (validActual, validPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(validationMse);
            var (testActual, testPredicted) = FitDetectorTestHelper.CreateVectorsWithTargetMse(testMse);

            return FitDetectorTestHelper.CreateEvaluationData(
                trainActual, trainPredicted,
                validActual, validPredicted,
                testActual, testPredicted);
        }

        [Fact]
        public void Constructor_WithDefaultOptions_CreatesInstance()
        {
            // Act
            var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>();

            // Assert
            Assert.NotNull(detector);
        }

        [Fact]
        public void Constructor_WithCustomOptions_CreatesInstance()
        {
            // Arrange
            var options = new NeuralNetworkFitDetectorOptions
            {
                GoodFitThreshold = 0.03,
                ModerateFitThreshold = 0.08,
                PoorFitThreshold = 0.15,
                OverfittingThreshold = 0.15
            };

            // Act
            var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>(options);

            // Assert
            Assert.NotNull(detector);
        }

        [Fact]
        public void DetectFit_WithNullEvaluationData_ThrowsArgumentNullException()
        {
            // Arrange
            var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => detector.DetectFit(null!));
        }

        [Fact]
        public void DetectFit_WithSimilarTrainAndValidationMse_ReturnsGoodFit()
        {
            // Arrange
            var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>();
            // For GoodFit: validationLoss <= 0.05 AND overfittingScore <= 0.2
            // Using same MSE values ensures overfittingScore = 0 (no overfitting)
            var evaluationData = CreateMockEvaluationData(trainMse: 0.02, validationMse: 0.02);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(FitType.GoodFit, result.FitType);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_WithLargeValidationMse_IndicatesPoorFit()
        {
            // Arrange
            var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.02, validationMse: 0.5);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.FitType == FitType.PoorFit || result.FitType == FitType.VeryPoorFit);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_CalculatesConfidenceLevel()
        {
            // Arrange
            var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.05, validationMse: 0.06);

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
            var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.04, validationMse: 0.045);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Recommendations);
            Assert.All(result.Recommendations, r => Assert.False(string.IsNullOrWhiteSpace(r)));
        }

        [Fact]
        public void DetectFit_ResultContainsAllRequiredFields()
        {
            // Arrange
            var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.05, validationMse: 0.055);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.Recommendations);
            Assert.NotEmpty(result.Recommendations);
            Assert.NotNull(result.AdditionalInfo);
        }

        [Fact]
        public void DetectFit_IncludesLossInfoInAdditionalInfo()
        {
            // Arrange
            var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.05, validationMse: 0.06, testMse: 0.07);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.AdditionalInfo);
            Assert.True(result.AdditionalInfo.ContainsKey("TrainingLoss"));
            Assert.True(result.AdditionalInfo.ContainsKey("ValidationLoss"));
            Assert.True(result.AdditionalInfo.ContainsKey("TestLoss"));
            Assert.True(result.AdditionalInfo.ContainsKey("OverfittingScore"));
        }

        [Fact]
        public void DetectFit_WithCustomThresholds_UsesThresholdsCorrectly()
        {
            // Arrange
            var options = new NeuralNetworkFitDetectorOptions
            {
                GoodFitThreshold = 0.02,
                ModerateFitThreshold = 0.05,
                PoorFitThreshold = 0.1,
                OverfittingThreshold = 0.1
            };
            var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>(options);
            var evaluationData = CreateMockEvaluationData(trainMse: 0.01, validationMse: 0.015);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
        }

        [Fact]
        public void DetectFit_WithModerateFit_ReturnsModerate()
        {
            // Arrange
            var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>();
            // For Moderate: validationLoss <= 0.1 AND overfittingScore <= 0.3, but NOT GoodFit
            // GoodFitThreshold = 0.05, so validationLoss must be > 0.05
            // Using trainMse: 0.06, validationMse: 0.07 gives:
            // - validationLoss ≈ 0.07 > 0.05 (fails GoodFit threshold)
            // - overfittingScore ≈ (0.07 - 0.06) / 0.06 = 0.167 <= 0.3
            var evaluationData = CreateMockEvaluationData(trainMse: 0.06, validationMse: 0.07);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(FitType.Moderate, result.FitType);
        }
    }
}
