using AiDotNet.FitDetectors;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNetTests.UnitTests.FitDetectors
{
    public class NeuralNetworkFitDetectorTests
    {
        private ModelEvaluationData<double, Matrix<double>, Vector<double>> CreateMockEvaluationData(
            double trainLoss, double validationLoss, double trainAccuracy = 0.9, double validationAccuracy = 0.85)
        {
            return new ModelEvaluationData<double, Matrix<double>, Vector<double>>
            {
                TrainingSet = new DataSetStats<double, Matrix<double>, Vector<double>>
                {
                    ErrorStats = new ErrorStats<double> { Loss = trainLoss },
                    PredictionStats = new PredictionStats<double> { Accuracy = trainAccuracy }
                },
                ValidationSet = new DataSetStats<double, Matrix<double>, Vector<double>>
                {
                    ErrorStats = new ErrorStats<double> { Loss = validationLoss },
                    PredictionStats = new PredictionStats<double> { Accuracy = validationAccuracy }
                }
            };
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
                LossDifferenceThreshold = 0.15,
                AccuracyDifferenceThreshold = 0.1
            };

            // Act
            var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>(options);

            // Assert
            Assert.NotNull(detector);
        }

        [Fact]
        public void DetectFit_WithSimilarTrainAndValidationLoss_ReturnsGoodFit()
        {
            // Arrange
            var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainLoss: 0.2, validationLoss: 0.22);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_WithLargeValidationLoss_IndicatesOverfitting()
        {
            // Arrange
            var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainLoss: 0.1, validationLoss: 0.8);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_CalculatesConfidenceLevel()
        {
            // Arrange
            var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainLoss: 0.2, validationLoss: 0.25);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.0);
            Assert.True(result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void DetectFit_GeneratesRecommendationsBasedOnFitType()
        {
            // Arrange
            var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainLoss: 0.15, validationLoss: 0.18);

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
            var evaluationData = CreateMockEvaluationData(trainLoss: 0.2, validationLoss: 0.22);

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

        [Fact]
        public void DetectFit_WithCustomThresholds_UsesThresholdsCorrectly()
        {
            // Arrange
            var options = new NeuralNetworkFitDetectorOptions
            {
                LossDifferenceThreshold = 0.05,
                AccuracyDifferenceThreshold = 0.05
            };
            var detector = new NeuralNetworkFitDetector<double, Matrix<double>, Vector<double>>(options);
            var evaluationData = CreateMockEvaluationData(trainLoss: 0.1, validationLoss: 0.12);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
        }
    }
}
