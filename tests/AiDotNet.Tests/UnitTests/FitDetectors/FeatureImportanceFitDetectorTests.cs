using AiDotNet.Enums;
using AiDotNet.Exceptions;
using AiDotNet.FitDetectors;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNetTests.UnitTests.FitDetectors
{
    public class FeatureImportanceFitDetectorTests
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
            var detector = new FeatureImportanceFitDetector<double, Matrix<double>, Vector<double>>();

            // Assert
            Assert.NotNull(detector);
        }

        [Fact]
        public void Constructor_WithCustomOptions_CreatesInstance()
        {
            // Arrange
            var options = new FeatureImportanceFitDetectorOptions
            {
                HighImportanceThreshold = 0.3,
                LowImportanceThreshold = 0.05,
                CorrelationThreshold = 0.8
            };

            // Act
            var detector = new FeatureImportanceFitDetector<double, Matrix<double>, Vector<double>>(options);

            // Assert
            Assert.NotNull(detector);
        }

        [Fact]
        public void DetectFit_WithoutModel_ThrowsException()
        {
            // Arrange
            // FeatureImportanceFitDetector requires a Model to calculate feature importances
            // via permutation testing. Without a model, it should throw an exception.
            var detector = new FeatureImportanceFitDetector<double, Matrix<double>, Vector<double>>();
            var evaluationData = CreateMockEvaluationData(trainMse: 0.05, validationMse: 0.06);

            // Act & Assert
            var exception = Assert.Throws<AiDotNetException>(() => detector.DetectFit(evaluationData));
            Assert.Contains("Model is null", exception.Message);
        }

        [Fact]
        public void DetectFit_WithCustomThresholds_WithoutModel_ThrowsException()
        {
            // Arrange
            var options = new FeatureImportanceFitDetectorOptions
            {
                HighImportanceThreshold = 0.2,
                LowImportanceThreshold = 0.03,
                LowVarianceThreshold = 0.05,
                HighVarianceThreshold = 0.15
            };
            var detector = new FeatureImportanceFitDetector<double, Matrix<double>, Vector<double>>(options);
            var evaluationData = CreateMockEvaluationData(trainMse: 0.05, validationMse: 0.06);

            // Act & Assert
            var exception = Assert.Throws<AiDotNetException>(() => detector.DetectFit(evaluationData));
            Assert.Contains("Model is null", exception.Message);
        }
    }
}
