using AiDotNet.FitDetectors;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNetTests.UnitTests.FitDetectors
{
    public class BayesianFitDetectorTests
    {
        private ModelEvaluationData<double, Matrix<double>, Vector<double>> CreateMockEvaluationData(
            Vector<double> predicted, Vector<double> actual)
        {
            return new ModelEvaluationData<double, Matrix<double>, Vector<double>>
            {
                ModelStats = new ModelStats<double, Matrix<double>, Vector<double>>
                {
                    Predicted = predicted,
                    Actual = actual
                }
            };
        }

        [Fact]
        public void Constructor_WithDefaultOptions_CreatesInstance()
        {
            // Act
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();

            // Assert
            Assert.NotNull(detector);
        }

        [Fact]
        public void Constructor_WithCustomOptions_CreatesInstance()
        {
            // Arrange
            var options = new BayesianFitDetectorOptions
            {
                PriorStrength = 0.5,
                CredibleIntervalLevel = 0.9
            };

            // Act
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>(options);

            // Assert
            Assert.NotNull(detector);
        }

        [Fact]
        public void DetectFit_WithValidData_ReturnsResult()
        {
            // Arrange
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var actual = new Vector<double>(new double[] { 1.1, 1.9, 3.1, 3.9, 5.1 });

            var evaluationData = CreateMockEvaluationData(predicted, actual);

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
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            var evaluationData = CreateMockEvaluationData(predicted, actual);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.0);
            Assert.True(result.ConfidenceLevel <= 1.0);
        }

        [Fact]
        public void DetectFit_GeneratesRecommendations()
        {
            // Arrange
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var actual = new Vector<double>(new double[] { 1.2, 1.8, 3.2, 3.8, 5.2 });

            var evaluationData = CreateMockEvaluationData(predicted, actual);

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
            var detector = new BayesianFitDetector<double, Matrix<double>, Vector<double>>();
            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var actual = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            var evaluationData = CreateMockEvaluationData(predicted, actual);

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
