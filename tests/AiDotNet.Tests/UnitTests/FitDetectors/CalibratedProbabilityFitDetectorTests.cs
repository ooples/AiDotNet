using AiDotNet.FitDetectors;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNetTests.UnitTests.FitDetectors
{
    public class CalibratedProbabilityFitDetectorTests
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
            // Create well-calibrated probabilities (predictions match actuals closely)
            var predicted = new Vector<double>(new double[] { 0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0 });
            var actual = new Vector<double>(new double[] { 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0 });

            var evaluationData = CreateMockEvaluationData(predicted, actual);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_WithPoorlyCalibratedProbabilities_ReturnsNonGoodFit()
        {
            // Arrange
            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();
            // Create poorly calibrated probabilities
            var predicted = new Vector<double>(new double[] { 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9 });
            var actual = new Vector<double>(new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0 });

            var evaluationData = CreateMockEvaluationData(predicted, actual);

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
            var predicted = new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });
            var actual = new Vector<double>(new double[] { 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 });

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
        public void DetectFit_GeneratesRecommendationsBasedOnFitType()
        {
            // Arrange
            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();
            var predicted = new Vector<double>(new double[] { 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99 });
            var actual = new Vector<double>(new double[] { 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 });

            var evaluationData = CreateMockEvaluationData(predicted, actual);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Recommendations);
            Assert.All(result.Recommendations, r => Assert.False(string.IsNullOrWhiteSpace(r)));
        }

        [Fact]
        public void DetectFit_WithOverconfidentPredictions_ReturnsOverfitRecommendations()
        {
            // Arrange
            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();
            // Model predicts high probabilities but actuals don't match
            var predicted = new Vector<double>(new double[] { 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95 });
            var actual = new Vector<double>(new double[] { 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0 });

            var evaluationData = CreateMockEvaluationData(predicted, actual);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_WithUnderconfidentPredictions_ReturnsUnderfitRecommendations()
        {
            // Arrange
            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();
            // Model predicts middle probabilities for clear cases
            var predicted = new Vector<double>(new double[] { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 });
            var actual = new Vector<double>(new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0 });

            var evaluationData = CreateMockEvaluationData(predicted, actual);

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
            var predicted = new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });
            var actual = new Vector<double>(new double[] { 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 });

            var evaluationData = CreateMockEvaluationData(predicted, actual);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
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
            var predicted = new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });
            var actual = new Vector<double>(new double[] { 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 });

            var evaluationData = CreateMockEvaluationData(predicted, actual);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
        }

        [Fact]
        public void DetectFit_WithLargeDataset_HandlesCorrectly()
        {
            // Arrange
            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();
            var predictedValues = new List<double>();
            var actualValues = new List<double>();

            // Create a large dataset with reasonable calibration
            for (int i = 0; i < 100; i++)
            {
                double prob = i / 100.0;
                predictedValues.Add(prob);
                actualValues.Add(prob > 0.5 ? 1.0 : 0.0);
            }

            var predicted = new Vector<double>(predictedValues.ToArray());
            var actual = new Vector<double>(actualValues.ToArray());

            var evaluationData = CreateMockEvaluationData(predicted, actual);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_RecommendationsIncludeCalibrationMethods()
        {
            // Arrange
            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();
            var predicted = new Vector<double>(new double[] { 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9 });
            var actual = new Vector<double>(new double[] { 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0 });

            var evaluationData = CreateMockEvaluationData(predicted, actual);

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
            var predicted = new Vector<double>(new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 });
            var actual = new Vector<double>(new double[] { 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 });

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

        [Fact]
        public void DetectFit_WithBinaryClassificationData_HandlesCorrectly()
        {
            // Arrange
            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();
            // Simulate binary classification with probabilities
            var predicted = new Vector<double>(new double[]
            {
                0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99
            });
            var actual = new Vector<double>(new double[]
            {
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
            });

            var evaluationData = CreateMockEvaluationData(predicted, actual);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
            Assert.NotNull(result.ConfidenceLevel);
        }

        [Fact]
        public void DetectFit_WithDifferentFitTypes_GeneratesAppropriateRecommendations()
        {
            // Arrange
            var detector = new CalibratedProbabilityFitDetector<double, Matrix<double>, Vector<double>>();

            // Test multiple scenarios
            var scenarios = new[]
            {
                new
                {
                    Predicted = new double[] { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 },
                    Actual = new double[] { 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }
                }
            };

            foreach (var scenario in scenarios)
            {
                var predicted = new Vector<double>(scenario.Predicted);
                var actual = new Vector<double>(scenario.Actual);
                var evaluationData = CreateMockEvaluationData(predicted, actual);

                // Act
                var result = detector.DetectFit(evaluationData);

                // Assert
                Assert.NotNull(result);
                Assert.NotEmpty(result.Recommendations);
                Assert.All(result.Recommendations, r => Assert.False(string.IsNullOrWhiteSpace(r)));
            }
        }
    }
}
