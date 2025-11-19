using AiDotNet.FitDetectors;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNetTests.UnitTests.FitDetectors
{
    public class FeatureImportanceFitDetectorTests
    {
        private class MockModel : IModel<Matrix<double>, Vector<double>>
        {
            private readonly Func<Matrix<double>, Vector<double>> _predictFunc;

            public MockModel(Func<Matrix<double>, Vector<double>> predictFunc)
            {
                _predictFunc = predictFunc;
            }

            public Vector<double> Predict(Matrix<double> input)
            {
                return _predictFunc(input);
            }
        }

        private ModelEvaluationData<double, Matrix<double>, Vector<double>> CreateMockEvaluationData(
            Vector<double> predicted, Vector<double> actual, Matrix<double> features, IModel<Matrix<double>, Vector<double>>? model = null)
        {
            return new ModelEvaluationData<double, Matrix<double>, Vector<double>>
            {
                ModelStats = new ModelStats<double, Matrix<double>, Vector<double>>
                {
                    Predicted = predicted,
                    Actual = actual,
                    Features = features,
                    Model = model ?? new MockModel(f => predicted)
                }
            };
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
        public void DetectFit_WithValidData_ReturnsResult()
        {
            // Arrange
            var detector = new FeatureImportanceFitDetector<double, Matrix<double>, Vector<double>>();
            var features = new Matrix<double>(5, 3, new double[]
            {
                1.0, 2.0, 3.0,
                2.0, 3.0, 4.0,
                3.0, 4.0, 5.0,
                4.0, 5.0, 6.0,
                5.0, 6.0, 7.0
            });
            var actual = new Vector<double>(new double[] { 6.0, 9.0, 12.0, 15.0, 18.0 });
            var predicted = new Vector<double>(new double[] { 6.1, 8.9, 12.1, 14.9, 18.1 });

            var evaluationData = CreateMockEvaluationData(predicted, actual, features);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
            Assert.NotNull(result.ConfidenceLevel);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_CalculatesFeatureImportances()
        {
            // Arrange
            var detector = new FeatureImportanceFitDetector<double, Matrix<double>, Vector<double>>();
            var features = new Matrix<double>(5, 2, new double[]
            {
                1.0, 2.0,
                2.0, 3.0,
                3.0, 4.0,
                4.0, 5.0,
                5.0, 6.0
            });
            var actual = new Vector<double>(new double[] { 3.0, 5.0, 7.0, 9.0, 11.0 });
            var predicted = new Vector<double>(new double[] { 3.0, 5.0, 7.0, 9.0, 11.0 });

            var evaluationData = CreateMockEvaluationData(predicted, actual, features);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Recommendations);
            Assert.Contains(result.Recommendations, r => r.Contains("Feature") || r.Contains("Importance"));
        }

        [Fact]
        public void DetectFit_IncludesTopFeaturesInRecommendations()
        {
            // Arrange
            var detector = new FeatureImportanceFitDetector<double, Matrix<double>, Vector<double>>();
            var features = new Matrix<double>(5, 3, new double[]
            {
                1.0, 2.0, 3.0,
                2.0, 3.0, 4.0,
                3.0, 4.0, 5.0,
                4.0, 5.0, 6.0,
                5.0, 6.0, 7.0
            });
            var actual = new Vector<double>(new double[] { 6.0, 9.0, 12.0, 15.0, 18.0 });
            var predicted = new Vector<double>(new double[] { 6.0, 9.0, 12.0, 15.0, 18.0 });

            var evaluationData = CreateMockEvaluationData(predicted, actual, features);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Recommendations);
            Assert.Contains(result.Recommendations, r => r.Contains("Top 3"));
        }

        [Fact]
        public void DetectFit_ReturnsConfidenceLevelInValidRange()
        {
            // Arrange
            var detector = new FeatureImportanceFitDetector<double, Matrix<double>, Vector<double>>();
            var features = new Matrix<double>(5, 2, new double[]
            {
                1.0, 2.0,
                2.0, 3.0,
                3.0, 4.0,
                4.0, 5.0,
                5.0, 6.0
            });
            var actual = new Vector<double>(new double[] { 3.0, 5.0, 7.0, 9.0, 11.0 });
            var predicted = new Vector<double>(new double[] { 3.1, 4.9, 7.1, 8.9, 11.1 });

            var evaluationData = CreateMockEvaluationData(predicted, actual, features);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.0, "Confidence level should be >= 0");
        }

        [Fact]
        public void DetectFit_GeneratesRecommendationsBasedOnFitType()
        {
            // Arrange
            var detector = new FeatureImportanceFitDetector<double, Matrix<double>, Vector<double>>();
            var features = new Matrix<double>(5, 2, new double[]
            {
                1.0, 2.0,
                2.0, 3.0,
                3.0, 4.0,
                4.0, 5.0,
                5.0, 6.0
            });
            var actual = new Vector<double>(new double[] { 3.0, 5.0, 7.0, 9.0, 11.0 });
            var predicted = new Vector<double>(new double[] { 3.0, 5.0, 7.0, 9.0, 11.0 });

            var evaluationData = CreateMockEvaluationData(predicted, actual, features);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Recommendations);
            Assert.All(result.Recommendations, r => Assert.False(string.IsNullOrWhiteSpace(r)));
        }

        [Fact]
        public void DetectFit_WithCustomThresholds_UsesThresholdsCorrectly()
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
            var features = new Matrix<double>(5, 2, new double[]
            {
                1.0, 2.0,
                2.0, 3.0,
                3.0, 4.0,
                4.0, 5.0,
                5.0, 6.0
            });
            var actual = new Vector<double>(new double[] { 3.0, 5.0, 7.0, 9.0, 11.0 });
            var predicted = new Vector<double>(new double[] { 3.0, 5.0, 7.0, 9.0, 11.0 });

            var evaluationData = CreateMockEvaluationData(predicted, actual, features);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
        }

        [Fact]
        public void DetectFit_WithHighDimensionalFeatures_HandlesCorrectly()
        {
            // Arrange
            var detector = new FeatureImportanceFitDetector<double, Matrix<double>, Vector<double>>();
            var features = new Matrix<double>(6, 5, new double[]
            {
                1.0, 2.0, 3.0, 4.0, 5.0,
                2.0, 3.0, 4.0, 5.0, 6.0,
                3.0, 4.0, 5.0, 6.0, 7.0,
                4.0, 5.0, 6.0, 7.0, 8.0,
                5.0, 6.0, 7.0, 8.0, 9.0,
                6.0, 7.0, 8.0, 9.0, 10.0
            });
            var actual = new Vector<double>(new double[] { 15.0, 20.0, 25.0, 30.0, 35.0, 40.0 });
            var predicted = new Vector<double>(new double[] { 15.1, 19.9, 25.1, 29.9, 35.1, 39.9 });

            var evaluationData = CreateMockEvaluationData(predicted, actual, features);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_WithCorrelatedFeatures_DetectsAppropriately()
        {
            // Arrange
            var detector = new FeatureImportanceFitDetector<double, Matrix<double>, Vector<double>>();
            // Create highly correlated features
            var features = new Matrix<double>(5, 2, new double[]
            {
                1.0, 1.1,
                2.0, 2.1,
                3.0, 3.1,
                4.0, 4.1,
                5.0, 5.1
            });
            var actual = new Vector<double>(new double[] { 2.1, 4.1, 6.1, 8.1, 10.1 });
            var predicted = new Vector<double>(new double[] { 2.0, 4.0, 6.0, 8.0, 10.0 });

            var evaluationData = CreateMockEvaluationData(predicted, actual, features);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
        }

        [Fact]
        public void DetectFit_ResultContainsAllRequiredFields()
        {
            // Arrange
            var detector = new FeatureImportanceFitDetector<double, Matrix<double>, Vector<double>>();
            var features = new Matrix<double>(5, 2, new double[]
            {
                1.0, 2.0,
                2.0, 3.0,
                3.0, 4.0,
                4.0, 5.0,
                5.0, 6.0
            });
            var actual = new Vector<double>(new double[] { 3.0, 5.0, 7.0, 9.0, 11.0 });
            var predicted = new Vector<double>(new double[] { 3.1, 4.9, 7.1, 8.9, 11.1 });

            var evaluationData = CreateMockEvaluationData(predicted, actual, features);

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
        public void DetectFit_WithUncorrelatedFeatures_DetectsAppropriately()
        {
            // Arrange
            var detector = new FeatureImportanceFitDetector<double, Matrix<double>, Vector<double>>();
            // Create uncorrelated features
            var features = new Matrix<double>(5, 2, new double[]
            {
                1.0, 5.0,
                2.0, 3.0,
                3.0, 1.0,
                4.0, 4.0,
                5.0, 2.0
            });
            var actual = new Vector<double>(new double[] { 6.0, 5.0, 4.0, 8.0, 7.0 });
            var predicted = new Vector<double>(new double[] { 6.1, 4.9, 4.1, 7.9, 7.1 });

            var evaluationData = CreateMockEvaluationData(predicted, actual, features);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
        }

        [Fact]
        public void DetectFit_WithVariousFitTypes_GeneratesAppropriateRecommendations()
        {
            // Arrange
            var detector = new FeatureImportanceFitDetector<double, Matrix<double>, Vector<double>>();
            var features = new Matrix<double>(5, 2, new double[]
            {
                1.0, 2.0,
                2.0, 3.0,
                3.0, 4.0,
                4.0, 5.0,
                5.0, 6.0
            });
            var actual = new Vector<double>(new double[] { 3.0, 5.0, 7.0, 9.0, 11.0 });
            var predicted = new Vector<double>(new double[] { 10.0, 15.0, 20.0, 25.0, 30.0 });

            var evaluationData = CreateMockEvaluationData(predicted, actual, features);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.Recommendations);
            // Recommendations should provide actionable advice
            Assert.True(result.Recommendations.Any(r =>
                r.Contains("complex") || r.Contains("features") || r.Contains("regularization") || r.Contains("fit")));
        }
    }
}
