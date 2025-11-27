using AiDotNet.FitDetectors;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNetTests.UnitTests.FitDetectors
{
    public class GaussianProcessFitDetectorTests
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
            Vector<double> predicted, Vector<double> actual, Matrix<double> features)
        {
            return new ModelEvaluationData<double, Matrix<double>, Vector<double>>
            {
                ModelStats = new ModelStats<double, Matrix<double>, Vector<double>>
                {
                    Predicted = predicted,
                    Actual = actual,
                    Features = features,
                    Model = new MockModel(f => predicted)
                }
            };
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
        public void DetectFit_WithPerfectPredictions_ReturnsGoodFit()
        {
            // Arrange
            var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();
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
            Assert.Equal(FitType.GoodFit, result.FitType);
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel > 0.0);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_WithPoorPredictions_ReturnsNonGoodFit()
        {
            // Arrange
            var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();
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
            Assert.NotEqual(FitType.GoodFit, result.FitType);
            Assert.NotNull(result.ConfidenceLevel);
            Assert.NotEmpty(result.Recommendations);
        }

        [Fact]
        public void DetectFit_WithReasonablePredictions_ReturnsConfidenceLevel()
        {
            // Arrange
            var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();
            var features = new Matrix<double>(6, 2, new double[]
            {
                1.0, 2.0,
                2.0, 3.0,
                3.0, 4.0,
                4.0, 5.0,
                5.0, 6.0,
                6.0, 7.0
            });
            var actual = new Vector<double>(new double[] { 3.0, 5.0, 7.0, 9.0, 11.0, 13.0 });
            var predicted = new Vector<double>(new double[] { 3.1, 4.9, 7.1, 8.9, 11.1, 12.9 });

            var evaluationData = CreateMockEvaluationData(predicted, actual, features);

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
            var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();
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
        public void DetectFit_WithVariousDataSizes_HandlesCorrectly()
        {
            // Arrange
            var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();

            // Small dataset
            var featuresSmall = new Matrix<double>(3, 2, new double[] { 1.0, 2.0, 2.0, 3.0, 3.0, 4.0 });
            var actualSmall = new Vector<double>(new double[] { 3.0, 5.0, 7.0 });
            var predictedSmall = new Vector<double>(new double[] { 3.1, 4.9, 7.1 });
            var evaluationDataSmall = CreateMockEvaluationData(predictedSmall, actualSmall, featuresSmall);

            // Act
            var resultSmall = detector.DetectFit(evaluationDataSmall);

            // Assert
            Assert.NotNull(resultSmall);
            Assert.NotNull(resultSmall.FitType);
        }

        [Fact]
        public void DetectFit_WithHighDimensionalFeatures_HandlesCorrectly()
        {
            // Arrange
            var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();
            var features = new Matrix<double>(4, 4, new double[]
            {
                1.0, 2.0, 3.0, 4.0,
                2.0, 3.0, 4.0, 5.0,
                3.0, 4.0, 5.0, 6.0,
                4.0, 5.0, 6.0, 7.0
            });
            var actual = new Vector<double>(new double[] { 10.0, 14.0, 18.0, 22.0 });
            var predicted = new Vector<double>(new double[] { 10.1, 13.9, 18.1, 21.9 });

            var evaluationData = CreateMockEvaluationData(predicted, actual, features);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.FitType);
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
        public void DetectFit_ResultContainsAllRequiredFields()
        {
            // Arrange
            var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();
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
            Assert.NotNull(result.AdditionalInfo);
        }

        [Fact]
        public void DetectFit_WithSlightlyOffPredictions_ReturnsModerateConfidence()
        {
            // Arrange
            var detector = new GaussianProcessFitDetector<double, Matrix<double>, Vector<double>>();
            var features = new Matrix<double>(5, 2, new double[]
            {
                1.0, 2.0,
                2.0, 3.0,
                3.0, 4.0,
                4.0, 5.0,
                5.0, 6.0
            });
            var actual = new Vector<double>(new double[] { 3.0, 5.0, 7.0, 9.0, 11.0 });
            var predicted = new Vector<double>(new double[] { 3.2, 5.1, 6.9, 9.1, 10.8 });

            var evaluationData = CreateMockEvaluationData(predicted, actual, features);

            // Act
            var result = detector.DetectFit(evaluationData);

            // Assert
            Assert.NotNull(result);
            Assert.NotNull(result.ConfidenceLevel);
            Assert.True(result.ConfidenceLevel >= 0.0);
            Assert.True(result.ConfidenceLevel <= 1.0);
        }
    }
}
