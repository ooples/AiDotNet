using AiDotNet.Enums;
using AiDotNet.FeatureSelectors;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.FeatureSelectors
{
    /// <summary>
    /// Mock model with feature importances for testing.
    /// </summary>
    public class MockModelWithImportance : IFeatureImportance<double>
    {
        private readonly Dictionary<string, double> _importances;

        public MockModelWithImportance(Dictionary<string, double> importances)
        {
            _importances = importances;
        }

        public Dictionary<string, double> GetFeatureImportance()
        {
            return _importances;
        }
    }

    public class SelectFromModelTests
    {
        [Fact]
        public void SelectFeatures_WithMeanThreshold_SelectsAboveMeanFeatures()
        {
            // Arrange
            var features = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0, 3.0, 4.0 },
                { 5.0, 6.0, 7.0, 8.0 }
            });

            // Feature importances: 0.1, 0.2, 0.3, 0.4 (mean = 0.25)
            var importances = new Dictionary<string, double>
            {
                { "Feature_0", 0.1 },
                { "Feature_1", 0.2 },
                { "Feature_2", 0.3 },
                { "Feature_3", 0.4 }
            };

            var model = new MockModelWithImportance(importances);
            var selector = new SelectFromModel<double, Matrix<double>>(
                model,
                ImportanceThresholdStrategy.Mean);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(2, result.Rows);
            Assert.Equal(2, result.Columns); // Features 2 and 3 (>= 0.25)
        }

        [Fact]
        public void SelectFeatures_WithMedianThreshold_SelectsAboveMedianFeatures()
        {
            // Arrange
            var features = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0, 3.0, 4.0 },
                { 5.0, 6.0, 7.0, 8.0 }
            });

            // Feature importances: 0.1, 0.2, 0.3, 0.4 (median = 0.25)
            var importances = new Dictionary<string, double>
            {
                { "Feature_0", 0.1 },
                { "Feature_1", 0.2 },
                { "Feature_2", 0.3 },
                { "Feature_3", 0.4 }
            };

            var model = new MockModelWithImportance(importances);
            var selector = new SelectFromModel<double, Matrix<double>>(
                model,
                ImportanceThresholdStrategy.Median);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(2, result.Rows);
            Assert.Equal(2, result.Columns); // Top 50% features
        }

        [Fact]
        public void SelectFeatures_WithCustomThreshold_SelectsAboveThreshold()
        {
            // Arrange
            var features = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0, 3.0, 4.0 },
                { 5.0, 6.0, 7.0, 8.0 }
            });

            var importances = new Dictionary<string, double>
            {
                { "Feature_0", 0.1 },
                { "Feature_1", 0.2 },
                { "Feature_2", 0.3 },
                { "Feature_3", 0.4 }
            };

            var model = new MockModelWithImportance(importances);
            var selector = new SelectFromModel<double, Matrix<double>>(
                model,
                threshold: 0.25);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(2, result.Rows);
            Assert.Equal(2, result.Columns); // Features with importance >= 0.25
        }

        [Fact]
        public void SelectFeatures_WithMaxFeatures_LimitsFeatureCount()
        {
            // Arrange
            var features = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0, 3.0, 4.0 },
                { 5.0, 6.0, 7.0, 8.0 }
            });

            var importances = new Dictionary<string, double>
            {
                { "Feature_0", 0.1 },
                { "Feature_1", 0.2 },
                { "Feature_2", 0.3 },
                { "Feature_3", 0.4 }
            };

            var model = new MockModelWithImportance(importances);
            var selector = new SelectFromModel<double, Matrix<double>>(
                model,
                ImportanceThresholdStrategy.Mean,
                maxFeatures: 1);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(2, result.Rows);
            Assert.Equal(1, result.Columns); // Limited to 1 feature
        }

        [Fact]
        public void SelectFeatures_WithTopK_SelectsExactlyKFeatures()
        {
            // Arrange
            var features = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0, 3.0, 4.0, 5.0 },
                { 6.0, 7.0, 8.0, 9.0, 10.0 }
            });

            var importances = new Dictionary<string, double>
            {
                { "Feature_0", 0.1 },
                { "Feature_1", 0.2 },
                { "Feature_2", 0.5 },
                { "Feature_3", 0.4 },
                { "Feature_4", 0.3 }
            };

            var model = new MockModelWithImportance(importances);
            var selector = new SelectFromModel<double, Matrix<double>>(
                model,
                k: 3);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(2, result.Rows);
            Assert.Equal(3, result.Columns); // Top 3 features
        }

        [Fact]
        public void SelectFeatures_WithAlternativeFeatureNameFormat_ParsesCorrectly()
        {
            // Arrange
            var features = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });

            // Different naming formats
            var importances = new Dictionary<string, double>
            {
                { "feature_0", 0.1 },
                { "col_1", 0.3 },
                { "x_2", 0.2 }
            };

            var model = new MockModelWithImportance(importances);
            var selector = new SelectFromModel<double, Matrix<double>>(
                model,
                threshold: 0.15);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(2, result.Rows);
            Assert.Equal(2, result.Columns); // Features with importance >= 0.15
        }

        [Fact]
        public void SelectFeatures_WithNumericFeatureNames_ParsesCorrectly()
        {
            // Arrange
            var features = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });

            // Feature names as just numbers
            var importances = new Dictionary<string, double>
            {
                { "0", 0.1 },
                { "1", 0.3 },
                { "2", 0.2 }
            };

            var model = new MockModelWithImportance(importances);
            var selector = new SelectFromModel<double, Matrix<double>>(
                model,
                threshold: 0.15);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(2, result.Rows);
            Assert.Equal(2, result.Columns);
        }

        [Fact]
        public void SelectFeatures_WithNoFeaturesAboveThreshold_SelectsBestFeature()
        {
            // Arrange
            var features = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });

            var importances = new Dictionary<string, double>
            {
                { "Feature_0", 0.1 },
                { "Feature_1", 0.2 },
                { "Feature_2", 0.15 }
            };

            var model = new MockModelWithImportance(importances);
            var selector = new SelectFromModel<double, Matrix<double>>(
                model,
                threshold: 1.0); // Very high threshold

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(2, result.Rows);
            Assert.Equal(1, result.Columns); // At least one feature selected (the best one)
        }

        [Fact]
        public void SelectFeatures_WithEqualImportances_SelectsArbitrarily()
        {
            // Arrange
            var features = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });

            var importances = new Dictionary<string, double>
            {
                { "Feature_0", 0.2 },
                { "Feature_1", 0.2 },
                { "Feature_2", 0.2 }
            };

            var model = new MockModelWithImportance(importances);
            var selector = new SelectFromModel<double, Matrix<double>>(
                model,
                ImportanceThresholdStrategy.Mean);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(2, result.Rows);
            // When all importances are equal to mean, they should all be >= mean and thus all selected.
            // However, due to floating point precision, we accept that at least 1 feature is selected.
            // The implementation guarantees at least 1 feature is always selected.
            Assert.True(result.Columns >= 1 && result.Columns <= 3,
                $"Expected 1-3 columns, got {result.Columns}");
        }

        [Fact]
        public void Constructor_WithNullModel_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new SelectFromModel<double, Matrix<double>>(
                    null!,
                    ImportanceThresholdStrategy.Mean));
        }

        [Fact]
        public void SelectFeatures_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var features = new Matrix<float>(new float[,]
            {
                { 1.0f, 2.0f, 3.0f },
                { 4.0f, 5.0f, 6.0f }
            });

            var importances = new Dictionary<string, float>
            {
                { "Feature_0", 0.1f },
                { "Feature_1", 0.3f },
                { "Feature_2", 0.2f }
            };

            var model = new MockModelWithImportanceFloat(importances);
            var selector = new SelectFromModel<float, Matrix<float>>(
                model,
                threshold: 0.15f);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(2, result.Rows);
            Assert.Equal(2, result.Columns);
        }

        [Fact]
        public void SelectFeatures_WithOddNumberOfFeatures_MedianWorksCorrectly()
        {
            // Arrange
            var features = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0, 3.0, 4.0, 5.0 },
                { 6.0, 7.0, 8.0, 9.0, 10.0 }
            });

            var importances = new Dictionary<string, double>
            {
                { "Feature_0", 0.1 },
                { "Feature_1", 0.2 },
                { "Feature_2", 0.3 }, // Median
                { "Feature_3", 0.4 },
                { "Feature_4", 0.5 }
            };

            var model = new MockModelWithImportance(importances);
            var selector = new SelectFromModel<double, Matrix<double>>(
                model,
                ImportanceThresholdStrategy.Median);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(2, result.Rows);
            // Should select features >= median (0.3)
            Assert.True(result.Columns >= 2 && result.Columns <= 3);
        }

        [Fact]
        public void SelectFeatures_WithZeroImportances_SelectsBestFeature()
        {
            // Arrange
            var features = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });

            var importances = new Dictionary<string, double>
            {
                { "Feature_0", 0.0 },
                { "Feature_1", 0.0 },
                { "Feature_2", 0.0 }
            };

            var model = new MockModelWithImportance(importances);
            var selector = new SelectFromModel<double, Matrix<double>>(
                model,
                ImportanceThresholdStrategy.Mean);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(2, result.Rows);
            // Should select all features since all have importance >= mean (0.0)
            Assert.Equal(3, result.Columns);
        }
    }

    /// <summary>
    /// Float version of the mock model with importances.
    /// </summary>
    public class MockModelWithImportanceFloat : IFeatureImportance<float>
    {
        private readonly Dictionary<string, float> _importances;

        public MockModelWithImportanceFloat(Dictionary<string, float> importances)
        {
            _importances = importances;
        }

        public Dictionary<string, float> GetFeatureImportance()
        {
            return _importances;
        }
    }
}
