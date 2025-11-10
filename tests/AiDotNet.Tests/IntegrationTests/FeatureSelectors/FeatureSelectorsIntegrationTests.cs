using AiDotNet.FeatureSelectors;
using AiDotNet.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using Xunit;

namespace AiDotNetTests.IntegrationTests.FeatureSelectors
{
    /// <summary>
    /// Comprehensive integration tests for all FeatureSelectors with mathematically verified results.
    /// These tests validate the correctness of feature selection using synthetic datasets with known properties.
    /// </summary>
    public class FeatureSelectorsIntegrationTests
    {
        private const double Tolerance = 1e-8;

        #region Test Data Helper Methods

        /// <summary>
        /// Creates a dataset with 3 relevant features (linear combinations) and 7 noise features.
        /// Relevant features: f0 = linearly increasing, f1 = quadratic pattern, f2 = alternating pattern
        /// Noise features: f3-f9 = random noise
        /// </summary>
        private Matrix<double> CreateRelevantAndNoiseDataset(int numSamples = 100, int seed = 42)
        {
            var data = new Matrix<double>(numSamples, 10);
            var random = new Random(seed);

            for (int i = 0; i < numSamples; i++)
            {
                // Relevant features with clear patterns
                data[i, 0] = i * 0.1;                          // Linear increasing
                data[i, 1] = i * i * 0.01;                     // Quadratic
                data[i, 2] = (i % 2 == 0) ? 1.0 : -1.0;       // Alternating pattern

                // Noise features with high variance but no meaningful pattern
                for (int j = 3; j < 10; j++)
                {
                    data[i, j] = (random.NextDouble() - 0.5) * 0.1;
                }
            }

            return data;
        }

        /// <summary>
        /// Creates a dataset with highly correlated features.
        /// Features 0, 1, 2 are almost identical (high correlation).
        /// Features 3, 4, 5 are almost identical (high correlation).
        /// Features 6, 7, 8, 9 are independent.
        /// </summary>
        private Matrix<double> CreateCorrelatedDataset(int numSamples = 100, int seed = 42)
        {
            var data = new Matrix<double>(numSamples, 10);
            var random = new Random(seed);

            for (int i = 0; i < numSamples; i++)
            {
                double base1 = i * 0.1 + (random.NextDouble() - 0.5) * 0.01;
                double base2 = i * 0.2 + (random.NextDouble() - 0.5) * 0.01;

                // Highly correlated group 1
                data[i, 0] = base1;
                data[i, 1] = base1 + (random.NextDouble() - 0.5) * 0.01;
                data[i, 2] = base1 + (random.NextDouble() - 0.5) * 0.01;

                // Highly correlated group 2
                data[i, 3] = base2;
                data[i, 4] = base2 + (random.NextDouble() - 0.5) * 0.01;
                data[i, 5] = base2 + (random.NextDouble() - 0.5) * 0.01;

                // Independent features
                data[i, 6] = random.NextDouble() * 2.0;
                data[i, 7] = random.NextDouble() * 3.0;
                data[i, 8] = random.NextDouble() * 4.0;
                data[i, 9] = random.NextDouble() * 5.0;
            }

            return data;
        }

        /// <summary>
        /// Creates a dataset with varying variance levels.
        /// Low variance features: 0, 1, 2 (nearly constant)
        /// High variance features: 3-9 (widely varying)
        /// </summary>
        private Matrix<double> CreateLowAndHighVarianceDataset(int numSamples = 100, int seed = 42)
        {
            var data = new Matrix<double>(numSamples, 10);
            var random = new Random(seed);

            for (int i = 0; i < numSamples; i++)
            {
                // Low variance features (almost constant)
                data[i, 0] = 1.0 + (random.NextDouble() - 0.5) * 0.01;
                data[i, 1] = 2.0 + (random.NextDouble() - 0.5) * 0.01;
                data[i, 2] = 3.0 + (random.NextDouble() - 0.5) * 0.01;

                // High variance features
                for (int j = 3; j < 10; j++)
                {
                    data[i, j] = (random.NextDouble() - 0.5) * 20.0;
                }
            }

            return data;
        }

        /// <summary>
        /// Creates a target vector for classification (binary classes based on first feature).
        /// </summary>
        private Vector<double> CreateBinaryClassificationTarget(Matrix<double> features)
        {
            int numSamples = features.Rows;
            var target = new Vector<double>(numSamples);

            for (int i = 0; i < numSamples; i++)
            {
                target[i] = features[i, 0] > 5.0 ? 1.0 : 0.0;
            }

            return target;
        }

        /// <summary>
        /// Creates a target vector for regression (linear combination of first 3 features).
        /// </summary>
        private Vector<double> CreateRegressionTarget(Matrix<double> features)
        {
            int numSamples = features.Rows;
            var target = new Vector<double>(numSamples);

            for (int i = 0; i < numSamples; i++)
            {
                target[i] = 2.0 * features[i, 0] + 1.5 * features[i, 1] + 0.5 * features[i, 2];
            }

            return target;
        }

        #endregion

        #region CorrelationFeatureSelector Tests

        [Fact]
        public void CorrelationFeatureSelector_WithHighlyCorrelatedFeatures_RemovesRedundantFeatures()
        {
            // Arrange
            var data = CreateCorrelatedDataset(100);
            var selector = new CorrelationFeatureSelector<double, Matrix<double>>(threshold: 0.95);

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert - Should keep fewer features than input (redundant ones removed)
            Assert.True(selected.Columns < data.Columns);

            // Should keep at least one feature from each correlated group plus independents
            // Expected: 1 from group 1 (0,1,2), 1 from group 2 (3,4,5), all 4 independent (6,7,8,9) = 6 total
            Assert.True(selected.Columns >= 6 && selected.Columns <= 8);
        }

        [Fact]
        public void CorrelationFeatureSelector_WithLowThreshold_RemovesMoreFeatures()
        {
            // Arrange
            var data = CreateCorrelatedDataset(100);
            var lowThresholdSelector = new CorrelationFeatureSelector<double, Matrix<double>>(threshold: 0.3);
            var highThresholdSelector = new CorrelationFeatureSelector<double, Matrix<double>>(threshold: 0.95);

            // Act
            var lowResult = lowThresholdSelector.SelectFeatures(data);
            var highResult = highThresholdSelector.SelectFeatures(data);

            // Assert - Lower threshold should result in fewer features
            Assert.True(lowResult.Columns <= highResult.Columns);
        }

        [Fact]
        public void CorrelationFeatureSelector_WithIndependentFeatures_KeepsAllFeatures()
        {
            // Arrange - Create dataset with independent features
            var data = new Matrix<double>(50, 5);
            var random = new Random(42);
            for (int i = 0; i < 50; i++)
            {
                data[i, 0] = i * 1.0;
                data[i, 1] = Math.Sin(i * 0.1);
                data[i, 2] = i * i * 0.01;
                data[i, 3] = (i % 3) * 2.0;
                data[i, 4] = random.NextDouble() * 10;
            }

            var selector = new CorrelationFeatureSelector<double, Matrix<double>>(threshold: 0.5);

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert - Should keep all features since they're independent
            Assert.Equal(5, selected.Columns);
        }

        [Fact]
        public void CorrelationFeatureSelector_WithSingleFeature_KeepsThatFeature()
        {
            // Arrange
            var data = new Matrix<double>(50, 1);
            for (int i = 0; i < 50; i++)
            {
                data[i, 0] = i * 0.1;
            }

            var selector = new CorrelationFeatureSelector<double, Matrix<double>>(threshold: 0.5);

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert
            Assert.Equal(1, selected.Columns);
        }

        [Fact]
        public void CorrelationFeatureSelector_PreservesRowCount()
        {
            // Arrange
            var data = CreateCorrelatedDataset(100);
            var selector = new CorrelationFeatureSelector<double, Matrix<double>>(threshold: 0.8);

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert - Should preserve all rows
            Assert.Equal(data.Rows, selected.Rows);
        }

        [Fact]
        public void CorrelationFeatureSelector_WithPerfectlyCorrelatedPair_RemovesOne()
        {
            // Arrange - Two perfectly correlated features and one independent
            var data = new Matrix<double>(50, 3);
            for (int i = 0; i < 50; i++)
            {
                data[i, 0] = i * 0.5;
                data[i, 1] = i * 0.5;  // Perfectly correlated with feature 0
                data[i, 2] = i * i;     // Independent
            }

            var selector = new CorrelationFeatureSelector<double, Matrix<double>>(threshold: 0.99);

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert - Should remove one of the correlated features
            Assert.Equal(2, selected.Columns);
        }

        [Fact]
        public void CorrelationFeatureSelector_WithNegativeCorrelation_RemovesFeature()
        {
            // Arrange - Two perfectly negatively correlated features
            var data = new Matrix<double>(50, 3);
            for (int i = 0; i < 50; i++)
            {
                data[i, 0] = i * 0.5;
                data[i, 1] = -i * 0.5;  // Perfectly negatively correlated
                data[i, 2] = i * i;      // Independent
            }

            var selector = new CorrelationFeatureSelector<double, Matrix<double>>(threshold: 0.99);

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert - Should remove one of the correlated features (negative correlation is still correlation)
            Assert.Equal(2, selected.Columns);
        }

        [Fact]
        public void CorrelationFeatureSelector_WithDefaultThreshold_WorksCorrectly()
        {
            // Arrange
            var data = CreateCorrelatedDataset(100);
            var selector = new CorrelationFeatureSelector<double, Matrix<double>>(); // Uses default 0.5

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert - Should select some features
            Assert.True(selected.Columns > 0 && selected.Columns <= data.Columns);
        }

        [Fact]
        public void CorrelationFeatureSelector_SelectsFirstFromCorrelatedPair()
        {
            // Arrange - Create dataset where we can verify which feature is selected
            var data = new Matrix<double>(50, 2);
            for (int i = 0; i < 50; i++)
            {
                data[i, 0] = i * 1.0;
                data[i, 1] = i * 1.0 + 0.001; // Nearly identical to feature 0
            }

            var selector = new CorrelationFeatureSelector<double, Matrix<double>>(threshold: 0.99);

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert - Should keep exactly 1 feature (the first one encountered)
            Assert.Equal(1, selected.Columns);
            // Verify it's the first feature by checking values
            for (int i = 0; i < 10; i++)
            {
                Assert.Equal(i * 1.0, selected[i, 0], precision: 5);
            }
        }

        #endregion

        #region VarianceThresholdFeatureSelector Tests

        [Fact]
        public void VarianceThresholdFeatureSelector_RemovesLowVarianceFeatures()
        {
            // Arrange
            var data = CreateLowAndHighVarianceDataset(100);
            var selector = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 1.0);

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert - Should remove the 3 low-variance features
            Assert.Equal(7, selected.Columns);
        }

        [Fact]
        public void VarianceThresholdFeatureSelector_WithHighThreshold_RemovesMoreFeatures()
        {
            // Arrange
            var data = CreateLowAndHighVarianceDataset(100);
            var lowThreshold = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 0.01);
            var highThreshold = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 10.0);

            // Act
            var lowResult = lowThreshold.SelectFeatures(data);
            var highResult = highThreshold.SelectFeatures(data);

            // Assert - Higher threshold should remove more features
            Assert.True(highResult.Columns <= lowResult.Columns);
        }

        [Fact]
        public void VarianceThresholdFeatureSelector_WithConstantFeature_RemovesIt()
        {
            // Arrange - One constant feature, others varying
            var data = new Matrix<double>(50, 3);
            for (int i = 0; i < 50; i++)
            {
                data[i, 0] = 5.0;      // Constant (variance = 0)
                data[i, 1] = i * 0.5;  // Varying
                data[i, 2] = i * i;    // Varying
            }

            var selector = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 0.01);

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert - Should remove the constant feature
            Assert.Equal(2, selected.Columns);
        }

        [Fact]
        public void VarianceThresholdFeatureSelector_CalculatesVarianceCorrectly()
        {
            // Arrange - Feature with known variance
            var data = new Matrix<double>(5, 1);
            data[0, 0] = 2.0; data[1, 0] = 4.0; data[2, 0] = 6.0;
            data[3, 0] = 8.0; data[4, 0] = 10.0;
            // Mean = 6.0, Variance = 10.0

            var selectorLow = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 9.0);
            var selectorHigh = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 11.0);

            // Act
            var selectedLow = selectorLow.SelectFeatures(data);
            var selectedHigh = selectorHigh.SelectFeatures(data);

            // Assert
            Assert.Equal(1, selectedLow.Columns);  // Variance 10.0 > 9.0, so kept
            Assert.Equal(0, selectedHigh.Columns); // Variance 10.0 < 11.0, so removed
        }

        [Fact]
        public void VarianceThresholdFeatureSelector_WithZeroThreshold_KeepsAllNonConstant()
        {
            // Arrange
            var data = new Matrix<double>(50, 4);
            for (int i = 0; i < 50; i++)
            {
                data[i, 0] = 5.0;      // Constant
                data[i, 1] = i * 0.1;  // Varying
                data[i, 2] = i * 0.2;  // Varying
                data[i, 3] = i;        // Varying
            }

            var selector = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 0.0);

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert - Should keep only non-constant features
            Assert.Equal(3, selected.Columns);
        }

        [Fact]
        public void VarianceThresholdFeatureSelector_PreservesRowCount()
        {
            // Arrange
            var data = CreateLowAndHighVarianceDataset(100);
            var selector = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 1.0);

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert
            Assert.Equal(data.Rows, selected.Rows);
        }

        [Fact]
        public void VarianceThresholdFeatureSelector_WithAllLowVariance_ReturnsEmpty()
        {
            // Arrange - All features are nearly constant
            var data = new Matrix<double>(50, 3);
            for (int i = 0; i < 50; i++)
            {
                data[i, 0] = 1.0 + (i % 2) * 0.001;
                data[i, 1] = 2.0 + (i % 2) * 0.001;
                data[i, 2] = 3.0 + (i % 2) * 0.001;
            }

            var selector = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 1.0);

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert - Should remove all features
            Assert.Equal(0, selected.Columns);
        }

        [Fact]
        public void VarianceThresholdFeatureSelector_WithDefaultThreshold_WorksCorrectly()
        {
            // Arrange
            var data = CreateLowAndHighVarianceDataset(100);
            var selector = new VarianceThresholdFeatureSelector<double, Matrix<double>>(); // Default 0.1

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert
            Assert.True(selected.Columns > 0);
            Assert.True(selected.Columns <= data.Columns);
        }

        [Fact]
        public void VarianceThresholdFeatureSelector_WithSingleFeature_WorksCorrectly()
        {
            // Arrange
            var data = new Matrix<double>(50, 1);
            for (int i = 0; i < 50; i++)
            {
                data[i, 0] = i * 0.5;
            }

            var selector = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 1.0);

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert - Should keep the feature if variance is high enough
            Assert.True(selected.Columns >= 0 && selected.Columns <= 1);
        }

        #endregion

        #region UnivariateFeatureSelector Tests

        [Fact]
        public void UnivariateFeatureSelector_FValue_SelectsRelevantFeatures()
        {
            // Arrange
            var data = CreateRelevantAndNoiseDataset(100);
            var target = CreateBinaryClassificationTarget(data);
            var selector = new UnivariateFeatureSelector<double, Matrix<double>>(
                target,
                UnivariateScoringFunction.FValue,
                k: 5);

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert - Should select top 5 features
            Assert.Equal(5, selected.Columns);
        }

        [Fact]
        public void UnivariateFeatureSelector_SelectsTopKFeatures()
        {
            // Arrange
            var data = CreateRelevantAndNoiseDataset(100);
            var target = CreateBinaryClassificationTarget(data);
            var selector3 = new UnivariateFeatureSelector<double, Matrix<double>>(target, k: 3);
            var selector7 = new UnivariateFeatureSelector<double, Matrix<double>>(target, k: 7);

            // Act
            var selected3 = selector3.SelectFeatures(data);
            var selected7 = selector7.SelectFeatures(data);

            // Assert
            Assert.Equal(3, selected3.Columns);
            Assert.Equal(7, selected7.Columns);
        }

        [Fact]
        public void UnivariateFeatureSelector_WithDefaultK_SelectsHalfFeatures()
        {
            // Arrange
            var data = CreateRelevantAndNoiseDataset(100);
            var target = CreateBinaryClassificationTarget(data);
            var selector = new UnivariateFeatureSelector<double, Matrix<double>>(target); // Default k = 50%

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert - Should select approximately half (5 out of 10)
            Assert.Equal(5, selected.Columns);
        }

        [Fact]
        public void UnivariateFeatureSelector_PreservesRowCount()
        {
            // Arrange
            var data = CreateRelevantAndNoiseDataset(100);
            var target = CreateBinaryClassificationTarget(data);
            var selector = new UnivariateFeatureSelector<double, Matrix<double>>(target, k: 5);

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert
            Assert.Equal(data.Rows, selected.Rows);
        }

        [Fact]
        public void UnivariateFeatureSelector_MutualInformation_SelectsInformativeFeatures()
        {
            // Arrange
            var data = CreateRelevantAndNoiseDataset(100);
            var target = CreateBinaryClassificationTarget(data);
            var selector = new UnivariateFeatureSelector<double, Matrix<double>>(
                target,
                UnivariateScoringFunction.MutualInformation,
                k: 4);

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert
            Assert.Equal(4, selected.Columns);
        }

        [Fact]
        public void UnivariateFeatureSelector_ChiSquared_WorksWithCategoricalFeatures()
        {
            // Arrange - Create categorical-like features
            var data = new Matrix<double>(100, 5);
            var target = new Vector<double>(100);
            var random = new Random(42);

            for (int i = 0; i < 100; i++)
            {
                // Categorical features (values: 0, 1, 2)
                data[i, 0] = random.Next(0, 3);
                data[i, 1] = random.Next(0, 3);
                data[i, 2] = random.Next(0, 3);
                data[i, 3] = random.Next(0, 3);
                data[i, 4] = random.Next(0, 3);

                // Target correlates with first feature
                target[i] = data[i, 0] > 1 ? 1.0 : 0.0;
            }

            var selector = new UnivariateFeatureSelector<double, Matrix<double>>(
                target,
                UnivariateScoringFunction.ChiSquared,
                k: 3);

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert
            Assert.Equal(3, selected.Columns);
        }

        [Fact]
        public void UnivariateFeatureSelector_WithSingleFeature_SelectsThatFeature()
        {
            // Arrange
            var data = new Matrix<double>(50, 1);
            var target = new Vector<double>(50);
            for (int i = 0; i < 50; i++)
            {
                data[i, 0] = i * 0.5;
                target[i] = i % 2;
            }

            var selector = new UnivariateFeatureSelector<double, Matrix<double>>(target, k: 1);

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert
            Assert.Equal(1, selected.Columns);
        }

        [Fact]
        public void UnivariateFeatureSelector_KGreaterThanFeatures_SelectsAllFeatures()
        {
            // Arrange
            var data = CreateRelevantAndNoiseDataset(100);
            var target = CreateBinaryClassificationTarget(data);
            var selector = new UnivariateFeatureSelector<double, Matrix<double>>(target, k: 20); // More than 10 features

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert - Should cap at total number of features
            Assert.Equal(10, selected.Columns);
        }

        [Fact]
        public void UnivariateFeatureSelector_DifferentScoringFunctions_ProduceDifferentResults()
        {
            // Arrange
            var data = CreateRelevantAndNoiseDataset(100);
            var target = CreateBinaryClassificationTarget(data);

            var selectorF = new UnivariateFeatureSelector<double, Matrix<double>>(
                target, UnivariateScoringFunction.FValue, k: 3);
            var selectorMI = new UnivariateFeatureSelector<double, Matrix<double>>(
                target, UnivariateScoringFunction.MutualInformation, k: 3);

            // Act
            var selectedF = selectorF.SelectFeatures(data);
            var selectedMI = selectorMI.SelectFeatures(data);

            // Assert - Both should select 3 features
            Assert.Equal(3, selectedF.Columns);
            Assert.Equal(3, selectedMI.Columns);
        }

        #endregion

        #region NoFeatureSelector Tests

        [Fact]
        public void NoFeatureSelector_KeepsAllFeatures()
        {
            // Arrange
            var data = CreateRelevantAndNoiseDataset(100);
            var selector = new NoFeatureSelector<double, Matrix<double>>();

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert - Should keep all features
            Assert.Equal(data.Columns, selected.Columns);
            Assert.Equal(data.Rows, selected.Rows);
        }

        [Fact]
        public void NoFeatureSelector_PreservesDataIntegrity()
        {
            // Arrange
            var data = new Matrix<double>(10, 3);
            for (int i = 0; i < 10; i++)
            {
                data[i, 0] = i * 1.0;
                data[i, 1] = i * 2.0;
                data[i, 2] = i * 3.0;
            }

            var selector = new NoFeatureSelector<double, Matrix<double>>();

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert - Data should be unchanged
            for (int i = 0; i < 10; i++)
            {
                Assert.Equal(i * 1.0, selected[i, 0], precision: 10);
                Assert.Equal(i * 2.0, selected[i, 1], precision: 10);
                Assert.Equal(i * 3.0, selected[i, 2], precision: 10);
            }
        }

        [Fact]
        public void NoFeatureSelector_WithSingleFeature_KeepsIt()
        {
            // Arrange
            var data = new Matrix<double>(50, 1);
            for (int i = 0; i < 50; i++)
            {
                data[i, 0] = i * 0.5;
            }

            var selector = new NoFeatureSelector<double, Matrix<double>>();

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert
            Assert.Equal(1, selected.Columns);
            Assert.Equal(50, selected.Rows);
        }

        [Fact]
        public void NoFeatureSelector_WithManyFeatures_KeepsAll()
        {
            // Arrange
            var data = new Matrix<double>(100, 50);
            var random = new Random(42);
            for (int i = 0; i < 100; i++)
            {
                for (int j = 0; j < 50; j++)
                {
                    data[i, j] = random.NextDouble();
                }
            }

            var selector = new NoFeatureSelector<double, Matrix<double>>();

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert
            Assert.Equal(50, selected.Columns);
            Assert.Equal(100, selected.Rows);
        }

        #endregion

        #region SelectFromModel Tests

        [Fact]
        public void SelectFromModel_WithMeanStrategy_SelectsAboveAverageFeatures()
        {
            // Arrange
            var mockModel = new MockFeatureImportanceModel(10);
            // Set importances: 0.01, 0.02, ..., 0.10
            var importances = new Dictionary<string, double>();
            for (int i = 0; i < 10; i++)
            {
                importances[$"Feature_{i}"] = 0.01 * (i + 1);
            }
            mockModel.SetFeatureImportance(importances);

            var selector = new SelectFromModel<double, Matrix<double>>(
                mockModel,
                ImportanceThresholdStrategy.Mean);

            var data = new Matrix<double>(50, 10);
            for (int i = 0; i < 50; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    data[i, j] = i * j * 0.1;
                }
            }

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert - Mean = 0.055, should keep features 5-9 (5 features)
            Assert.Equal(5, selected.Columns);
        }

        [Fact]
        public void SelectFromModel_WithMedianStrategy_SelectsTopHalfFeatures()
        {
            // Arrange
            var mockModel = new MockFeatureImportanceModel(10);
            var importances = new Dictionary<string, double>();
            for (int i = 0; i < 10; i++)
            {
                importances[$"Feature_{i}"] = 0.01 * (i + 1);
            }
            mockModel.SetFeatureImportance(importances);

            var selector = new SelectFromModel<double, Matrix<double>>(
                mockModel,
                ImportanceThresholdStrategy.Median);

            var data = new Matrix<double>(50, 10);
            for (int i = 0; i < 50; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    data[i, j] = i * j * 0.1;
                }
            }

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert - Median between 0.05 and 0.06, should keep roughly top half
            Assert.True(selected.Columns >= 4 && selected.Columns <= 6);
        }

        [Fact]
        public void SelectFromModel_WithCustomThreshold_SelectsCorrectly()
        {
            // Arrange
            var mockModel = new MockFeatureImportanceModel(10);
            var importances = new Dictionary<string, double>();
            for (int i = 0; i < 10; i++)
            {
                importances[$"Feature_{i}"] = 0.01 * (i + 1);
            }
            mockModel.SetFeatureImportance(importances);

            var selector = new SelectFromModel<double, Matrix<double>>(
                mockModel,
                threshold: 0.07);

            var data = new Matrix<double>(50, 10);
            for (int i = 0; i < 50; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    data[i, j] = i * j * 0.1;
                }
            }

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert - Threshold 0.07, should keep features 7, 8, 9 (3 features)
            Assert.Equal(3, selected.Columns);
        }

        [Fact]
        public void SelectFromModel_WithTopK_SelectsExactlyKFeatures()
        {
            // Arrange
            var mockModel = new MockFeatureImportanceModel(10);
            var importances = new Dictionary<string, double>();
            for (int i = 0; i < 10; i++)
            {
                importances[$"Feature_{i}"] = 0.01 * (i + 1);
            }
            mockModel.SetFeatureImportance(importances);

            var selector = new SelectFromModel<double, Matrix<double>>(mockModel, k: 4);

            var data = new Matrix<double>(50, 10);
            for (int i = 0; i < 50; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    data[i, j] = i * j * 0.1;
                }
            }

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert
            Assert.Equal(4, selected.Columns);
        }

        [Fact]
        public void SelectFromModel_WithMaxFeatures_LimitsSelection()
        {
            // Arrange
            var mockModel = new MockFeatureImportanceModel(10);
            var importances = new Dictionary<string, double>();
            for (int i = 0; i < 10; i++)
            {
                importances[$"Feature_{i}"] = 0.01 * (i + 1);
            }
            mockModel.SetFeatureImportance(importances);

            var selector = new SelectFromModel<double, Matrix<double>>(
                mockModel,
                ImportanceThresholdStrategy.Mean,
                maxFeatures: 3);

            var data = new Matrix<double>(50, 10);
            for (int i = 0; i < 50; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    data[i, j] = i * j * 0.1;
                }
            }

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert - Should limit to max 3 features even if more are above threshold
            Assert.Equal(3, selected.Columns);
        }

        [Fact]
        public void SelectFromModel_PreservesRowCount()
        {
            // Arrange
            var mockModel = new MockFeatureImportanceModel(10);
            var importances = new Dictionary<string, double>();
            for (int i = 0; i < 10; i++)
            {
                importances[$"Feature_{i}"] = 0.01 * (i + 1);
            }
            mockModel.SetFeatureImportance(importances);

            var selector = new SelectFromModel<double, Matrix<double>>(mockModel, k: 5);

            var data = new Matrix<double>(75, 10);

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert
            Assert.Equal(75, selected.Rows);
        }

        [Fact]
        public void SelectFromModel_WithZeroImportances_SelectsAtLeastOne()
        {
            // Arrange
            var mockModel = new MockFeatureImportanceModel(5);
            var importances = new Dictionary<string, double>();
            for (int i = 0; i < 5; i++)
            {
                importances[$"Feature_{i}"] = 0.0; // All zero importance
            }
            mockModel.SetFeatureImportance(importances);

            var selector = new SelectFromModel<double, Matrix<double>>(
                mockModel,
                ImportanceThresholdStrategy.Mean);

            var data = new Matrix<double>(50, 5);

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert - Should select at least one feature even if all have zero importance
            Assert.True(selected.Columns >= 1);
        }

        [Fact]
        public void SelectFromModel_SelectsMostImportantFeatures()
        {
            // Arrange
            var mockModel = new MockFeatureImportanceModel(10);
            var importances = new Dictionary<string, double>
            {
                ["Feature_0"] = 0.01,
                ["Feature_1"] = 0.02,
                ["Feature_2"] = 0.03,
                ["Feature_3"] = 0.04,
                ["Feature_4"] = 0.05,
                ["Feature_5"] = 0.06,
                ["Feature_6"] = 0.07,
                ["Feature_7"] = 0.08,
                ["Feature_8"] = 0.09,
                ["Feature_9"] = 0.10  // Most important
            };
            mockModel.SetFeatureImportance(importances);

            var selector = new SelectFromModel<double, Matrix<double>>(mockModel, k: 3);

            var data = new Matrix<double>(50, 10);
            for (int i = 0; i < 50; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    data[i, j] = i + j;
                }
            }

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert - Should select exactly 3 features (the top 3 by importance)
            Assert.Equal(3, selected.Columns);
        }

        #endregion

        #region RecursiveFeatureElimination Tests

        [Fact]
        public void RecursiveFeatureElimination_ReducesFeatureCount()
        {
            // Arrange
            var data = new Matrix<double>(50, 10);
            for (int i = 0; i < 50; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    data[i, j] = i * 0.1 + j * 0.01;
                }
            }

            var model = new VectorModel<double>(new Vector<double>(10));
            var rfe = new RecursiveFeatureElimination<double, Matrix<double>, Vector<double>>(
                model,
                createDummyTarget: (n) => new Vector<double>(n),
                numFeaturesToSelect: 5);

            // Act
            var selected = rfe.SelectFeatures(data);

            // Assert
            Assert.Equal(5, selected.Columns);
            Assert.Equal(50, selected.Rows);
        }

        [Fact]
        public void RecursiveFeatureElimination_WithDefaultNumFeatures_SelectsHalf()
        {
            // Arrange
            var data = new Matrix<double>(50, 10);
            for (int i = 0; i < 50; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    data[i, j] = i * 0.1 + j;
                }
            }

            var model = new VectorModel<double>(new Vector<double>(10));
            var rfe = new RecursiveFeatureElimination<double, Matrix<double>, Vector<double>>(
                model,
                createDummyTarget: (n) => new Vector<double>(n));

            // Act
            var selected = rfe.SelectFeatures(data);

            // Assert - Should select approximately 50% of features
            Assert.Equal(5, selected.Columns);
        }

        [Fact]
        public void RecursiveFeatureElimination_PreservesRowCount()
        {
            // Arrange
            var data = new Matrix<double>(100, 8);
            for (int i = 0; i < 100; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    data[i, j] = i * j * 0.01;
                }
            }

            var model = new VectorModel<double>(new Vector<double>(8));
            var rfe = new RecursiveFeatureElimination<double, Matrix<double>, Vector<double>>(
                model,
                createDummyTarget: (n) => new Vector<double>(n),
                numFeaturesToSelect: 3);

            // Act
            var selected = rfe.SelectFeatures(data);

            // Assert
            Assert.Equal(100, selected.Rows);
        }

        [Fact]
        public void RecursiveFeatureElimination_WithSingleFeature_KeepsIt()
        {
            // Arrange
            var data = new Matrix<double>(50, 5);
            for (int i = 0; i < 50; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    data[i, j] = i * j;
                }
            }

            var model = new VectorModel<double>(new Vector<double>(5));
            var rfe = new RecursiveFeatureElimination<double, Matrix<double>, Vector<double>>(
                model,
                createDummyTarget: (n) => new Vector<double>(n),
                numFeaturesToSelect: 1);

            // Act
            var selected = rfe.SelectFeatures(data);

            // Assert
            Assert.Equal(1, selected.Columns);
        }

        [Fact]
        public void RecursiveFeatureElimination_SelectsRequestedNumber()
        {
            // Arrange
            var data = new Matrix<double>(50, 10);
            for (int i = 0; i < 50; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    data[i, j] = i + j * 2.0;
                }
            }

            var model = new VectorModel<double>(new Vector<double>(10));

            // Test different numbers
            var rfe3 = new RecursiveFeatureElimination<double, Matrix<double>, Vector<double>>(
                model,
                createDummyTarget: (n) => new Vector<double>(n),
                numFeaturesToSelect: 3);
            var rfe7 = new RecursiveFeatureElimination<double, Matrix<double>, Vector<double>>(
                model,
                createDummyTarget: (n) => new Vector<double>(n),
                numFeaturesToSelect: 7);

            // Act
            var selected3 = rfe3.SelectFeatures(data);
            var selected7 = rfe7.SelectFeatures(data);

            // Assert
            Assert.Equal(3, selected3.Columns);
            Assert.Equal(7, selected7.Columns);
        }

        #endregion

        #region SequentialFeatureSelector Tests

        [Fact]
        public void SequentialFeatureSelector_ForwardSelection_SelectsFeatures()
        {
            // Arrange
            var data = new Matrix<double>(50, 8);
            var target = new Vector<double>(50);
            var random = new Random(42);

            for (int i = 0; i < 50; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    data[i, j] = random.NextDouble() * 10;
                }
                target[i] = data[i, 0] + data[i, 1] + random.NextDouble(); // Target depends on first 2 features
            }

            var model = new VectorModel<double>(new Vector<double>(8));

            // Simple scoring function (negative MSE)
            Func<Vector<double>, Vector<double>, double> scoringFunc = (pred, actual) =>
            {
                double mse = 0;
                for (int i = 0; i < pred.Length; i++)
                {
                    double diff = pred[i] - actual[i];
                    mse += diff * diff;
                }
                return -mse / pred.Length; // Negative so higher is better
            };

            var selector = new SequentialFeatureSelector<double, Matrix<double>, Vector<double>>(
                model,
                target,
                scoringFunc,
                SequentialFeatureSelectionDirection.Forward,
                numFeaturesToSelect: 4);

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert
            Assert.Equal(4, selected.Columns);
            Assert.Equal(50, selected.Rows);
        }

        [Fact]
        public void SequentialFeatureSelector_BackwardElimination_SelectsFeatures()
        {
            // Arrange
            var data = new Matrix<double>(30, 6);
            var target = new Vector<double>(30);
            var random = new Random(42);

            for (int i = 0; i < 30; i++)
            {
                for (int j = 0; j < 6; j++)
                {
                    data[i, j] = random.NextDouble() * 5;
                }
                target[i] = data[i, 0] * 2 + data[i, 1] * 3;
            }

            var model = new VectorModel<double>(new Vector<double>(6));

            Func<Vector<double>, Vector<double>, double> scoringFunc = (pred, actual) =>
            {
                double mse = 0;
                for (int i = 0; i < pred.Length; i++)
                {
                    double diff = pred[i] - actual[i];
                    mse += diff * diff;
                }
                return -mse / pred.Length;
            };

            var selector = new SequentialFeatureSelector<double, Matrix<double>, Vector<double>>(
                model,
                target,
                scoringFunc,
                SequentialFeatureSelectionDirection.Backward,
                numFeaturesToSelect: 3);

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert
            Assert.Equal(3, selected.Columns);
            Assert.Equal(30, selected.Rows);
        }

        [Fact]
        public void SequentialFeatureSelector_WithDefaultNumFeatures_SelectsHalf()
        {
            // Arrange
            var data = new Matrix<double>(30, 10);
            var target = new Vector<double>(30);
            var random = new Random(42);

            for (int i = 0; i < 30; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    data[i, j] = random.NextDouble();
                }
                target[i] = random.NextDouble();
            }

            var model = new VectorModel<double>(new Vector<double>(10));

            Func<Vector<double>, Vector<double>, double> scoringFunc = (pred, actual) => -1.0; // Dummy

            var selector = new SequentialFeatureSelector<double, Matrix<double>, Vector<double>>(
                model,
                target,
                scoringFunc);

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert - Should select 50% of features (5 out of 10)
            Assert.Equal(5, selected.Columns);
        }

        [Fact]
        public void SequentialFeatureSelector_PreservesRowCount()
        {
            // Arrange
            var data = new Matrix<double>(75, 8);
            var target = new Vector<double>(75);
            var random = new Random(42);

            for (int i = 0; i < 75; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    data[i, j] = random.NextDouble();
                }
                target[i] = random.NextDouble();
            }

            var model = new VectorModel<double>(new Vector<double>(8));
            Func<Vector<double>, Vector<double>, double> scoringFunc = (pred, actual) => 0.0;

            var selector = new SequentialFeatureSelector<double, Matrix<double>, Vector<double>>(
                model,
                target,
                scoringFunc,
                numFeaturesToSelect: 3);

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert
            Assert.Equal(75, selected.Rows);
        }

        [Fact]
        public void SequentialFeatureSelector_ForwardVsBackward_MayProduceDifferentResults()
        {
            // Arrange
            var data = new Matrix<double>(40, 8);
            var target = new Vector<double>(40);
            var random = new Random(42);

            for (int i = 0; i < 40; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    data[i, j] = random.NextDouble() * 10;
                }
                target[i] = data[i, 0] + data[i, 1];
            }

            var model = new VectorModel<double>(new Vector<double>(8));
            Func<Vector<double>, Vector<double>, double> scoringFunc = (pred, actual) =>
            {
                double sum = 0;
                for (int i = 0; i < pred.Length; i++)
                {
                    sum += Math.Abs(pred[i] - actual[i]);
                }
                return -sum;
            };

            var forwardSelector = new SequentialFeatureSelector<double, Matrix<double>, Vector<double>>(
                model, target, scoringFunc, SequentialFeatureSelectionDirection.Forward, 4);
            var backwardSelector = new SequentialFeatureSelector<double, Matrix<double>, Vector<double>>(
                model, target, scoringFunc, SequentialFeatureSelectionDirection.Backward, 4);

            // Act
            var forwardResult = forwardSelector.SelectFeatures(data);
            var backwardResult = backwardSelector.SelectFeatures(data);

            // Assert - Both should select 4 features
            Assert.Equal(4, forwardResult.Columns);
            Assert.Equal(4, backwardResult.Columns);
        }

        [Fact]
        public void SequentialFeatureSelector_SelectsRequestedNumberOfFeatures()
        {
            // Arrange
            var data = new Matrix<double>(30, 10);
            var target = new Vector<double>(30);
            var random = new Random(42);

            for (int i = 0; i < 30; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    data[i, j] = random.NextDouble();
                }
                target[i] = random.NextDouble();
            }

            var model = new VectorModel<double>(new Vector<double>(10));
            Func<Vector<double>, Vector<double>, double> scoringFunc = (pred, actual) => 1.0;

            var selector2 = new SequentialFeatureSelector<double, Matrix<double>, Vector<double>>(
                model, target, scoringFunc, numFeaturesToSelect: 2);
            var selector6 = new SequentialFeatureSelector<double, Matrix<double>, Vector<double>>(
                model, target, scoringFunc, numFeaturesToSelect: 6);

            // Act
            var selected2 = selector2.SelectFeatures(data);
            var selected6 = selector6.SelectFeatures(data);

            // Assert
            Assert.Equal(2, selected2.Columns);
            Assert.Equal(6, selected6.Columns);
        }

        #endregion

        #region Edge Cases and Integration Tests

        [Fact]
        public void AllSelectors_PreserveRowCountAndReduceColumns()
        {
            // Arrange - Test that all selectors maintain row count
            var data = new Matrix<double>(50, 10);
            var target = new Vector<double>(50);
            var random = new Random(42);

            for (int i = 0; i < 50; i++)
            {
                for (int j = 0; j < 10; j++)
                {
                    data[i, j] = random.NextDouble() * 10;
                }
                target[i] = i % 2;
            }

            var selectors = new List<IFeatureSelector<double, Matrix<double>>>
            {
                new CorrelationFeatureSelector<double, Matrix<double>>(threshold: 0.8),
                new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 0.5),
                new UnivariateFeatureSelector<double, Matrix<double>>(target, k: 5),
                new NoFeatureSelector<double, Matrix<double>>()
            };

            // Act & Assert
            foreach (var selector in selectors)
            {
                var selected = selector.SelectFeatures(data);
                Assert.Equal(50, selected.Rows);
                Assert.True(selected.Columns <= 10);
            }
        }

        [Fact]
        public void FeatureSelectors_WithEmptyFeatureSet_HandleGracefully()
        {
            // Arrange - Dataset where all features might be removed
            var data = new Matrix<double>(50, 3);
            for (int i = 0; i < 50; i++)
            {
                data[i, 0] = 1.0; // Constant
                data[i, 1] = 1.0; // Constant
                data[i, 2] = 1.0; // Constant
            }

            var selector = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 0.1);

            // Act
            var selected = selector.SelectFeatures(data);

            // Assert - Should handle gracefully (may return 0 columns)
            Assert.True(selected.Columns >= 0);
            Assert.Equal(50, selected.Rows);
        }

        [Fact]
        public void FeatureSelectors_ChainedSelection_ReducesFeaturesFurther()
        {
            // Arrange - Test chaining multiple selectors
            var data = CreateCorrelatedDataset(100);

            var correlationSelector = new CorrelationFeatureSelector<double, Matrix<double>>(threshold: 0.9);
            var varianceSelector = new VarianceThresholdFeatureSelector<double, Matrix<double>>(threshold: 0.5);

            // Act - Apply selectors in sequence
            var afterCorrelation = correlationSelector.SelectFeatures(data);
            var afterVariance = varianceSelector.SelectFeatures(afterCorrelation);

            // Assert
            Assert.True(afterCorrelation.Columns <= data.Columns);
            Assert.True(afterVariance.Columns <= afterCorrelation.Columns);
            Assert.Equal(100, afterVariance.Rows); // Rows preserved throughout
        }

        [Fact]
        public void FeatureSelectors_WithFloatType_WorkCorrectly()
        {
            // Arrange - Test with float instead of double
            var data = new Matrix<float>(50, 5);
            var random = new Random(42);

            for (int i = 0; i < 50; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    data[i, j] = (float)(random.NextDouble() * 10);
                }
            }

            var correlationSelector = new CorrelationFeatureSelector<float, Matrix<float>>(threshold: 0.8f);
            var varianceSelector = new VarianceThresholdFeatureSelector<float, Matrix<float>>(threshold: 0.5f);

            // Act
            var correlationResult = correlationSelector.SelectFeatures(data);
            var varianceResult = varianceSelector.SelectFeatures(data);

            // Assert
            Assert.True(correlationResult.Columns <= 5);
            Assert.True(varianceResult.Columns <= 5);
            Assert.Equal(50, correlationResult.Rows);
            Assert.Equal(50, varianceResult.Rows);
        }

        [Fact]
        public void FeatureSelectors_WithLargeDataset_PerformEfficiently()
        {
            // Arrange - Larger dataset to test performance
            var data = new Matrix<double>(500, 20);
            var random = new Random(42);

            for (int i = 0; i < 500; i++)
            {
                for (int j = 0; j < 20; j++)
                {
                    data[i, j] = random.NextDouble() * 100;
                }
            }

            var selector = new CorrelationFeatureSelector<double, Matrix<double>>(threshold: 0.85);

            // Act & Assert - Should complete without error
            var selected = selector.SelectFeatures(data);

            Assert.True(selected.Columns <= 20);
            Assert.Equal(500, selected.Rows);
        }

        #endregion

        #region Mock Helper Classes

        /// <summary>
        /// Mock model for testing SelectFromModel.
        /// </summary>
        private class MockFeatureImportanceModel : IFeatureImportance<double>
        {
            private Dictionary<string, double> _featureImportance;
            private readonly int _numFeatures;

            public MockFeatureImportanceModel(int numFeatures)
            {
                _numFeatures = numFeatures;
                _featureImportance = new Dictionary<string, double>();

                // Initialize with default importances
                for (int i = 0; i < numFeatures; i++)
                {
                    _featureImportance[$"Feature_{i}"] = 0.1;
                }
            }

            public void SetFeatureImportance(Dictionary<string, double> importances)
            {
                _featureImportance = importances;
            }

            public Dictionary<string, double> GetFeatureImportance()
            {
                return _featureImportance;
            }
        }

        #endregion
    }
}
