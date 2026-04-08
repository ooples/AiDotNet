using AiDotNet.LinearAlgebra;
using AiDotNet.Normalizers;
using AiDotNet.Models;
using AiDotNet.Enums;
using Xunit;
using System;

namespace AiDotNetTests.UnitTests.Normalizers
{
    public class QuantileTransformerTests
    {
        [Fact]
        public void Constructor_WithValidUniformDistribution_Succeeds()
        {
            // Act
            var transformer = new QuantileTransformer<double, Matrix<double>, Vector<double>>(OutputDistribution.Uniform, 100);

            // Assert
            Assert.NotNull(transformer);
        }

        [Fact]
        public void Constructor_WithValidNormalDistribution_Succeeds()
        {
            // Act
            var transformer = new QuantileTransformer<double, Matrix<double>, Vector<double>>(OutputDistribution.Normal, 100);

            // Assert
            Assert.NotNull(transformer);
        }

        [Fact]
        public void Constructor_WithTooFewQuantiles_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                new QuantileTransformer<double, Matrix<double>, Vector<double>>(OutputDistribution.Uniform, 5));
        }

        [Fact]
        public void NormalizeOutput_WithUniformDistribution_MapsToZeroOne()
        {
            // Arrange
            var transformer = new QuantileTransformer<double, Matrix<double>, Vector<double>>(OutputDistribution.Uniform, 100);
            var data = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 });

            // Act
            var (normalized, parameters) = transformer.NormalizeOutput(data);

            // Assert
            Assert.Equal(NormalizationMethod.QuantileTransformer, parameters.Method);
            Assert.Equal(OutputDistribution.Uniform, parameters.OutputDistribution);
            Assert.Equal(100, parameters.Quantiles.Count);

            // Values should be between 0 and 1
            for (int i = 0; i < normalized.Length; i++)
            {
                Assert.True(normalized[i] >= 0.0 && normalized[i] <= 1.0,
                    $"Normalized value {normalized[i]} should be between 0 and 1");
            }

            // Smallest value should map close to 0, largest close to 1
            Assert.True(normalized[0] < 0.2, "Smallest value should map close to 0");
            Assert.True(normalized[normalized.Length - 1] > 0.8, "Largest value should map close to 1");
        }

        [Fact]
        public void NormalizeOutput_WithNormalDistribution_TransformsData()
        {
            // Arrange
            var transformer = new QuantileTransformer<double, Matrix<double>, Vector<double>>(OutputDistribution.Normal, 100);
            var data = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 });

            // Act
            var (normalized, parameters) = transformer.NormalizeOutput(data);

            // Assert
            Assert.Equal(NormalizationMethod.QuantileTransformer, parameters.Method);
            Assert.Equal(OutputDistribution.Normal, parameters.OutputDistribution);
            Assert.Equal(100, parameters.Quantiles.Count);

            // With normal distribution, values are not strictly bounded but should be reasonable
            // Most values should be between -3 and 3 for a normal distribution
            int valuesInRange = 0;
            for (int i = 0; i < normalized.Length; i++)
            {
                if (normalized[i] >= -3.0 && normalized[i] <= 3.0)
                {
                    valuesInRange++;
                }
            }
            Assert.True(valuesInRange >= normalized.Length * 0.7,
                "Most values should fall within 3 standard deviations");
        }

        [Fact]
        public void NormalizeOutput_WithSkewedData_HandlesOutliers()
        {
            // Arrange
            var transformer = new QuantileTransformer<double, Matrix<double>, Vector<double>>(OutputDistribution.Uniform, 100);
            // Data with outliers: most values are small, few are very large
            var data = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 1000.0 });

            // Act
            var (normalized, _) = transformer.NormalizeOutput(data);

            // Assert
            // All values should still be mapped to [0, 1] range
            for (int i = 0; i < normalized.Length; i++)
            {
                Assert.True(normalized[i] >= 0.0 && normalized[i] <= 1.0,
                    $"Value {normalized[i]} should be in [0, 1] range despite outliers");
            }

            // The outliers should not cause the smaller values to cluster too tightly
            // There should be some spread in the normalized values
            var distinctValues = new HashSet<double>();
            foreach (var value in normalized)
            {
                distinctValues.Add(Math.Round(value, 2));
            }
            Assert.True(distinctValues.Count > 3, "Values should be reasonably spread out");
        }

        [Fact]
        public void NormalizeInput_WithMatrix_NormalizesEachColumnIndependently()
        {
            // Arrange
            var transformer = new QuantileTransformer<double, Matrix<double>, Vector<double>>(OutputDistribution.Uniform, 100);
            var matrix = new Matrix<double>(5, 2);
            // First column: 1-5
            matrix[0, 0] = 1.0; matrix[0, 1] = 10.0;
            matrix[1, 0] = 2.0; matrix[1, 1] = 20.0;
            matrix[2, 0] = 3.0; matrix[2, 1] = 30.0;
            matrix[3, 0] = 4.0; matrix[3, 1] = 40.0;
            matrix[4, 0] = 5.0; matrix[4, 1] = 50.0;

            // Act
            var (normalized, parametersList) = transformer.NormalizeInput(matrix);

            // Assert
            Assert.Equal(2, parametersList.Count);
            Assert.Equal(100, parametersList[0].Quantiles.Count);
            Assert.Equal(100, parametersList[1].Quantiles.Count);

            // Each column should have values in [0, 1]
            for (int col = 0; col < 2; col++)
            {
                for (int row = 0; row < 5; row++)
                {
                    Assert.True(normalized[row, col] >= 0.0 && normalized[row, col] <= 1.0,
                        $"Value at [{row}, {col}] should be in [0, 1] range");
                }
            }
        }

        [Fact]
        public void Denormalize_WithUniformDistribution_RestoresApproximateValues()
        {
            // Arrange
            var transformer = new QuantileTransformer<double, Matrix<double>, Vector<double>>(OutputDistribution.Uniform, 1000);
            var original = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 });
            var (normalized, parameters) = transformer.NormalizeOutput(original);

            // Act
            var denormalized = transformer.Denormalize(normalized, parameters);

            // Assert
            // Due to quantile interpolation, values might not be exactly the same but should be close
            for (int i = 0; i < original.Length; i++)
            {
                Assert.Equal(original[i], denormalized[i], 1); // Within 1 unit
            }
        }

        [Fact]
        public void Denormalize_WithNormalDistribution_RestoresApproximateValues()
        {
            // Arrange
            var transformer = new QuantileTransformer<double, Matrix<double>, Vector<double>>(OutputDistribution.Normal, 1000);
            var original = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 });
            var (normalized, parameters) = transformer.NormalizeOutput(original);

            // Act
            var denormalized = transformer.Denormalize(normalized, parameters);

            // Assert
            // Values should be reasonably close to original
            for (int i = 0; i < original.Length; i++)
            {
                Assert.Equal(original[i], denormalized[i], 1); // Within 1 unit
            }
        }

        [Fact]
        public void Denormalize_Coefficients_ThrowsNotSupportedException()
        {
            // Arrange
            var transformer = new QuantileTransformer<double, Matrix<double>, Vector<double>>();
            var coefficients = new Vector<double>(new double[] { 0.5, 1.0 });
            var xParams = new List<NormalizationParameters<double>>();
            var yParams = new NormalizationParameters<double>();

            // Act & Assert
            Assert.Throws<NotSupportedException>(() =>
                transformer.Denormalize(coefficients, xParams, yParams));
        }

        [Fact]
        public void Denormalize_Intercept_ThrowsNotSupportedException()
        {
            // Arrange
            var transformer = new QuantileTransformer<double, Matrix<double>, Vector<double>>();
            var xMatrix = new Matrix<double>(2, 2);
            var y = new Vector<double>(new double[] { 1.0, 2.0 });
            var coefficients = new Vector<double>(new double[] { 0.5, 1.0 });
            var xParams = new List<NormalizationParameters<double>>();
            var yParams = new NormalizationParameters<double>();

            // Act & Assert
            Assert.Throws<NotSupportedException>(() =>
                transformer.Denormalize(xMatrix, y, coefficients, xParams, yParams));
        }

        [Fact]
        public void NormalizeOutput_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var transformer = new QuantileTransformer<float, Matrix<float>, Vector<float>>(OutputDistribution.Uniform, 100);
            var data = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });

            // Act
            var (normalized, parameters) = transformer.NormalizeOutput(data);

            // Assert
            Assert.Equal(100, parameters.Quantiles.Count);
            for (int i = 0; i < normalized.Length; i++)
            {
                Assert.True(normalized[i] >= 0.0f && normalized[i] <= 1.0f);
            }
        }

        [Fact]
        public void NormalizeOutput_WithTensor_WorksCorrectly()
        {
            // Arrange
            var transformer = new QuantileTransformer<double, Matrix<double>, Tensor<double>>(OutputDistribution.Uniform, 100);
            var dataVector = Vector<double>.FromArray(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var data = new Tensor<double>(new[] { 5 }, dataVector);

            // Act
            var (normalized, parameters) = transformer.NormalizeOutput(data);

            // Assert
            Assert.Equal(100, parameters.Quantiles.Count);
            for (int i = 0; i < normalized.Shape[0]; i++)
            {
                Assert.True(normalized[i] >= 0.0 && normalized[i] <= 1.0);
            }
        }

        [Fact]
        public void NormalizeInput_WithTensor_WorksCorrectly()
        {
            // Arrange
            var transformer = new QuantileTransformer<double, Tensor<double>, Vector<double>>(OutputDistribution.Uniform, 100);
            var tensor = new Tensor<double>(new[] { 5, 2 });
            tensor[0, 0] = 1.0; tensor[0, 1] = 10.0;
            tensor[1, 0] = 2.0; tensor[1, 1] = 20.0;
            tensor[2, 0] = 3.0; tensor[2, 1] = 30.0;
            tensor[3, 0] = 4.0; tensor[3, 1] = 40.0;
            tensor[4, 0] = 5.0; tensor[4, 1] = 50.0;

            // Act
            var (_, parametersList) = transformer.NormalizeInput(tensor);

            // Assert
            Assert.Equal(2, parametersList.Count);
        }

        [Fact]
        public void NormalizeOutput_PreservesRankOrdering()
        {
            // Arrange
            var transformer = new QuantileTransformer<double, Matrix<double>, Vector<double>>(OutputDistribution.Uniform, 100);
            var data = new Vector<double>(new double[] { 10.0, 5.0, 20.0, 15.0, 1.0 });

            // Act
            var (normalized, _) = transformer.NormalizeOutput(data);

            // Assert
            // The rank ordering should be preserved: 1.0 < 5.0 < 10.0 < 15.0 < 20.0
            // So indices: 4 < 1 < 0 < 3 < 2
            Assert.True(normalized[4] < normalized[1], "1.0 should map lower than 5.0");
            Assert.True(normalized[1] < normalized[0], "5.0 should map lower than 10.0");
            Assert.True(normalized[0] < normalized[3], "10.0 should map lower than 15.0");
            Assert.True(normalized[3] < normalized[2], "15.0 should map lower than 20.0");
        }

        [Fact]
        public void RoundTrip_NormalizeAndDenormalize_ReturnsApproximateOriginal()
        {
            // Arrange
            var transformer = new QuantileTransformer<double, Matrix<double>, Vector<double>>(OutputDistribution.Uniform, 1000);
            var original = new Vector<double>(new double[] { 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0 });

            // Act
            var (normalized, parameters) = transformer.NormalizeOutput(original);
            var denormalized = transformer.Denormalize(normalized, parameters);

            // Assert
            for (int i = 0; i < original.Length; i++)
            {
                // Allow some tolerance due to quantile interpolation
                var relativeError = Math.Abs((denormalized[i] - original[i]) / original[i]);
                Assert.True(relativeError < 0.1,
                    $"Relative error {relativeError} should be less than 10% for value {original[i]}");
            }
        }

        [Fact]
        public void NormalizeOutput_WithRepeatedValues_HandlesCorrectly()
        {
            // Arrange
            var transformer = new QuantileTransformer<double, Matrix<double>, Vector<double>>(OutputDistribution.Uniform, 100);
            var data = new Vector<double>(new double[] { 1.0, 1.0, 1.0, 5.0, 5.0, 10.0 });

            // Act
            var (normalized, _) = transformer.NormalizeOutput(data);

            // Assert
            // Repeated values should map to similar (not necessarily identical) normalized values
            Assert.True(Math.Abs(normalized[0] - normalized[1]) < 0.2,
                "Repeated value 1.0 should map to similar values");
            Assert.True(Math.Abs(normalized[3] - normalized[4]) < 0.2,
                "Repeated value 5.0 should map to similar values");
        }

        [Fact]
        public void NormalizeOutput_WithExtremeOutliers_HandlesGracefully()
        {
            // Arrange
            var transformer = new QuantileTransformer<double, Matrix<double>, Vector<double>>(OutputDistribution.Uniform, 100);
            var data = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 1000000.0 });

            // Act
            var (normalized, _) = transformer.NormalizeOutput(data);

            // Assert
            // All values should still be in valid range
            for (int i = 0; i < normalized.Length; i++)
            {
                Assert.True(normalized[i] >= 0.0 && normalized[i] <= 1.0,
                    "Even with extreme outliers, values should be in [0, 1]");
            }

            // The extreme outlier shouldn't cause numerical issues
            Assert.False(double.IsNaN(normalized[normalized.Length - 1]),
                "Extreme outlier shouldn't result in NaN");
            Assert.False(double.IsInfinity(normalized[normalized.Length - 1]),
                "Extreme outlier shouldn't result in infinity");
        }
    }
}
