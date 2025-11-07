using AiDotNet.LinearAlgebra;
using AiDotNet.Normalizers;
using AiDotNet.Models;
using AiDotNet.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Normalizers
{
    public class MaxAbsScalerTests
    {
        [Fact]
        public void NormalizeOutput_WithPositiveValues_ScalesToOneOrLess()
        {
            // Arrange
            var scaler = new MaxAbsScaler<double, Matrix<double>, Vector<double>>();
            var data = new Vector<double>(new double[] { 10.0, 20.0, 50.0, 100.0 });

            // Act
            var (normalized, parameters) = scaler.NormalizeOutput(data);

            // Assert
            Assert.Equal(NormalizationMethod.MaxAbsScaler, parameters.Method);
            Assert.Equal(100.0, parameters.MaxAbs);
            Assert.Equal(0.1, normalized[0], 10);
            Assert.Equal(0.2, normalized[1], 10);
            Assert.Equal(0.5, normalized[2], 10);
            Assert.Equal(1.0, normalized[3], 10);
        }

        [Fact]
        public void NormalizeOutput_WithNegativeAndPositiveValues_PreservesSign()
        {
            // Arrange
            var scaler = new MaxAbsScaler<double, Matrix<double>, Vector<double>>();
            var data = new Vector<double>(new double[] { -50.0, -25.0, 25.0, 100.0 });

            // Act
            var (normalized, parameters) = scaler.NormalizeOutput(data);

            // Assert
            Assert.Equal(100.0, parameters.MaxAbs);
            Assert.Equal(-0.5, normalized[0], 10);
            Assert.Equal(-0.25, normalized[1], 10);
            Assert.Equal(0.25, normalized[2], 10);
            Assert.Equal(1.0, normalized[3], 10);
        }

        [Fact]
        public void NormalizeOutput_WithZeros_PreservesZeros()
        {
            // Arrange
            var scaler = new MaxAbsScaler<double, Matrix<double>, Vector<double>>();
            var data = new Vector<double>(new double[] { 0.0, 50.0, 0.0, 100.0 });

            // Act
            var (normalized, parameters) = scaler.NormalizeOutput(data);

            // Assert
            Assert.Equal(100.0, parameters.MaxAbs);
            Assert.Equal(0.0, normalized[0], 10);
            Assert.Equal(0.5, normalized[1], 10);
            Assert.Equal(0.0, normalized[2], 10);
            Assert.Equal(1.0, normalized[3], 10);
        }

        [Fact]
        public void NormalizeOutput_WithAllZeros_HandlesGracefully()
        {
            // Arrange
            var scaler = new MaxAbsScaler<double, Matrix<double>, Vector<double>>();
            var data = new Vector<double>(new double[] { 0.0, 0.0, 0.0 });

            // Act
            var (normalized, parameters) = scaler.NormalizeOutput(data);

            // Assert
            Assert.Equal(1.0, parameters.MaxAbs); // Set to 1 to avoid division by zero
            Assert.Equal(0.0, normalized[0], 10);
            Assert.Equal(0.0, normalized[1], 10);
            Assert.Equal(0.0, normalized[2], 10);
        }

        [Fact]
        public void NormalizeOutput_WithSingleValue_NormalizesToOne()
        {
            // Arrange
            var scaler = new MaxAbsScaler<double, Matrix<double>, Vector<double>>();
            var data = new Vector<double>(new double[] { 42.0 });

            // Act
            var (normalized, parameters) = scaler.NormalizeOutput(data);

            // Assert
            Assert.Equal(42.0, parameters.MaxAbs);
            Assert.Equal(1.0, normalized[0], 10);
        }

        [Fact]
        public void NormalizeInput_WithMatrix_NormalizesEachColumnIndependently()
        {
            // Arrange
            var scaler = new MaxAbsScaler<double, Matrix<double>, Vector<double>>();
            var matrix = new Matrix<double>(3, 2);
            matrix[0, 0] = 10.0; matrix[0, 1] = 200.0;
            matrix[1, 0] = 20.0; matrix[1, 1] = 400.0;
            matrix[2, 0] = 50.0; matrix[2, 1] = 100.0;

            // Act
            var (normalized, parametersList) = scaler.NormalizeInput(matrix);

            // Assert
            Assert.Equal(2, parametersList.Count);
            Assert.Equal(50.0, parametersList[0].MaxAbs);
            Assert.Equal(400.0, parametersList[1].MaxAbs);

            // Check first column
            Assert.Equal(0.2, normalized[0, 0], 10);
            Assert.Equal(0.4, normalized[1, 0], 10);
            Assert.Equal(1.0, normalized[2, 0], 10);

            // Check second column
            Assert.Equal(0.5, normalized[0, 1], 10);
            Assert.Equal(1.0, normalized[1, 1], 10);
            Assert.Equal(0.25, normalized[2, 1], 10);
        }

        [Fact]
        public void Denormalize_WithVector_RestoresOriginalValues()
        {
            // Arrange
            var scaler = new MaxAbsScaler<double, Matrix<double>, Vector<double>>();
            var original = new Vector<double>(new double[] { -50.0, -25.0, 25.0, 100.0 });
            var (normalized, parameters) = scaler.NormalizeOutput(original);

            // Act
            var denormalized = scaler.Denormalize(normalized, parameters);

            // Assert
            Assert.Equal(original[0], denormalized[0], 10);
            Assert.Equal(original[1], denormalized[1], 10);
            Assert.Equal(original[2], denormalized[2], 10);
            Assert.Equal(original[3], denormalized[3], 10);
        }

        [Fact]
        public void Denormalize_Coefficients_ReturnsCorrectValues()
        {
            // Arrange
            var scaler = new MaxAbsScaler<double, Matrix<double>, Vector<double>>();
            var coefficients = new Vector<double>(new double[] { 0.5, 1.0 });

            var xParams = new List<NormalizationParameters<double>>
            {
                new NormalizationParameters<double> { MaxAbs = 100.0, Method = NormalizationMethod.MaxAbsScaler },
                new NormalizationParameters<double> { MaxAbs = 200.0, Method = NormalizationMethod.MaxAbsScaler }
            };

            var yParams = new NormalizationParameters<double> { MaxAbs = 50.0, Method = NormalizationMethod.MaxAbsScaler };

            // Act
            var denormalized = scaler.Denormalize(coefficients, xParams, yParams);

            // Assert
            // coef_denorm = coef_norm * (maxAbs_y / maxAbs_x)
            // First: 0.5 * (50 / 100) = 0.25
            // Second: 1.0 * (50 / 200) = 0.25
            Assert.Equal(0.25, denormalized[0], 10);
            Assert.Equal(0.25, denormalized[1], 10);
        }

        [Fact]
        public void Denormalize_Intercept_ReturnsZero()
        {
            // Arrange
            var scaler = new MaxAbsScaler<double, Matrix<double>, Vector<double>>();
            var xMatrix = new Matrix<double>(2, 2);
            var y = new Vector<double>(new double[] { 1.0, 2.0 });
            var coefficients = new Vector<double>(new double[] { 0.5, 1.0 });
            var xParams = new List<NormalizationParameters<double>>
            {
                new NormalizationParameters<double> { MaxAbs = 100.0, Method = NormalizationMethod.MaxAbsScaler },
                new NormalizationParameters<double> { MaxAbs = 200.0, Method = NormalizationMethod.MaxAbsScaler }
            };
            var yParams = new NormalizationParameters<double> { MaxAbs = 50.0, Method = NormalizationMethod.MaxAbsScaler };

            // Act
            var intercept = scaler.Denormalize(xMatrix, y, coefficients, xParams, yParams);

            // Assert
            // MaxAbsScaler doesn't involve shifting, so intercept should be zero
            Assert.Equal(0.0, intercept, 10);
        }

        [Fact]
        public void NormalizeOutput_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var scaler = new MaxAbsScaler<float, Matrix<float>, Vector<float>>();
            var data = new Vector<float>(new float[] { 10.0f, 20.0f, 50.0f, 100.0f });

            // Act
            var (normalized, parameters) = scaler.NormalizeOutput(data);

            // Assert
            Assert.Equal(100.0f, parameters.MaxAbs);
            Assert.Equal(0.1f, normalized[0], 5);
            Assert.Equal(0.2f, normalized[1], 5);
            Assert.Equal(0.5f, normalized[2], 5);
            Assert.Equal(1.0f, normalized[3], 5);
        }

        [Fact]
        public void NormalizeInput_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var scaler = new MaxAbsScaler<float, Matrix<float>, Vector<float>>();
            var matrix = new Matrix<float>(3, 2);
            matrix[0, 0] = 10.0f; matrix[0, 1] = 200.0f;
            matrix[1, 0] = 20.0f; matrix[1, 1] = 400.0f;
            matrix[2, 0] = 50.0f; matrix[2, 1] = 100.0f;

            // Act
            var (normalized, parametersList) = scaler.NormalizeInput(matrix);

            // Assert
            Assert.Equal(2, parametersList.Count);
            Assert.Equal(50.0f, parametersList[0].MaxAbs);
            Assert.Equal(400.0f, parametersList[1].MaxAbs);
        }

        [Fact]
        public void NormalizeOutput_WithTensor_WorksCorrectly()
        {
            // Arrange
            var scaler = new MaxAbsScaler<double, Matrix<double>, Tensor<double>>();
            var data = new Tensor<double>(new[] { 4 }, new double[] { 10.0, 20.0, 50.0, 100.0 });

            // Act
            var (normalized, parameters) = scaler.NormalizeOutput(data);

            // Assert
            Assert.Equal(100.0, parameters.MaxAbs);
            Assert.Equal(0.1, normalized[0], 10);
            Assert.Equal(0.2, normalized[1], 10);
            Assert.Equal(0.5, normalized[2], 10);
            Assert.Equal(1.0, normalized[3], 10);
        }

        [Fact]
        public void NormalizeInput_WithTensor_WorksCorrectly()
        {
            // Arrange
            var scaler = new MaxAbsScaler<double, Tensor<double>, Vector<double>>();
            var tensor = new Tensor<double>(new[] { 3, 2 });
            tensor[0, 0] = 10.0; tensor[0, 1] = 200.0;
            tensor[1, 0] = 20.0; tensor[1, 1] = 400.0;
            tensor[2, 0] = 50.0; tensor[2, 1] = 100.0;

            // Act
            var (normalized, parametersList) = scaler.NormalizeInput(tensor);

            // Assert
            Assert.Equal(2, parametersList.Count);
            Assert.Equal(50.0, parametersList[0].MaxAbs);
            Assert.Equal(400.0, parametersList[1].MaxAbs);
        }

        [Fact]
        public void NormalizeOutput_WithLargeNegativeValue_NormalizesCorrectly()
        {
            // Arrange
            var scaler = new MaxAbsScaler<double, Matrix<double>, Vector<double>>();
            var data = new Vector<double>(new double[] { -200.0, -100.0, 50.0, 100.0 });

            // Act
            var (normalized, parameters) = scaler.NormalizeOutput(data);

            // Assert
            Assert.Equal(200.0, parameters.MaxAbs); // Largest absolute value is 200
            Assert.Equal(-1.0, normalized[0], 10);
            Assert.Equal(-0.5, normalized[1], 10);
            Assert.Equal(0.25, normalized[2], 10);
            Assert.Equal(0.5, normalized[3], 10);
        }

        [Fact]
        public void RoundTrip_NormalizeAndDenormalize_ReturnsOriginal()
        {
            // Arrange
            var scaler = new MaxAbsScaler<double, Matrix<double>, Vector<double>>();
            var original = new Vector<double>(new double[] { -200.0, -100.0, 0.0, 50.0, 100.0, 150.0 });

            // Act
            var (normalized, parameters) = scaler.NormalizeOutput(original);
            var denormalized = scaler.Denormalize(normalized, parameters);

            // Assert
            for (int i = 0; i < original.Length; i++)
            {
                Assert.Equal(original[i], denormalized[i], 10);
            }
        }
    }
}
