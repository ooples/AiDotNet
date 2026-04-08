using System;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.Helpers
{
    public class ConversionsHelperTests
    {
        [Fact]
        public void ConvertToMatrix_WithMatrix_ReturnsOriginalMatrix()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 3);
            matrix[0, 0] = 1.0;
            matrix[1, 2] = 5.0;

            // Act
            var result = ConversionsHelper.ConvertToMatrix<double, Matrix<double>>(matrix);

            // Assert
            Assert.Same(matrix, result);
        }

        [Fact]
        public void ConvertToMatrix_With2DTensor_ConvertsCorrectly()
        {
            // Arrange - use proper multi-dimensional indexing
            var tensor = new Tensor<double>(new int[] { 2, 3 });
            tensor[0, 0] = 1.0;
            tensor[0, 1] = 2.0;
            tensor[0, 2] = 3.0;
            tensor[1, 0] = 4.0;
            tensor[1, 1] = 5.0;
            tensor[1, 2] = 6.0;

            // Act
            var result = ConversionsHelper.ConvertToMatrix<double, Tensor<double>>(tensor);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(2, result.Rows);
            Assert.Equal(3, result.Columns);
        }

        [Fact]
        public void ConvertToMatrix_WithHigherDimensionalTensor_Reshapes()
        {
            // Arrange - use proper 3D indexing
            var tensor = new Tensor<double>(new int[] { 2, 2, 2 });
            int val = 0;
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    for (int k = 0; k < 2; k++)
                        tensor[i, j, k] = val++;

            // Act
            var result = ConversionsHelper.ConvertToMatrix<double, Tensor<double>>(tensor);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(8 / 2, result.Rows);
            Assert.Equal(2, result.Columns);
        }

        [Fact]
        public void ConvertToVector_WithVector_ReturnsOriginalVector()
        {
            // Arrange
            var vector = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = ConversionsHelper.ConvertToVector<double, Vector<double>>(vector);

            // Assert
            Assert.Same(vector, result);
        }

        [Fact]
        public void ConvertToVector_WithTensor_FlattensCorrectly()
        {
            // Arrange - use proper 2D indexing
            var tensor = new Tensor<double>(new int[] { 2, 3 });
            tensor[0, 0] = 1.0;
            tensor[0, 1] = 2.0;
            tensor[0, 2] = 3.0;
            tensor[1, 0] = 4.0;
            tensor[1, 1] = 5.0;
            tensor[1, 2] = 6.0;

            // Act
            var result = ConversionsHelper.ConvertToVector<double, Tensor<double>>(tensor);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(6, result.Length);
        }

        [Fact]
        public void ConvertToScalar_WithScalar_ReturnsValue()
        {
            // Arrange
            double value = 42.0;

            // Act
            var result = ConversionsHelper.ConvertToScalar<double, double>(value);

            // Assert
            Assert.Equal(42.0, result);
        }

        [Fact]
        public void ConvertToScalar_WithVector_ReturnsFirstElement()
        {
            // Arrange
            var vector = new Vector<double>(new double[] { 10.0, 20.0, 30.0 });

            // Act
            var result = ConversionsHelper.ConvertToScalar<double, Vector<double>>(vector);

            // Assert
            Assert.Equal(10.0, result);
        }

        [Fact]
        public void ConvertToScalar_WithMatrix_ReturnsFirstElement()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 2);
            matrix[0, 0] = 5.5;
            matrix[0, 1] = 6.6;

            // Act
            var result = ConversionsHelper.ConvertToScalar<double, Matrix<double>>(matrix);

            // Assert
            Assert.Equal(5.5, result);
        }

        [Fact]
        public void ConvertToScalar_WithTensor_ReturnsFirstElement()
        {
            // Arrange - use proper 2D indexing
            var tensor = new Tensor<double>(new int[] { 2, 3 });
            tensor[0, 0] = 7.7;
            tensor[0, 1] = 8.8;

            // Act
            var result = ConversionsHelper.ConvertToScalar<double, Tensor<double>>(tensor);

            // Assert
            Assert.Equal(7.7, result);
        }

        [Fact]
        public void ConvertToScalar_WithEmptyVector_ThrowsInvalidOperationException()
        {
            // Arrange
            var vector = new Vector<double>(0);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() =>
                ConversionsHelper.ConvertToScalar<double, Vector<double>>(vector));
        }

        [Fact]
        public void ConvertToScalar_WithEmptyMatrix_ThrowsInvalidOperationException()
        {
            // Arrange
            var matrix = new Matrix<double>(0, 0);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() =>
                ConversionsHelper.ConvertToScalar<double, Matrix<double>>(matrix));
        }

        [Fact]
        public void ConvertToScalar_WithEmptyTensor_ThrowsInvalidOperationException()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 0 });

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() =>
                ConversionsHelper.ConvertToScalar<double, Tensor<double>>(tensor));
        }

        [Fact]
        public void ConvertObjectToVector_WithNull_ReturnsNull()
        {
            // Act
            var result = ConversionsHelper.ConvertObjectToVector<double>(null);

            // Assert
            Assert.Null(result);
        }

        [Fact]
        public void ConvertObjectToVector_WithVector_ReturnsVector()
        {
            // Arrange
            object vector = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = ConversionsHelper.ConvertObjectToVector<double>(vector);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(3, result.Length);
        }

        [Fact]
        public void ConvertObjectToVector_WithTensor_ConvertsToVector()
        {
            // Arrange
            object tensor = new Tensor<double>(new int[] { 3 });
            ((Tensor<double>)tensor)[0] = 1.0;
            ((Tensor<double>)tensor)[1] = 2.0;
            ((Tensor<double>)tensor)[2] = 3.0;

            // Act
            var result = ConversionsHelper.ConvertObjectToVector<double>(tensor);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(3, result.Length);
        }

        [Fact]
        public void ConvertFitFunction_WithMatrixToMatrix_WorksCorrectly()
        {
            // Arrange
            Func<Matrix<double>, Vector<double>> originalFunc = m =>
            {
                var result = new Vector<double>(m.Rows);
                for (int i = 0; i < m.Rows; i++)
                    result[i] = m[i, 0];
                return result;
            };
            var matrix = new Matrix<double>(3, 2);
            matrix[0, 0] = 1.0;
            matrix[1, 0] = 2.0;
            matrix[2, 0] = 3.0;

            // Act
            var convertedFunc = ConversionsHelper.ConvertFitFunction<double, Matrix<double>, Vector<double>>(originalFunc);
            var result = convertedFunc(matrix);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(3, result.Length);
            Assert.Equal(1.0, result[0]);
            Assert.Equal(2.0, result[1]);
            Assert.Equal(3.0, result[2]);
        }

        [Fact]
        public void TensorToMatrix_WithMatchingDimensions_ConvertsCorrectly()
        {
            // Arrange - use proper 2D indexing
            var tensor = new Tensor<double>(new int[] { 2, 3 });
            int val = 0;
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++)
                    tensor[i, j] = val++;

            // Act
            var result = ConversionsHelper.TensorToMatrix(tensor, 2, 3);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(2, result.Rows);
            Assert.Equal(3, result.Columns);
        }

        [Fact]
        public void TensorToMatrix_WithMismatchedDimensions_ThrowsArgumentException()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 2, 3 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                ConversionsHelper.TensorToMatrix(tensor, 3, 3));
        }

        [Fact]
        public void MatrixToTensor_WithValidShape_ConvertsCorrectly()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 3);
            for (int i = 0; i < matrix.Rows; i++)
                for (int j = 0; j < matrix.Columns; j++)
                    matrix[i, j] = (double)i * matrix.Columns + j;

            // Act
            var result = ConversionsHelper.MatrixToTensor(matrix, new int[] { 2, 3 });

            // Assert
            Assert.NotNull(result);
            Assert.Equal(2, result.Rank);
            Assert.Equal(2, result.Shape[0]);
            Assert.Equal(3, result.Shape[1]);
        }

        [Fact]
        public void MatrixToTensor_WithInvalidShape_ThrowsArgumentException()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 3);

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                ConversionsHelper.MatrixToTensor(matrix, new int[] { 3, 3 }));
        }

        [Fact]
        public void VectorToTensor_WithValidShape_ConvertsCorrectly()
        {
            // Arrange
            var vector = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

            // Act
            var result = ConversionsHelper.VectorToTensor(vector, new int[] { 2, 3 });

            // Assert
            Assert.NotNull(result);
            Assert.Equal(2, result.Rank);
            Assert.Equal(2, result.Shape[0]);
            Assert.Equal(3, result.Shape[1]);
        }

        [Fact]
        public void VectorToTensor_WithInvalidShape_ThrowsArgumentException()
        {
            // Arrange
            var vector = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                ConversionsHelper.VectorToTensor(vector, new int[] { 2, 3 }));
        }

        [Fact]
        public void ConvertToTensor_WithMatrix_ConvertsCorrectly()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 2);
            matrix[0, 0] = 1.0;
            matrix[1, 1] = 4.0;

            // Act
            var result = ConversionsHelper.ConvertToTensor<double>(matrix);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(2, result.Rank);
        }

        [Fact]
        public void ConvertToTensor_WithVector_ConvertsCorrectly()
        {
            // Arrange
            var vector = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            var result = ConversionsHelper.ConvertToTensor<double>(vector);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(1, result.Rank);
        }

        [Fact]
        public void ConvertToTensor_WithTensor_ReturnsSameTensor()
        {
            // Arrange
            var tensor = new Tensor<double>(new int[] { 2, 3 });

            // Act
            var result = ConversionsHelper.ConvertToTensor<double>(tensor);

            // Assert
            Assert.Same(tensor, result);
        }

        [Fact]
        public void ConvertToMatrix_WithFloat_WorksCorrectly()
        {
            // Arrange
            var matrix = new Matrix<float>(2, 2);
            matrix[0, 0] = 1.0f;
            matrix[1, 1] = 2.0f;

            // Act
            var result = ConversionsHelper.ConvertToMatrix<float, Matrix<float>>(matrix);

            // Assert
            Assert.Same(matrix, result);
        }

        [Fact]
        public void ConvertToVector_WithFloat_WorksCorrectly()
        {
            // Arrange
            var vector = new Vector<float>(new float[] { 1.0f, 2.0f, 3.0f });

            // Act
            var result = ConversionsHelper.ConvertToVector<float, Vector<float>>(vector);

            // Assert
            Assert.Same(vector, result);
        }

        [Fact]
        public void ConvertToScalar_WithFloat_WorksCorrectly()
        {
            // Arrange
            float value = 42.5f;

            // Act
            var result = ConversionsHelper.ConvertToScalar<float, float>(value);

            // Assert
            Assert.Equal(42.5f, result);
        }

        [Fact]
        public void VectorToTensor_With3DShape_ConvertsCorrectly()
        {
            // Arrange
            var vector = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8 });

            // Act
            var result = ConversionsHelper.VectorToTensor(vector, new int[] { 2, 2, 2 });

            // Assert
            Assert.NotNull(result);
            Assert.Equal(3, result.Rank);
            Assert.Equal(2, result.Shape[0]);
            Assert.Equal(2, result.Shape[1]);
            Assert.Equal(2, result.Shape[2]);
        }

        [Fact]
        public void MatrixToTensor_With3DShape_ConvertsCorrectly()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 4);
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 4; j++)
                    matrix[i, j] = (double)i * 4 + j;

            // Act
            var result = ConversionsHelper.MatrixToTensor(matrix, new int[] { 2, 2, 2 });

            // Assert
            Assert.NotNull(result);
            Assert.Equal(3, result.Rank);
            Assert.Equal(8, result.Length);
        }
    }
}
