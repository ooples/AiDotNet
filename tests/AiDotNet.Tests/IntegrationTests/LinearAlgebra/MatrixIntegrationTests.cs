using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.IntegrationTests.LinearAlgebra
{
    /// <summary>
    /// Integration tests for Matrix operations with mathematically verified results.
    /// These tests validate the mathematical correctness of matrix operations.
    /// </summary>
    public class MatrixIntegrationTests
    {
        private const double Tolerance = 1e-10;

        [Fact]
        public void MatrixMultiplication_WithKnownValues_ProducesCorrectResult()
        {
            // Arrange - Using well-known matrix multiplication example
            // A = [[1, 2], [3, 4]]
            // B = [[5, 6], [7, 8]]
            // Expected A * B = [[19, 22], [43, 50]]
            var matrixA = new Matrix<double>(2, 2);
            matrixA[0, 0] = 1.0; matrixA[0, 1] = 2.0;
            matrixA[1, 0] = 3.0; matrixA[1, 1] = 4.0;

            var matrixB = new Matrix<double>(2, 2);
            matrixB[0, 0] = 5.0; matrixB[0, 1] = 6.0;
            matrixB[1, 0] = 7.0; matrixB[1, 1] = 8.0;

            // Act
            var result = matrixA * matrixB;

            // Assert - Mathematically verified results
            Assert.Equal(19.0, result[0, 0], precision: 10);
            Assert.Equal(22.0, result[0, 1], precision: 10);
            Assert.Equal(43.0, result[1, 0], precision: 10);
            Assert.Equal(50.0, result[1, 1], precision: 10);
        }

        [Fact]
        public void MatrixMultiplication_3x3Matrices_ProducesCorrectResult()
        {
            // Arrange
            // A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            // B = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
            // Expected: [[30, 24, 18], [84, 69, 54], [138, 114, 90]]
            var matrixA = new Matrix<double>(3, 3);
            matrixA[0, 0] = 1.0; matrixA[0, 1] = 2.0; matrixA[0, 2] = 3.0;
            matrixA[1, 0] = 4.0; matrixA[1, 1] = 5.0; matrixA[1, 2] = 6.0;
            matrixA[2, 0] = 7.0; matrixA[2, 1] = 8.0; matrixA[2, 2] = 9.0;

            var matrixB = new Matrix<double>(3, 3);
            matrixB[0, 0] = 9.0; matrixB[0, 1] = 8.0; matrixB[0, 2] = 7.0;
            matrixB[1, 0] = 6.0; matrixB[1, 1] = 5.0; matrixB[1, 2] = 4.0;
            matrixB[2, 0] = 3.0; matrixB[2, 1] = 2.0; matrixB[2, 2] = 1.0;

            // Act
            var result = matrixA * matrixB;

            // Assert
            Assert.Equal(30.0, result[0, 0], precision: 10);
            Assert.Equal(24.0, result[0, 1], precision: 10);
            Assert.Equal(18.0, result[0, 2], precision: 10);
            Assert.Equal(84.0, result[1, 0], precision: 10);
            Assert.Equal(69.0, result[1, 1], precision: 10);
            Assert.Equal(54.0, result[1, 2], precision: 10);
            Assert.Equal(138.0, result[2, 0], precision: 10);
            Assert.Equal(114.0, result[2, 1], precision: 10);
            Assert.Equal(90.0, result[2, 2], precision: 10);
        }

        [Fact]
        public void MatrixTranspose_ProducesCorrectResult()
        {
            // Arrange
            // A = [[1, 2, 3], [4, 5, 6]]
            // Expected transpose: [[1, 4], [2, 5], [3, 6]]
            var matrix = new Matrix<double>(2, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = 4.0; matrix[1, 1] = 5.0; matrix[1, 2] = 6.0;

            // Act
            var transposed = matrix.Transpose();

            // Assert
            Assert.Equal(3, transposed.Rows);
            Assert.Equal(2, transposed.Columns);
            Assert.Equal(1.0, transposed[0, 0], precision: 10);
            Assert.Equal(4.0, transposed[0, 1], precision: 10);
            Assert.Equal(2.0, transposed[1, 0], precision: 10);
            Assert.Equal(5.0, transposed[1, 1], precision: 10);
            Assert.Equal(3.0, transposed[2, 0], precision: 10);
            Assert.Equal(6.0, transposed[2, 1], precision: 10);
        }

        [Fact]
        public void IdentityMatrix_MultipliedByAnyMatrix_ReturnsOriginal()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 1.5; matrix[0, 1] = 2.3; matrix[0, 2] = 3.7;
            matrix[1, 0] = 4.2; matrix[1, 1] = 5.9; matrix[1, 2] = 6.1;
            matrix[2, 0] = 7.8; matrix[2, 1] = 8.4; matrix[2, 2] = 9.6;

            var identity = Matrix<double>.Identity(3);

            // Act
            var result = identity * matrix;

            // Assert - Should equal original matrix
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    Assert.Equal(matrix[i, j], result[i, j], precision: 10);
                }
            }
        }

        [Fact]
        public void MatrixAddition_ProducesCorrectResult()
        {
            // Arrange
            var matrixA = new Matrix<double>(2, 2);
            matrixA[0, 0] = 1.0; matrixA[0, 1] = 2.0;
            matrixA[1, 0] = 3.0; matrixA[1, 1] = 4.0;

            var matrixB = new Matrix<double>(2, 2);
            matrixB[0, 0] = 5.0; matrixB[0, 1] = 6.0;
            matrixB[1, 0] = 7.0; matrixB[1, 1] = 8.0;

            // Act
            var result = matrixA + matrixB;

            // Assert
            Assert.Equal(6.0, result[0, 0], precision: 10);
            Assert.Equal(8.0, result[0, 1], precision: 10);
            Assert.Equal(10.0, result[1, 0], precision: 10);
            Assert.Equal(12.0, result[1, 1], precision: 10);
        }

        [Fact]
        public void MatrixSubtraction_ProducesCorrectResult()
        {
            // Arrange
            var matrixA = new Matrix<double>(2, 2);
            matrixA[0, 0] = 10.0; matrixA[0, 1] = 20.0;
            matrixA[1, 0] = 30.0; matrixA[1, 1] = 40.0;

            var matrixB = new Matrix<double>(2, 2);
            matrixB[0, 0] = 1.0; matrixB[0, 1] = 2.0;
            matrixB[1, 0] = 3.0; matrixB[1, 1] = 4.0;

            // Act
            var result = matrixA - matrixB;

            // Assert
            Assert.Equal(9.0, result[0, 0], precision: 10);
            Assert.Equal(18.0, result[0, 1], precision: 10);
            Assert.Equal(27.0, result[1, 0], precision: 10);
            Assert.Equal(36.0, result[1, 1], precision: 10);
        }

        [Fact]
        public void MatrixScalarMultiplication_ProducesCorrectResult()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 2);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0;
            matrix[1, 0] = 3.0; matrix[1, 1] = 4.0;

            double scalar = 2.5;

            // Act
            var result = matrix * scalar;

            // Assert
            Assert.Equal(2.5, result[0, 0], precision: 10);
            Assert.Equal(5.0, result[0, 1], precision: 10);
            Assert.Equal(7.5, result[1, 0], precision: 10);
            Assert.Equal(10.0, result[1, 1], precision: 10);
        }

        [Fact]
        public void MatrixDeterminant_2x2_ProducesCorrectResult()
        {
            // Arrange
            // A = [[3, 8], [4, 6]]
            // det(A) = (3*6) - (8*4) = 18 - 32 = -14
            var matrix = new Matrix<double>(2, 2);
            matrix[0, 0] = 3.0; matrix[0, 1] = 8.0;
            matrix[1, 0] = 4.0; matrix[1, 1] = 6.0;

            // Act
            var determinant = matrix.Determinant();

            // Assert
            Assert.Equal(-14.0, determinant, precision: 10);
        }

        [Fact]
        public void MatrixDeterminant_3x3_ProducesCorrectResult()
        {
            // Arrange
            // A = [[6, 1, 1], [4, -2, 5], [2, 8, 7]]
            // det(A) = -306
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 6.0; matrix[0, 1] = 1.0; matrix[0, 2] = 1.0;
            matrix[1, 0] = 4.0; matrix[1, 1] = -2.0; matrix[1, 2] = 5.0;
            matrix[2, 0] = 2.0; matrix[2, 1] = 8.0; matrix[2, 2] = 7.0;

            // Act
            var determinant = matrix.Determinant();

            // Assert
            Assert.Equal(-306.0, determinant, precision: 8);
        }

        [Fact]
        public void MatrixInverse_2x2_ProducesCorrectResult()
        {
            // Arrange
            // A = [[4, 7], [2, 6]]
            // det(A) = 24 - 14 = 10
            // A^-1 = (1/10) * [[6, -7], [-2, 4]] = [[0.6, -0.7], [-0.2, 0.4]]
            var matrix = new Matrix<double>(2, 2);
            matrix[0, 0] = 4.0; matrix[0, 1] = 7.0;
            matrix[1, 0] = 2.0; matrix[1, 1] = 6.0;

            // Act
            var inverse = matrix.Inverse();

            // Assert
            Assert.Equal(0.6, inverse[0, 0], precision: 10);
            Assert.Equal(-0.7, inverse[0, 1], precision: 10);
            Assert.Equal(-0.2, inverse[1, 0], precision: 10);
            Assert.Equal(0.4, inverse[1, 1], precision: 10);

            // Verify: A * A^-1 = I
            var identity = matrix * inverse;
            Assert.Equal(1.0, identity[0, 0], precision: 10);
            Assert.Equal(0.0, identity[0, 1], precision: 10);
            Assert.Equal(0.0, identity[1, 0], precision: 10);
            Assert.Equal(1.0, identity[1, 1], precision: 10);
        }

        [Fact]
        public void MatrixInverse_MultiplyByOriginal_GivesIdentity()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = 0.0; matrix[1, 1] = 1.0; matrix[1, 2] = 4.0;
            matrix[2, 0] = 5.0; matrix[2, 1] = 6.0; matrix[2, 2] = 0.0;

            // Act
            var inverse = matrix.Inverse();
            var result = matrix * inverse;

            // Assert - Should produce identity matrix
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    double expected = (i == j) ? 1.0 : 0.0;
                    Assert.Equal(expected, result[i, j], precision: 8);
                }
            }
        }

        [Fact]
        public void MatrixTrace_ProducesCorrectResult()
        {
            // Arrange
            // A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            // trace(A) = 1 + 5 + 9 = 15
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = 4.0; matrix[1, 1] = 5.0; matrix[1, 2] = 6.0;
            matrix[2, 0] = 7.0; matrix[2, 1] = 8.0; matrix[2, 2] = 9.0;

            // Act
            var trace = matrix.Trace();

            // Assert
            Assert.Equal(15.0, trace, precision: 10);
        }

        [Fact]
        public void Matrix_ElementWiseMultiplication_ProducesCorrectResult()
        {
            // Arrange
            var matrixA = new Matrix<double>(2, 2);
            matrixA[0, 0] = 2.0; matrixA[0, 1] = 3.0;
            matrixA[1, 0] = 4.0; matrixA[1, 1] = 5.0;

            var matrixB = new Matrix<double>(2, 2);
            matrixB[0, 0] = 6.0; matrixB[0, 1] = 7.0;
            matrixB[1, 0] = 8.0; matrixB[1, 1] = 9.0;

            // Act
            var result = matrixA.ElementWiseMultiply(matrixB);

            // Assert
            Assert.Equal(12.0, result[0, 0], precision: 10);
            Assert.Equal(21.0, result[0, 1], precision: 10);
            Assert.Equal(32.0, result[1, 0], precision: 10);
            Assert.Equal(45.0, result[1, 1], precision: 10);
        }

        [Fact]
        public void Matrix_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var matrixA = new Matrix<float>(2, 2);
            matrixA[0, 0] = 1.0f; matrixA[0, 1] = 2.0f;
            matrixA[1, 0] = 3.0f; matrixA[1, 1] = 4.0f;

            var matrixB = new Matrix<float>(2, 2);
            matrixB[0, 0] = 5.0f; matrixB[0, 1] = 6.0f;
            matrixB[1, 0] = 7.0f; matrixB[1, 1] = 8.0f;

            // Act
            var result = matrixA * matrixB;

            // Assert
            Assert.Equal(19.0f, result[0, 0], precision: 6);
            Assert.Equal(22.0f, result[0, 1], precision: 6);
            Assert.Equal(43.0f, result[1, 0], precision: 6);
            Assert.Equal(50.0f, result[1, 1], precision: 6);
        }

        [Fact]
        public void Matrix_WithDecimalType_WorksCorrectly()
        {
            // Arrange
            var matrixA = new Matrix<decimal>(2, 2);
            matrixA[0, 0] = 1.5m; matrixA[0, 1] = 2.5m;
            matrixA[1, 0] = 3.5m; matrixA[1, 1] = 4.5m;

            var matrixB = new Matrix<decimal>(2, 2);
            matrixB[0, 0] = 2.0m; matrixB[0, 1] = 3.0m;
            matrixB[1, 0] = 4.0m; matrixB[1, 1] = 5.0m;

            // Act
            var result = matrixA + matrixB;

            // Assert
            Assert.Equal(3.5m, result[0, 0]);
            Assert.Equal(5.5m, result[0, 1]);
            Assert.Equal(7.5m, result[1, 0]);
            Assert.Equal(9.5m, result[1, 1]);
        }
    }
}
