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

        // ===== GetColumn Tests =====

        [Fact]
        public void GetColumn_WithValidIndex_ReturnsCorrectColumn()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = 4.0; matrix[1, 1] = 5.0; matrix[1, 2] = 6.0;
            matrix[2, 0] = 7.0; matrix[2, 1] = 8.0; matrix[2, 2] = 9.0;

            // Act
            var column = matrix.GetColumn(1);

            // Assert
            Assert.Equal(3, column.Length);
            Assert.Equal(2.0, column[0], precision: 10);
            Assert.Equal(5.0, column[1], precision: 10);
            Assert.Equal(8.0, column[2], precision: 10);
        }

        [Fact]
        public void GetRow_WithValidIndex_ReturnsCorrectRow()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = 4.0; matrix[1, 1] = 5.0; matrix[1, 2] = 6.0;
            matrix[2, 0] = 7.0; matrix[2, 1] = 8.0; matrix[2, 2] = 9.0;

            // Act
            var row = matrix.GetRow(1);

            // Assert
            Assert.Equal(3, row.Length);
            Assert.Equal(4.0, row[0], precision: 10);
            Assert.Equal(5.0, row[1], precision: 10);
            Assert.Equal(6.0, row[2], precision: 10);
        }

        [Fact]
        public void GetColumnSegment_WithValidParameters_ReturnsCorrectSegment()
        {
            // Arrange
            var matrix = new Matrix<double>(4, 3);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 3; j++)
                    matrix[i, j] = i * 3 + j + 1;

            // Act
            var segment = matrix.GetColumnSegment(1, 1, 2);

            // Assert
            Assert.Equal(2, segment.Length);
            Assert.Equal(5.0, segment[0], precision: 10); // matrix[1,1]
            Assert.Equal(8.0, segment[1], precision: 10); // matrix[2,1]
        }

        [Fact]
        public void GetRowSegment_WithValidParameters_ReturnsCorrectSegment()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 4);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                    matrix[i, j] = i * 4 + j + 1;

            // Act
            var segment = matrix.GetRowSegment(1, 1, 2);

            // Assert
            Assert.Equal(2, segment.Length);
            Assert.Equal(6.0, segment[0], precision: 10); // matrix[1,1]
            Assert.Equal(7.0, segment[1], precision: 10); // matrix[1,2]
        }

        // ===== GetSubMatrix Tests =====

        [Fact]
        public void GetSubMatrix_WithValidParameters_ReturnsCorrectSubMatrix()
        {
            // Arrange
            var matrix = new Matrix<double>(4, 4);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    matrix[i, j] = i * 4 + j + 1;

            // Act
            var subMatrix = matrix.GetSubMatrix(1, 1, 2, 2);

            // Assert
            Assert.Equal(2, subMatrix.Rows);
            Assert.Equal(2, subMatrix.Columns);
            Assert.Equal(6.0, subMatrix[0, 0], precision: 10); // matrix[1,1]
            Assert.Equal(7.0, subMatrix[0, 1], precision: 10); // matrix[1,2]
            Assert.Equal(10.0, subMatrix[1, 0], precision: 10); // matrix[2,1]
            Assert.Equal(11.0, subMatrix[1, 1], precision: 10); // matrix[2,2]
        }

        [Fact]
        public void SetSubMatrix_WithValidParameters_SetsValuesCorrectly()
        {
            // Arrange
            var matrix = new Matrix<double>(4, 4);
            var subMatrix = new Matrix<double>(2, 2);
            subMatrix[0, 0] = 10.0; subMatrix[0, 1] = 20.0;
            subMatrix[1, 0] = 30.0; subMatrix[1, 1] = 40.0;

            // Act
            matrix.SetSubMatrix(1, 1, subMatrix);

            // Assert
            Assert.Equal(10.0, matrix[1, 1], precision: 10);
            Assert.Equal(20.0, matrix[1, 2], precision: 10);
            Assert.Equal(30.0, matrix[2, 1], precision: 10);
            Assert.Equal(40.0, matrix[2, 2], precision: 10);
        }

        // ===== RemoveRow and RemoveColumn Tests =====

        [Fact]
        public void RemoveRow_WithValidIndex_RemovesRowCorrectly()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = 4.0; matrix[1, 1] = 5.0; matrix[1, 2] = 6.0;
            matrix[2, 0] = 7.0; matrix[2, 1] = 8.0; matrix[2, 2] = 9.0;

            // Act
            var result = matrix.RemoveRow(1);

            // Assert
            Assert.Equal(2, result.Rows);
            Assert.Equal(3, result.Columns);
            Assert.Equal(1.0, result[0, 0], precision: 10);
            Assert.Equal(7.0, result[1, 0], precision: 10);
        }

        [Fact]
        public void RemoveColumn_WithValidIndex_RemovesColumnCorrectly()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = 4.0; matrix[1, 1] = 5.0; matrix[1, 2] = 6.0;
            matrix[2, 0] = 7.0; matrix[2, 1] = 8.0; matrix[2, 2] = 9.0;

            // Act
            var result = matrix.RemoveColumn(1);

            // Assert
            Assert.Equal(3, result.Rows);
            Assert.Equal(2, result.Columns);
            Assert.Equal(1.0, result[0, 0], precision: 10);
            Assert.Equal(3.0, result[0, 1], precision: 10);
        }

        // ===== GetRows Tests =====

        [Fact]
        public void GetRows_WithIndices_ReturnsCorrectRows()
        {
            // Arrange
            var matrix = new Matrix<double>(4, 3);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 3; j++)
                    matrix[i, j] = i * 3 + j + 1;

            // Act
            var result = matrix.GetRows(new[] { 0, 2 });

            // Assert
            Assert.Equal(2, result.Rows);
            Assert.Equal(3, result.Columns);
            Assert.Equal(1.0, result[0, 0], precision: 10);
            Assert.Equal(7.0, result[1, 0], precision: 10);
        }

        [Fact]
        public void GetRows_Enumerable_ReturnsAllRows()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 2);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0;
            matrix[1, 0] = 3.0; matrix[1, 1] = 4.0;
            matrix[2, 0] = 5.0; matrix[2, 1] = 6.0;

            // Act
            var rows = matrix.GetRows().ToList();

            // Assert
            Assert.Equal(3, rows.Count);
            Assert.Equal(1.0, rows[0][0], precision: 10);
            Assert.Equal(2.0, rows[0][1], precision: 10);
            Assert.Equal(5.0, rows[2][0], precision: 10);
        }

        // ===== Slice Tests =====

        [Fact]
        public void Slice_WithValidParameters_ReturnsCorrectSlice()
        {
            // Arrange
            var matrix = new Matrix<double>(5, 3);
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 3; j++)
                    matrix[i, j] = i * 3 + j + 1;

            // Act
            var slice = matrix.Slice(1, 3);

            // Assert
            Assert.Equal(3, slice.Rows);
            Assert.Equal(3, slice.Columns);
            Assert.Equal(4.0, slice[0, 0], precision: 10);
            Assert.Equal(13.0, slice[2, 0], precision: 10);
        }

        // ===== Transform Tests =====

        [Fact]
        public void Transform_WithFunction_TransformsAllElements()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 2);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0;
            matrix[1, 0] = 3.0; matrix[1, 1] = 4.0;

            // Act - Double each element
            var result = matrix.Transform((val, i, j) => val * 2.0);

            // Assert
            Assert.Equal(2.0, result[0, 0], precision: 10);
            Assert.Equal(4.0, result[0, 1], precision: 10);
            Assert.Equal(6.0, result[1, 0], precision: 10);
            Assert.Equal(8.0, result[1, 1], precision: 10);
        }

        // ===== PointwiseDivide Tests =====

        [Fact]
        public void PointwiseDivide_WithValidMatrix_DividesElementWise()
        {
            // Arrange
            var matrixA = new Matrix<double>(2, 2);
            matrixA[0, 0] = 10.0; matrixA[0, 1] = 20.0;
            matrixA[1, 0] = 30.0; matrixA[1, 1] = 40.0;

            var matrixB = new Matrix<double>(2, 2);
            matrixB[0, 0] = 2.0; matrixB[0, 1] = 4.0;
            matrixB[1, 0] = 5.0; matrixB[1, 1] = 8.0;

            // Act
            var result = matrixA.PointwiseDivide(matrixB);

            // Assert
            Assert.Equal(5.0, result[0, 0], precision: 10);
            Assert.Equal(5.0, result[0, 1], precision: 10);
            Assert.Equal(6.0, result[1, 0], precision: 10);
            Assert.Equal(5.0, result[1, 1], precision: 10);
        }

        [Fact]
        public void Divide_ScalarDivision_DividesAllElements()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 2);
            matrix[0, 0] = 10.0; matrix[0, 1] = 20.0;
            matrix[1, 0] = 30.0; matrix[1, 1] = 40.0;

            // Act
            var result = matrix / 5.0;

            // Assert
            Assert.Equal(2.0, result[0, 0], precision: 10);
            Assert.Equal(4.0, result[0, 1], precision: 10);
            Assert.Equal(6.0, result[1, 0], precision: 10);
            Assert.Equal(8.0, result[1, 1], precision: 10);
        }

        [Fact]
        public void Divide_MatrixDivision_DividesElementWise()
        {
            // Arrange
            var matrixA = new Matrix<double>(2, 2);
            matrixA[0, 0] = 12.0; matrixA[0, 1] = 15.0;
            matrixA[1, 0] = 18.0; matrixA[1, 1] = 21.0;

            var matrixB = new Matrix<double>(2, 2);
            matrixB[0, 0] = 3.0; matrixB[0, 1] = 5.0;
            matrixB[1, 0] = 6.0; matrixB[1, 1] = 7.0;

            // Act
            var result = matrixA / matrixB;

            // Assert
            Assert.Equal(4.0, result[0, 0], precision: 10);
            Assert.Equal(3.0, result[0, 1], precision: 10);
            Assert.Equal(3.0, result[1, 0], precision: 10);
            Assert.Equal(3.0, result[1, 1], precision: 10);
        }

        // ===== OuterProduct Tests =====

        [Fact]
        public void OuterProduct_TwoVectors_ProducesCorrectMatrix()
        {
            // Arrange
            var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var v2 = new Vector<double>(new[] { 4.0, 5.0 });

            // Act
            var result = Matrix<double>.OuterProduct(v1, v2);

            // Assert
            Assert.Equal(3, result.Rows);
            Assert.Equal(2, result.Columns);
            Assert.Equal(4.0, result[0, 0], precision: 10); // 1*4
            Assert.Equal(5.0, result[0, 1], precision: 10); // 1*5
            Assert.Equal(8.0, result[1, 0], precision: 10); // 2*4
            Assert.Equal(10.0, result[1, 1], precision: 10); // 2*5
            Assert.Equal(12.0, result[2, 0], precision: 10); // 3*4
            Assert.Equal(15.0, result[2, 1], precision: 10); // 3*5
        }

        // ===== Static Factory Method Tests =====

        [Fact]
        public void CreateOnes_ProducesMatrixOfOnes()
        {
            // Act
            var matrix = Matrix<double>.CreateOnes(3, 2);

            // Assert
            Assert.Equal(3, matrix.Rows);
            Assert.Equal(2, matrix.Columns);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 2; j++)
                    Assert.Equal(1.0, matrix[i, j], precision: 10);
        }

        [Fact]
        public void CreateZeros_ProducesMatrixOfZeros()
        {
            // Act
            var matrix = Matrix<double>.CreateZeros(3, 2);

            // Assert
            Assert.Equal(3, matrix.Rows);
            Assert.Equal(2, matrix.Columns);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 2; j++)
                    Assert.Equal(0.0, matrix[i, j], precision: 10);
        }

        [Fact]
        public void CreateDiagonal_WithVector_CreatesDiagonalMatrix()
        {
            // Arrange
            var diagonal = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var matrix = Matrix<double>.CreateDiagonal(diagonal);

            // Assert
            Assert.Equal(3, matrix.Rows);
            Assert.Equal(3, matrix.Columns);
            Assert.Equal(1.0, matrix[0, 0], precision: 10);
            Assert.Equal(2.0, matrix[1, 1], precision: 10);
            Assert.Equal(3.0, matrix[2, 2], precision: 10);
            Assert.Equal(0.0, matrix[0, 1], precision: 10);
            Assert.Equal(0.0, matrix[1, 0], precision: 10);
        }

        [Fact]
        public void CreateIdentity_ProducesIdentityMatrix()
        {
            // Act
            var matrix = Matrix<double>.CreateIdentity(3);

            // Assert
            Assert.Equal(3, matrix.Rows);
            Assert.Equal(3, matrix.Columns);
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    double expected = (i == j) ? 1.0 : 0.0;
                    Assert.Equal(expected, matrix[i, j], precision: 10);
                }
            }
        }

        [Fact]
        public void CreateRandom_ProducesRandomMatrix()
        {
            // Act
            var matrix = Matrix<double>.CreateRandom(3, 3);

            // Assert
            Assert.Equal(3, matrix.Rows);
            Assert.Equal(3, matrix.Columns);
            // Check that at least some values are non-zero (probabilistic)
            bool hasNonZero = false;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    if (matrix[i, j] != 0.0)
                        hasNonZero = true;
            Assert.True(hasNonZero);
        }

        [Fact]
        public void CreateRandom_WithRange_ProducesValuesInRange()
        {
            // Act
            var matrix = Matrix<double>.CreateRandom(5, 5, -2.0, 2.0);

            // Assert
            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    Assert.True(matrix[i, j] >= -2.0);
                    Assert.True(matrix[i, j] <= 2.0);
                }
            }
        }

        [Fact]
        public void CreateDefault_ProducesMatrixWithDefaultValue()
        {
            // Act
            var matrix = Matrix<double>.CreateDefault(3, 2, 7.5);

            // Assert
            Assert.Equal(3, matrix.Rows);
            Assert.Equal(2, matrix.Columns);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 2; j++)
                    Assert.Equal(7.5, matrix[i, j], precision: 10);
        }

        [Fact]
        public void BlockDiagonal_WithMultipleMatrices_CreatesBlockDiagonalMatrix()
        {
            // Arrange
            var m1 = new Matrix<double>(2, 2);
            m1[0, 0] = 1.0; m1[0, 1] = 2.0;
            m1[1, 0] = 3.0; m1[1, 1] = 4.0;

            var m2 = new Matrix<double>(1, 1);
            m2[0, 0] = 5.0;

            // Act
            var result = Matrix<double>.BlockDiagonal(m1, m2);

            // Assert
            Assert.Equal(3, result.Rows);
            Assert.Equal(3, result.Columns);
            Assert.Equal(1.0, result[0, 0], precision: 10);
            Assert.Equal(4.0, result[1, 1], precision: 10);
            Assert.Equal(5.0, result[2, 2], precision: 10);
            Assert.Equal(0.0, result[0, 2], precision: 10);
            Assert.Equal(0.0, result[2, 0], precision: 10);
        }

        // ===== FromVector, FromRows, FromColumns Tests =====

        [Fact]
        public void FromVector_CreatesMatrixFromVector()
        {
            // Arrange
            var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var matrix = Matrix<double>.FromVector(vector);

            // Assert
            Assert.Equal(3, matrix.Rows);
            Assert.Equal(1, matrix.Columns);
            Assert.Equal(1.0, matrix[0, 0], precision: 10);
            Assert.Equal(2.0, matrix[1, 0], precision: 10);
            Assert.Equal(3.0, matrix[2, 0], precision: 10);
        }

        [Fact]
        public void CreateFromVector_CreatesRowMatrixFromVector()
        {
            // Arrange
            var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var matrix = Matrix<double>.CreateFromVector(vector);

            // Assert
            Assert.Equal(1, matrix.Rows);
            Assert.Equal(3, matrix.Columns);
            Assert.Equal(1.0, matrix[0, 0], precision: 10);
            Assert.Equal(2.0, matrix[0, 1], precision: 10);
            Assert.Equal(3.0, matrix[0, 2], precision: 10);
        }

        [Fact]
        public void FromRows_CreatesMatrixFromRowVectors()
        {
            // Arrange
            var row1 = new[] { 1.0, 2.0, 3.0 };
            var row2 = new[] { 4.0, 5.0, 6.0 };

            // Act
            var matrix = Matrix<double>.FromRows(row1, row2);

            // Assert
            Assert.Equal(2, matrix.Rows);
            Assert.Equal(3, matrix.Columns);
            Assert.Equal(1.0, matrix[0, 0], precision: 10);
            Assert.Equal(6.0, matrix[1, 2], precision: 10);
        }

        [Fact]
        public void FromColumns_CreatesMatrixFromColumnVectors()
        {
            // Arrange
            var col1 = new[] { 1.0, 2.0 };
            var col2 = new[] { 3.0, 4.0 };
            var col3 = new[] { 5.0, 6.0 };

            // Act
            var matrix = Matrix<double>.FromColumns(col1, col2, col3);

            // Assert
            Assert.Equal(2, matrix.Rows);
            Assert.Equal(3, matrix.Columns);
            Assert.Equal(1.0, matrix[0, 0], precision: 10);
            Assert.Equal(3.0, matrix[0, 1], precision: 10);
            Assert.Equal(6.0, matrix[1, 2], precision: 10);
        }

        [Fact]
        public void FromRowVectors_WithIEnumerable_CreatesMatrix()
        {
            // Arrange
            var rows = new List<IEnumerable<double>>
            {
                new[] { 1.0, 2.0 },
                new[] { 3.0, 4.0 },
                new[] { 5.0, 6.0 }
            };

            // Act
            var matrix = Matrix<double>.FromRowVectors(rows);

            // Assert
            Assert.Equal(3, matrix.Rows);
            Assert.Equal(2, matrix.Columns);
            Assert.Equal(1.0, matrix[0, 0], precision: 10);
            Assert.Equal(6.0, matrix[2, 1], precision: 10);
        }

        [Fact]
        public void FromColumnVectors_WithIEnumerable_CreatesMatrix()
        {
            // Arrange
            var columns = new List<IEnumerable<double>>
            {
                new[] { 1.0, 2.0, 3.0 },
                new[] { 4.0, 5.0, 6.0 }
            };

            // Act
            var matrix = Matrix<double>.FromColumnVectors(columns);

            // Assert
            Assert.Equal(3, matrix.Rows);
            Assert.Equal(2, matrix.Columns);
            Assert.Equal(1.0, matrix[0, 0], precision: 10);
            Assert.Equal(4.0, matrix[0, 1], precision: 10);
            Assert.Equal(6.0, matrix[2, 1], precision: 10);
        }

        // ===== Matrix * Vector Tests =====

        [Fact]
        public void MatrixVectorMultiplication_ProducesCorrectResult()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = 4.0; matrix[1, 1] = 5.0; matrix[1, 2] = 6.0;

            var vector = new Vector<double>(new[] { 2.0, 3.0, 4.0 });

            // Act
            var result = matrix * vector;

            // Assert
            Assert.Equal(2, result.Length);
            Assert.Equal(20.0, result[0], precision: 10); // 1*2 + 2*3 + 3*4 = 20
            Assert.Equal(47.0, result[1], precision: 10); // 4*2 + 5*3 + 6*4 = 47
        }

        // ===== ToRowVector and ToColumnVector Tests =====

        [Fact]
        public void ToRowVector_FlattensMatrixByRows()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = 4.0; matrix[1, 1] = 5.0; matrix[1, 2] = 6.0;

            // Act
            var vector = matrix.ToRowVector();

            // Assert
            Assert.Equal(6, vector.Length);
            Assert.Equal(1.0, vector[0], precision: 10);
            Assert.Equal(2.0, vector[1], precision: 10);
            Assert.Equal(3.0, vector[2], precision: 10);
            Assert.Equal(4.0, vector[3], precision: 10);
            Assert.Equal(6.0, vector[5], precision: 10);
        }

        [Fact]
        public void ToColumnVector_FlattensMatrixByColumns()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = 4.0; matrix[1, 1] = 5.0; matrix[1, 2] = 6.0;

            // Act
            var vector = matrix.ToColumnVector();

            // Assert
            Assert.Equal(6, vector.Length);
            Assert.Equal(1.0, vector[0], precision: 10);
            Assert.Equal(4.0, vector[1], precision: 10);
            Assert.Equal(2.0, vector[2], precision: 10);
            Assert.Equal(5.0, vector[3], precision: 10);
        }

        // ===== RowWiseSum and RowWiseMax Tests =====

        [Fact]
        public void RowWiseSum_CalculatesCorrectSums()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = 4.0; matrix[1, 1] = 5.0; matrix[1, 2] = 6.0;
            matrix[2, 0] = 7.0; matrix[2, 1] = 8.0; matrix[2, 2] = 9.0;

            // Act
            var sums = matrix.RowWiseSum();

            // Assert
            Assert.Equal(3, sums.Length);
            Assert.Equal(6.0, sums[0], precision: 10);
            Assert.Equal(15.0, sums[1], precision: 10);
            Assert.Equal(24.0, sums[2], precision: 10);
        }

        [Fact]
        public void RowWiseMax_FindsMaximumInEachRow()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 5.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = 9.0; matrix[1, 1] = 2.0; matrix[1, 2] = 6.0;
            matrix[2, 0] = 4.0; matrix[2, 1] = 8.0; matrix[2, 2] = 7.0;

            // Act
            var maxValues = matrix.RowWiseMax();

            // Assert
            Assert.Equal(3, maxValues.Length);
            Assert.Equal(5.0, maxValues[0], precision: 10);
            Assert.Equal(9.0, maxValues[1], precision: 10);
            Assert.Equal(8.0, maxValues[2], precision: 10);
        }

        // ===== Clone Test =====

        [Fact]
        public void Clone_CreatesIndependentCopy()
        {
            // Arrange
            var original = new Matrix<double>(2, 2);
            original[0, 0] = 1.0; original[0, 1] = 2.0;
            original[1, 0] = 3.0; original[1, 1] = 4.0;

            // Act
            var clone = original.Clone();
            clone[0, 0] = 99.0;

            // Assert
            Assert.Equal(1.0, original[0, 0], precision: 10);
            Assert.Equal(99.0, clone[0, 0], precision: 10);
        }

        // ===== Edge Case Tests =====

        [Fact]
        public void Matrix_1x1_WorksCorrectly()
        {
            // Arrange
            var matrix = new Matrix<double>(1, 1);
            matrix[0, 0] = 5.0;

            // Act
            var determinant = matrix.Determinant();
            var transpose = matrix.Transpose();

            // Assert
            Assert.Equal(5.0, determinant, precision: 10);
            Assert.Equal(5.0, transpose[0, 0], precision: 10);
        }

        [Fact]
        public void Matrix_Empty_HandlesCorrectly()
        {
            // Act
            var matrix = Matrix<double>.Empty();

            // Assert
            Assert.Equal(0, matrix.Rows);
            Assert.Equal(0, matrix.Columns);
        }

        [Fact]
        public void Matrix_NonSquare_3x5_WorksCorrectly()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 5);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 5; j++)
                    matrix[i, j] = i * 5 + j + 1;

            // Act
            var transpose = matrix.Transpose();

            // Assert
            Assert.Equal(5, transpose.Rows);
            Assert.Equal(3, transpose.Columns);
            Assert.Equal(1.0, transpose[0, 0], precision: 10);
            Assert.Equal(15.0, transpose[4, 2], precision: 10);
        }

        [Fact]
        public void Matrix_NonSquare_5x3_WorksCorrectly()
        {
            // Arrange
            var matrix = new Matrix<double>(5, 3);
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 3; j++)
                    matrix[i, j] = i * 3 + j + 1;

            // Act
            var transpose = matrix.Transpose();

            // Assert
            Assert.Equal(3, transpose.Rows);
            Assert.Equal(5, transpose.Columns);
        }

        [Fact]
        public void Matrix_AllZeros_WorksCorrectly()
        {
            // Arrange
            var matrix = Matrix<double>.CreateZeros(3, 3);

            // Act
            var trace = matrix.Trace();
            var transpose = matrix.Transpose();

            // Assert
            Assert.Equal(0.0, trace, precision: 10);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    Assert.Equal(0.0, transpose[i, j], precision: 10);
        }

        [Fact]
        public void Matrix_WithNegativeValues_WorksCorrectly()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 2);
            matrix[0, 0] = -1.0; matrix[0, 1] = -2.0;
            matrix[1, 0] = -3.0; matrix[1, 1] = -4.0;

            // Act
            var result = matrix * 2.0;

            // Assert
            Assert.Equal(-2.0, result[0, 0], precision: 10);
            Assert.Equal(-8.0, result[1, 1], precision: 10);
        }

        [Fact]
        public void Matrix_WithVeryLargeValues_MaintainsNumericalStability()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 2);
            matrix[0, 0] = 1e10; matrix[0, 1] = 2e10;
            matrix[1, 0] = 3e10; matrix[1, 1] = 4e10;

            // Act
            var result = matrix + matrix;

            // Assert
            Assert.Equal(2e10, result[0, 0], precision: 5);
            Assert.Equal(8e10, result[1, 1], precision: 5);
        }

        [Fact]
        public void Matrix_WithVerySmallValues_MaintainsNumericalStability()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 2);
            matrix[0, 0] = 1e-10; matrix[0, 1] = 2e-10;
            matrix[1, 0] = 3e-10; matrix[1, 1] = 4e-10;

            // Act
            var result = matrix * 2.0;

            // Assert
            Assert.Equal(2e-10, result[0, 0], precision: 15);
            Assert.Equal(8e-10, result[1, 1], precision: 15);
        }

        [Fact]
        public void Matrix_WithIntType_WorksCorrectly()
        {
            // Arrange
            var matrixA = new Matrix<int>(2, 2);
            matrixA[0, 0] = 1; matrixA[0, 1] = 2;
            matrixA[1, 0] = 3; matrixA[1, 1] = 4;

            var matrixB = new Matrix<int>(2, 2);
            matrixB[0, 0] = 5; matrixB[0, 1] = 6;
            matrixB[1, 0] = 7; matrixB[1, 1] = 8;

            // Act
            var result = matrixA + matrixB;

            // Assert
            Assert.Equal(6, result[0, 0]);
            Assert.Equal(12, result[1, 1]);
        }

        [Fact]
        public void Matrix_SparseMatrix_100x100_WorksCorrectly()
        {
            // Arrange - Create sparse matrix (mostly zeros)
            var matrix = Matrix<double>.CreateZeros(100, 100);
            matrix[0, 0] = 1.0;
            matrix[50, 50] = 2.0;
            matrix[99, 99] = 3.0;

            // Act
            var trace = matrix.Trace();

            // Assert
            Assert.Equal(6.0, trace, precision: 10);
        }

        [Fact]
        public void Matrix_SymmetricMatrix_PreservesSymmetry()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = 2.0; matrix[1, 1] = 4.0; matrix[1, 2] = 5.0;
            matrix[2, 0] = 3.0; matrix[2, 1] = 5.0; matrix[2, 2] = 6.0;

            // Act
            var transpose = matrix.Transpose();

            // Assert - Symmetric matrix equals its transpose
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    Assert.Equal(matrix[i, j], transpose[i, j], precision: 10);
        }

        [Fact]
        public void Matrix_DiagonalMatrix_PreservesDiagonalProperty()
        {
            // Arrange
            var diagonal = new Vector<double>(new[] { 2.0, 3.0, 4.0 });
            var matrix = Matrix<double>.CreateDiagonal(diagonal);

            // Act
            var squared = matrix * matrix;

            // Assert - Squaring diagonal matrix squares diagonal elements
            Assert.Equal(4.0, squared[0, 0], precision: 10);
            Assert.Equal(9.0, squared[1, 1], precision: 10);
            Assert.Equal(16.0, squared[2, 2], precision: 10);
            Assert.Equal(0.0, squared[0, 1], precision: 10);
        }

        // ===== MatrixHelper Tests =====

        [Fact]
        public void CalculateDeterminantRecursive_2x2_ProducesCorrectResult()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 2);
            matrix[0, 0] = 5.0; matrix[0, 1] = 6.0;
            matrix[1, 0] = 7.0; matrix[1, 1] = 8.0;

            // Act
            var det = AiDotNet.Helpers.MatrixHelper<double>.CalculateDeterminantRecursive(matrix);

            // Assert
            Assert.Equal(-2.0, det, precision: 10);
        }

        [Fact]
        public void CalculateDeterminantRecursive_3x3_ProducesCorrectResult()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = 4.0; matrix[1, 1] = 5.0; matrix[1, 2] = 6.0;
            matrix[2, 0] = 7.0; matrix[2, 1] = 8.0; matrix[2, 2] = 10.0;

            // Act
            var det = AiDotNet.Helpers.MatrixHelper<double>.CalculateDeterminantRecursive(matrix);

            // Assert
            Assert.Equal(-3.0, det, precision: 10);
        }

        [Fact]
        public void CalculateDeterminantRecursive_4x4_ProducesCorrectResult()
        {
            // Arrange
            var matrix = new Matrix<double>(4, 4);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0; matrix[0, 3] = 4.0;
            matrix[1, 0] = 5.0; matrix[1, 1] = 6.0; matrix[1, 2] = 7.0; matrix[1, 3] = 8.0;
            matrix[2, 0] = 9.0; matrix[2, 1] = 10.0; matrix[2, 2] = 11.0; matrix[2, 3] = 12.0;
            matrix[3, 0] = 13.0; matrix[3, 1] = 14.0; matrix[3, 2] = 15.0; matrix[3, 3] = 16.0;

            // Act
            var det = AiDotNet.Helpers.MatrixHelper<double>.CalculateDeterminantRecursive(matrix);

            // Assert
            Assert.Equal(0.0, det, precision: 10);
        }

        [Fact]
        public void CalculateDeterminantRecursive_5x5_ProducesCorrectResult()
        {
            // Arrange - Create a 5x5 matrix with known determinant
            var matrix = new Matrix<double>(5, 5);
            // Upper triangular matrix - determinant is product of diagonal
            for (int i = 0; i < 5; i++)
            {
                matrix[i, i] = i + 1.0;
                for (int j = i + 1; j < 5; j++)
                    matrix[i, j] = 1.0;
            }

            // Act
            var det = AiDotNet.Helpers.MatrixHelper<double>.CalculateDeterminantRecursive(matrix);

            // Assert - Product of diagonal: 1*2*3*4*5 = 120
            Assert.Equal(120.0, det, precision: 8);
        }

        [Fact]
        public void ExtractDiagonal_ExtractsDiagonalCorrectly()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = 4.0; matrix[1, 1] = 5.0; matrix[1, 2] = 6.0;
            matrix[2, 0] = 7.0; matrix[2, 1] = 8.0; matrix[2, 2] = 9.0;

            // Act
            var diagonal = AiDotNet.Helpers.MatrixHelper<double>.ExtractDiagonal(matrix);

            // Assert
            Assert.Equal(3, diagonal.Length);
            Assert.Equal(1.0, diagonal[0], precision: 10);
            Assert.Equal(5.0, diagonal[1], precision: 10);
            Assert.Equal(9.0, diagonal[2], precision: 10);
        }

        [Fact]
        public void OrthogonalizeColumns_ProducesOrthogonalColumns()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 2);
            matrix[0, 0] = 1.0; matrix[0, 1] = 1.0;
            matrix[1, 0] = 1.0; matrix[1, 1] = 0.0;
            matrix[2, 0] = 0.0; matrix[2, 1] = 1.0;

            // Act
            var orthogonal = AiDotNet.Helpers.MatrixHelper<double>.OrthogonalizeColumns(matrix);

            // Assert - Check orthogonality: column1 · column2 = 0
            var col1 = orthogonal.GetColumn(0);
            var col2 = orthogonal.GetColumn(1);
            var dotProduct = col1.DotProduct(col2);
            Assert.Equal(0.0, dotProduct, precision: 10);
        }

        [Fact]
        public void ComputeGivensRotation_WithNonZeroB_ComputesCorrectly()
        {
            // Act
            var (c, s) = AiDotNet.Helpers.MatrixHelper<double>.ComputeGivensRotation(3.0, 4.0);

            // Assert
            Assert.True(Math.Abs(c) <= 1.0);
            Assert.True(Math.Abs(s) <= 1.0);
        }

        [Fact]
        public void ComputeGivensRotation_WithZeroB_ReturnsCorrectValues()
        {
            // Act
            var (c, s) = AiDotNet.Helpers.MatrixHelper<double>.ComputeGivensRotation(5.0, 0.0);

            // Assert
            Assert.Equal(1.0, c, precision: 10);
            Assert.Equal(0.0, s, precision: 10);
        }

        [Fact]
        public void IsInvertible_WithInvertibleMatrix_ReturnsTrue()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 2);
            matrix[0, 0] = 4.0; matrix[0, 1] = 7.0;
            matrix[1, 0] = 2.0; matrix[1, 1] = 6.0;

            // Act
            var isInvertible = AiDotNet.Helpers.MatrixHelper<double>.IsInvertible(matrix);

            // Assert
            Assert.True(isInvertible);
        }

        [Fact]
        public void IsInvertible_WithSingularMatrix_ReturnsFalse()
        {
            // Arrange - Singular matrix (det = 0)
            var matrix = new Matrix<double>(2, 2);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0;
            matrix[1, 0] = 2.0; matrix[1, 1] = 4.0;

            // Act
            var isInvertible = AiDotNet.Helpers.MatrixHelper<double>.IsInvertible(matrix);

            // Assert
            Assert.False(isInvertible);
        }

        [Fact]
        public void IsInvertible_WithNonSquareMatrix_ReturnsFalse()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 3);

            // Act
            var isInvertible = AiDotNet.Helpers.MatrixHelper<double>.IsInvertible(matrix);

            // Assert
            Assert.False(isInvertible);
        }

        // ===== MatrixExtensions Tests =====

        [Fact]
        public void AddConstantColumn_AddsColumnAtBeginning()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 2);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0;
            matrix[1, 0] = 3.0; matrix[1, 1] = 4.0;

            // Act
            var result = matrix.AddConstantColumn(5.0);

            // Assert
            Assert.Equal(2, result.Rows);
            Assert.Equal(3, result.Columns);
            Assert.Equal(5.0, result[0, 0], precision: 10);
            Assert.Equal(5.0, result[1, 0], precision: 10);
            Assert.Equal(1.0, result[0, 1], precision: 10);
        }

        [Fact]
        public void ToVector_FlattensMatrixCorrectly()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = 4.0; matrix[1, 1] = 5.0; matrix[1, 2] = 6.0;

            // Act
            var vector = matrix.ToVector();

            // Assert
            Assert.Equal(6, vector.Length);
            Assert.Equal(1.0, vector[0], precision: 10);
            Assert.Equal(6.0, vector[5], precision: 10);
        }

        [Fact]
        public void AddVectorToEachRow_AddsCorrectly()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = 4.0; matrix[1, 1] = 5.0; matrix[1, 2] = 6.0;

            var vector = new Vector<double>(new[] { 10.0, 20.0, 30.0 });

            // Act
            var result = matrix.AddVectorToEachRow(vector);

            // Assert
            Assert.Equal(11.0, result[0, 0], precision: 10);
            Assert.Equal(22.0, result[0, 1], precision: 10);
            Assert.Equal(36.0, result[1, 2], precision: 10);
        }

        [Fact]
        public void SumColumns_CalculatesCorrectSums()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = 4.0; matrix[1, 1] = 5.0; matrix[1, 2] = 6.0;
            matrix[2, 0] = 7.0; matrix[2, 1] = 8.0; matrix[2, 2] = 9.0;

            // Act
            var sums = matrix.SumColumns();

            // Assert
            Assert.Equal(3, sums.Length);
            Assert.Equal(12.0, sums[0], precision: 10); // 1+4+7
            Assert.Equal(15.0, sums[1], precision: 10); // 2+5+8
            Assert.Equal(18.0, sums[2], precision: 10); // 3+6+9
        }

        [Fact]
        public void BackwardSubstitution_SolvesUpperTriangularSystem()
        {
            // Arrange - Upper triangular matrix
            var A = new Matrix<double>(3, 3);
            A[0, 0] = 2.0; A[0, 1] = 1.0; A[0, 2] = 1.0;
            A[1, 0] = 0.0; A[1, 1] = 3.0; A[1, 2] = 1.0;
            A[2, 0] = 0.0; A[2, 1] = 0.0; A[2, 2] = 4.0;

            var b = new Vector<double>(new[] { 6.0, 7.0, 8.0 });

            // Act
            var x = A.BackwardSubstitution(b);

            // Assert - Verify Ax = b
            var result = A.Multiply(x);
            Assert.Equal(6.0, result[0], precision: 10);
            Assert.Equal(7.0, result[1], precision: 10);
            Assert.Equal(8.0, result[2], precision: 10);
        }

        [Fact]
        public void IsSquareMatrix_WithSquareMatrix_ReturnsTrue()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 3);

            // Act
            var isSquare = matrix.IsSquareMatrix();

            // Assert
            Assert.True(isSquare);
        }

        [Fact]
        public void IsSquareMatrix_WithNonSquareMatrix_ReturnsFalse()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 4);

            // Act
            var isSquare = matrix.IsSquareMatrix();

            // Assert
            Assert.False(isSquare);
        }

        [Fact]
        public void IsRectangularMatrix_WithNonSquareMatrix_ReturnsTrue()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 5);

            // Act
            var isRectangular = matrix.IsRectangularMatrix();

            // Assert
            Assert.True(isRectangular);
        }

        [Fact]
        public void IsSymmetricMatrix_WithSymmetricMatrix_ReturnsTrue()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = 2.0; matrix[1, 1] = 4.0; matrix[1, 2] = 5.0;
            matrix[2, 0] = 3.0; matrix[2, 1] = 5.0; matrix[2, 2] = 6.0;

            // Act
            var isSymmetric = matrix.IsSymmetricMatrix();

            // Assert
            Assert.True(isSymmetric);
        }

        [Fact]
        public void IsSymmetricMatrix_WithNonSymmetricMatrix_ReturnsFalse()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = 4.0; matrix[1, 1] = 5.0; matrix[1, 2] = 6.0;
            matrix[2, 0] = 7.0; matrix[2, 1] = 8.0; matrix[2, 2] = 9.0;

            // Act
            var isSymmetric = matrix.IsSymmetricMatrix();

            // Assert
            Assert.False(isSymmetric);
        }

        [Fact]
        public void IsDiagonalMatrix_WithDiagonalMatrix_ReturnsTrue()
        {
            // Arrange
            var matrix = Matrix<double>.CreateDiagonal(new Vector<double>(new[] { 1.0, 2.0, 3.0 }));

            // Act
            var isDiagonal = matrix.IsDiagonalMatrix();

            // Assert
            Assert.True(isDiagonal);
        }

        [Fact]
        public void IsDiagonalMatrix_WithNonDiagonalMatrix_ReturnsFalse()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0;
            matrix[1, 1] = 5.0;

            // Act
            var isDiagonal = matrix.IsDiagonalMatrix();

            // Assert
            Assert.False(isDiagonal);
        }

        [Fact]
        public void IsIdentityMatrix_WithIdentityMatrix_ReturnsTrue()
        {
            // Arrange
            var matrix = Matrix<double>.CreateIdentity(3);

            // Act
            var isIdentity = matrix.IsIdentityMatrix();

            // Assert
            Assert.True(isIdentity);
        }

        [Fact]
        public void IsIdentityMatrix_WithNonIdentityMatrix_ReturnsFalse()
        {
            // Arrange
            var matrix = Matrix<double>.CreateOnes(3, 3);

            // Act
            var isIdentity = matrix.IsIdentityMatrix();

            // Assert
            Assert.False(isIdentity);
        }

        [Fact]
        public void IsUpperTriangularMatrix_WithUpperTriangular_ReturnsTrue()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 1] = 4.0; matrix[1, 2] = 5.0;
            matrix[2, 2] = 6.0;

            // Act
            var isUpperTriangular = matrix.IsUpperTriangularMatrix();

            // Assert
            Assert.True(isUpperTriangular);
        }

        [Fact]
        public void IsLowerTriangularMatrix_WithLowerTriangular_ReturnsTrue()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 1.0;
            matrix[1, 0] = 2.0; matrix[1, 1] = 3.0;
            matrix[2, 0] = 4.0; matrix[2, 1] = 5.0; matrix[2, 2] = 6.0;

            // Act
            var isLowerTriangular = matrix.IsLowerTriangularMatrix();

            // Assert
            Assert.True(isLowerTriangular);
        }

        [Fact]
        public void IsSkewSymmetricMatrix_WithSkewSymmetric_ReturnsTrue()
        {
            // Arrange - Skew symmetric: A^T = -A
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = -2.0; matrix[1, 2] = 4.0;
            matrix[2, 0] = -3.0; matrix[2, 1] = -4.0;

            // Act
            var isSkewSymmetric = matrix.IsSkewSymmetricMatrix();

            // Assert
            Assert.True(isSkewSymmetric);
        }

        [Fact]
        public void IsScalarMatrix_WithScalarMatrix_ReturnsTrue()
        {
            // Arrange - Scalar matrix has same value on diagonal, zeros elsewhere
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 5.0;
            matrix[1, 1] = 5.0;
            matrix[2, 2] = 5.0;

            // Act
            var isScalar = matrix.IsScalarMatrix();

            // Assert
            Assert.True(isScalar);
        }

        [Fact]
        public void IsUpperBidiagonalMatrix_WithUpperBidiagonal_ReturnsTrue()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0;
            matrix[1, 1] = 3.0; matrix[1, 2] = 4.0;
            matrix[2, 2] = 5.0;

            // Act
            var isUpperBidiagonal = matrix.IsUpperBidiagonalMatrix();

            // Assert
            Assert.True(isUpperBidiagonal);
        }

        [Fact]
        public void IsLowerBidiagonalMatrix_WithLowerBidiagonal_ReturnsTrue()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 1.0;
            matrix[1, 0] = 2.0; matrix[1, 1] = 3.0;
            matrix[2, 1] = 4.0; matrix[2, 2] = 5.0;

            // Act
            var isLowerBidiagonal = matrix.IsLowerBidiagonalMatrix();

            // Assert
            Assert.True(isLowerBidiagonal);
        }

        [Fact]
        public void IsTridiagonalMatrix_WithTridiagonal_ReturnsTrue()
        {
            // Arrange
            var matrix = new Matrix<double>(4, 4);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0;
            matrix[1, 0] = 3.0; matrix[1, 1] = 4.0; matrix[1, 2] = 5.0;
            matrix[2, 1] = 6.0; matrix[2, 2] = 7.0; matrix[2, 3] = 8.0;
            matrix[3, 2] = 9.0; matrix[3, 3] = 10.0;

            // Act
            var isTridiagonal = matrix.IsTridiagonalMatrix();

            // Assert
            Assert.True(isTridiagonal);
        }

        [Fact]
        public void IsSingularMatrix_WithSingularMatrix_ReturnsTrue()
        {
            // Arrange - Singular matrix has determinant = 0
            var matrix = new Matrix<double>(2, 2);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0;
            matrix[1, 0] = 2.0; matrix[1, 1] = 4.0;

            // Act
            var isSingular = matrix.IsSingularMatrix();

            // Assert
            Assert.True(isSingular);
        }

        [Fact]
        public void IsNonSingularMatrix_WithNonSingularMatrix_ReturnsTrue()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 2);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0;
            matrix[1, 0] = 3.0; matrix[1, 1] = 4.0;

            // Act
            var isNonSingular = matrix.IsNonSingularMatrix();

            // Assert
            Assert.True(isNonSingular);
        }

        [Fact]
        public void IsIdempotentMatrix_WithIdempotentMatrix_ReturnsTrue()
        {
            // Arrange - Idempotent: A*A = A
            var matrix = new Matrix<double>(2, 2);
            matrix[0, 0] = 1.0; matrix[0, 1] = 0.0;
            matrix[1, 0] = 0.0; matrix[1, 1] = 0.0;

            // Act
            var isIdempotent = matrix.IsIdempotentMatrix();

            // Assert
            Assert.True(isIdempotent);
        }

        [Fact]
        public void IsStochasticMatrix_WithStochasticMatrix_ReturnsTrue()
        {
            // Arrange - Stochastic matrix: each row sums to 1
            var matrix = new Matrix<double>(2, 2);
            matrix[0, 0] = 0.3; matrix[0, 1] = 0.7;
            matrix[1, 0] = 0.4; matrix[1, 1] = 0.6;

            // Act
            var isStochastic = matrix.IsStochasticMatrix();

            // Assert
            Assert.True(isStochastic);
        }

        [Fact]
        public void IsDoublyStochasticMatrix_WithDoublyStochastic_ReturnsTrue()
        {
            // Arrange - Doubly stochastic: rows and columns sum to 1
            var matrix = new Matrix<double>(2, 2);
            matrix[0, 0] = 0.5; matrix[0, 1] = 0.5;
            matrix[1, 0] = 0.5; matrix[1, 1] = 0.5;

            // Act
            var isDoublyStochastic = matrix.IsDoublyStochasticMatrix();

            // Assert
            Assert.True(isDoublyStochastic);
        }

        [Fact]
        public void IsAdjacencyMatrix_WithAdjacencyMatrix_ReturnsTrue()
        {
            // Arrange - Adjacency matrix: binary, symmetric, zero diagonal
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 1] = 1.0; matrix[0, 2] = 1.0;
            matrix[1, 0] = 1.0; matrix[1, 2] = 1.0;
            matrix[2, 0] = 1.0; matrix[2, 1] = 1.0;

            // Act
            var isAdjacency = matrix.IsAdjacencyMatrix();

            // Assert
            Assert.True(isAdjacency);
        }

        [Fact]
        public void IsCirculantMatrix_WithCirculantMatrix_ReturnsTrue()
        {
            // Arrange - Circulant matrix: each row is rotated version of previous
            var matrix = new Matrix<double>(3, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = 3.0; matrix[1, 1] = 1.0; matrix[1, 2] = 2.0;
            matrix[2, 0] = 2.0; matrix[2, 1] = 3.0; matrix[2, 2] = 1.0;

            // Act
            var isCirculant = matrix.IsCirculantMatrix();

            // Assert
            Assert.True(isCirculant);
        }

        [Fact]
        public void IsSparseMatrix_WithSparseMatrix_ReturnsTrue()
        {
            // Arrange - Sparse matrix: mostly zeros
            var matrix = Matrix<double>.CreateZeros(10, 10);
            matrix[0, 0] = 1.0;
            matrix[5, 5] = 2.0;

            // Act
            var isSparse = matrix.IsSparseMatrix();

            // Assert
            Assert.True(isSparse);
        }

        [Fact]
        public void IsDenseMatrix_WithDenseMatrix_ReturnsTrue()
        {
            // Arrange - Dense matrix: mostly non-zeros
            var matrix = Matrix<double>.CreateOnes(10, 10);

            // Act
            var isDense = matrix.IsDenseMatrix();

            // Assert
            Assert.True(isDense);
        }

        [Fact]
        public void IsBlockMatrix_WithBlockStructure_ReturnsTrue()
        {
            // Arrange
            var matrix = new Matrix<double>(4, 4);
            // Create a block structure
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0;
            matrix[1, 0] = 3.0; matrix[1, 1] = 4.0;
            matrix[2, 2] = 5.0; matrix[2, 3] = 6.0;
            matrix[3, 2] = 7.0; matrix[3, 3] = 8.0;

            // Act
            var isBlock = matrix.IsBlockMatrix(2, 2);

            // Assert
            Assert.True(isBlock);
        }

        [Fact]
        public void Serialize_Deserialize_RoundTrip_PreservesMatrix()
        {
            // Arrange
            var original = new Matrix<double>(2, 3);
            original[0, 0] = 1.0; original[0, 1] = 2.0; original[0, 2] = 3.0;
            original[1, 0] = 4.0; original[1, 1] = 5.0; original[1, 2] = 6.0;

            // Act
            var serialized = original.Serialize();
            var deserialized = Matrix<double>.Deserialize(serialized);

            // Assert
            Assert.Equal(original.Rows, deserialized.Rows);
            Assert.Equal(original.Columns, deserialized.Columns);
            for (int i = 0; i < original.Rows; i++)
                for (int j = 0; j < original.Columns; j++)
                    Assert.Equal(original[i, j], deserialized[i, j], precision: 10);
        }

        [Fact]
        public void Matrix_LargeMatrix_100x100_PerformanceTest()
        {
            // Arrange
            var matrixA = Matrix<double>.CreateRandom(100, 100);
            var matrixB = Matrix<double>.CreateRandom(100, 100);

            // Act
            var result = matrixA + matrixB;

            // Assert
            Assert.Equal(100, result.Rows);
            Assert.Equal(100, result.Columns);
        }

        [Fact]
        public void Matrix_DoubleTranspose_ReturnsOriginal()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 4);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                    matrix[i, j] = i * 4 + j + 1;

            // Act
            var doubleTranspose = matrix.Transpose().Transpose();

            // Assert
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 4; j++)
                    Assert.Equal(matrix[i, j], doubleTranspose[i, j], precision: 10);
        }

        [Fact]
        public void Matrix_AdditionCommutative_AEqualsB()
        {
            // Arrange
            var A = Matrix<double>.CreateRandom(3, 3);
            var B = Matrix<double>.CreateRandom(3, 3);

            // Act
            var result1 = A + B;
            var result2 = B + A;

            // Assert - A + B = B + A
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    Assert.Equal(result1[i, j], result2[i, j], precision: 10);
        }

        [Fact]
        public void Matrix_MultiplicationAssociative_ABC()
        {
            // Arrange
            var A = new Matrix<double>(2, 3);
            var B = new Matrix<double>(3, 2);
            var C = new Matrix<double>(2, 2);

            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++)
                    A[i, j] = i + j + 1;

            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 2; j++)
                    B[i, j] = i - j + 1;

            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    C[i, j] = i * j + 1;

            // Act
            var result1 = (A * B) * C;
            var result2 = A * (B * C);

            // Assert - (AB)C = A(BC)
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                    Assert.Equal(result1[i, j], result2[i, j], precision: 8);
        }

        [Fact]
        public void Matrix_DistributiveProperty_ABC()
        {
            // Arrange
            var A = Matrix<double>.CreateRandom(3, 3);
            var B = Matrix<double>.CreateRandom(3, 3);
            var C = Matrix<double>.CreateRandom(3, 3);

            // Act
            var left = A * (B + C);
            var right = (A * B) + (A * C);

            // Assert - A(B + C) = AB + AC
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    Assert.Equal(left[i, j], right[i, j], precision: 8);
        }

        [Fact]
        public void GetColumns_ReturnsAllColumns()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 3);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0; matrix[0, 2] = 3.0;
            matrix[1, 0] = 4.0; matrix[1, 1] = 5.0; matrix[1, 2] = 6.0;

            // Act
            var columns = matrix.GetColumns().ToList();

            // Assert
            Assert.Equal(3, columns.Count);
            Assert.Equal(1.0, columns[0][0], precision: 10);
            Assert.Equal(4.0, columns[0][1], precision: 10);
            Assert.Equal(6.0, columns[2][1], precision: 10);
        }

        [Fact]
        public void MatrixEnumerator_IteratesAllElements()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 2);
            matrix[0, 0] = 1.0; matrix[0, 1] = 2.0;
            matrix[1, 0] = 3.0; matrix[1, 1] = 4.0;

            // Act
            var elements = new List<double>();
            foreach (var element in matrix)
            {
                elements.Add(element);
            }

            // Assert
            Assert.Equal(4, elements.Count);
            Assert.Equal(1.0, elements[0], precision: 10);
            Assert.Equal(2.0, elements[1], precision: 10);
            Assert.Equal(3.0, elements[2], precision: 10);
            Assert.Equal(4.0, elements[3], precision: 10);
        }
    }
}
