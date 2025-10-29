using System;
using System.Linq;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.LinearAlgebra
{
    public class MatrixTests
    {
        [Fact]
        public void Constructor_WithDimensions_InitializesCorrectly()
        {
            // Arrange & Act
            var matrix = new Matrix<double>(3, 4);

            // Assert
            Assert.Equal(3, matrix.Rows);
            Assert.Equal(4, matrix.Columns);
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    Assert.Equal(0.0, matrix[i, j]);
                }
            }
        }

        [Fact]
        public void Constructor_WithZeroRows_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new Matrix<double>(0, 3));
        }

        [Fact]
        public void Constructor_WithZeroColumns_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new Matrix<double>(3, 0));
        }

        [Fact]
        public void Constructor_WithNegativeDimensions_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new Matrix<double>(-1, 3));
            Assert.Throws<ArgumentException>(() => new Matrix<double>(3, -1));
        }

        [Fact]
        public void Constructor_With2DArray_InitializesCorrectly()
        {
            // Arrange
            var data = new double[,]
            {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            };

            // Act
            var matrix = new Matrix<double>(data);

            // Assert
            Assert.Equal(2, matrix.Rows);
            Assert.Equal(3, matrix.Columns);
            Assert.Equal(1.0, matrix[0, 0]);
            Assert.Equal(2.0, matrix[0, 1]);
            Assert.Equal(3.0, matrix[0, 2]);
            Assert.Equal(4.0, matrix[1, 0]);
            Assert.Equal(5.0, matrix[1, 1]);
            Assert.Equal(6.0, matrix[1, 2]);
        }

        [Fact]
        public void Constructor_WithJaggedArray_InitializesCorrectly()
        {
            // Arrange
            var data = new double[][]
            {
                new double[] { 1.0, 2.0, 3.0 },
                new double[] { 4.0, 5.0, 6.0 }
            };

            // Act
            var matrix = new Matrix<double>(data);

            // Assert
            Assert.Equal(2, matrix.Rows);
            Assert.Equal(3, matrix.Columns);
            Assert.Equal(1.0, matrix[0, 0]);
            Assert.Equal(6.0, matrix[1, 2]);
        }

        [Fact]
        public void Indexer_GetAndSet_WorksCorrectly()
        {
            // Arrange
            var matrix = new Matrix<double>(2, 2);

            // Act
            matrix[0, 0] = 10.0;
            matrix[0, 1] = 20.0;
            matrix[1, 0] = 30.0;
            matrix[1, 1] = 40.0;

            // Assert
            Assert.Equal(10.0, matrix[0, 0]);
            Assert.Equal(20.0, matrix[0, 1]);
            Assert.Equal(30.0, matrix[1, 0]);
            Assert.Equal(40.0, matrix[1, 1]);
        }

        [Fact]
        public void Add_TwoMatrices_ReturnsCorrectSum()
        {
            // Arrange
            var m1 = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0 },
                { 3.0, 4.0 }
            });
            var m2 = new Matrix<double>(new double[,]
            {
                { 5.0, 6.0 },
                { 7.0, 8.0 }
            });

            // Act
            var result = m1.Add(m2);

            // Assert
            Assert.Equal(6.0, result[0, 0]);
            Assert.Equal(8.0, result[0, 1]);
            Assert.Equal(10.0, result[1, 0]);
            Assert.Equal(12.0, result[1, 1]);
        }

        [Fact]
        public void Add_DifferentDimensions_ThrowsArgumentException()
        {
            // Arrange
            var m1 = new Matrix<double>(2, 2);
            var m2 = new Matrix<double>(3, 3);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => m1.Add(m2));
        }

        [Fact]
        public void Subtract_TwoMatrices_ReturnsCorrectDifference()
        {
            // Arrange
            var m1 = new Matrix<double>(new double[,]
            {
                { 10.0, 20.0 },
                { 30.0, 40.0 }
            });
            var m2 = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0 },
                { 3.0, 4.0 }
            });

            // Act
            var result = m1.Subtract(m2);

            // Assert
            Assert.Equal(9.0, result[0, 0]);
            Assert.Equal(18.0, result[0, 1]);
            Assert.Equal(27.0, result[1, 0]);
            Assert.Equal(36.0, result[1, 1]);
        }

        [Fact]
        public void Multiply_ByScalar_ReturnsCorrectResult()
        {
            // Arrange
            var matrix = new Matrix<double>(new double[,]
            {
                { 2.0, 4.0 },
                { 6.0, 8.0 }
            });

            // Act
            var result = matrix.Multiply(3.0);

            // Assert
            Assert.Equal(6.0, result[0, 0]);
            Assert.Equal(12.0, result[0, 1]);
            Assert.Equal(18.0, result[1, 0]);
            Assert.Equal(24.0, result[1, 1]);
        }

        [Fact]
        public void Multiply_TwoMatrices_ReturnsCorrectProduct()
        {
            // Arrange
            var m1 = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0 },
                { 3.0, 4.0 }
            });
            var m2 = new Matrix<double>(new double[,]
            {
                { 5.0, 6.0 },
                { 7.0, 8.0 }
            });

            // Act
            var result = m1.Multiply(m2);

            // Assert
            // [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
            // [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
            Assert.Equal(19.0, result[0, 0]);
            Assert.Equal(22.0, result[0, 1]);
            Assert.Equal(43.0, result[1, 0]);
            Assert.Equal(50.0, result[1, 1]);
        }

        [Fact]
        public void Multiply_IncompatibleDimensions_ThrowsArgumentException()
        {
            // Arrange
            var m1 = new Matrix<double>(2, 3);
            var m2 = new Matrix<double>(2, 2);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => m1.Multiply(m2));
        }

        [Fact]
        public void Transpose_ReturnsCorrectResult()
        {
            // Arrange
            var matrix = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });

            // Act
            var result = matrix.Transpose();

            // Assert
            Assert.Equal(3, result.Rows);
            Assert.Equal(2, result.Columns);
            Assert.Equal(1.0, result[0, 0]);
            Assert.Equal(4.0, result[0, 1]);
            Assert.Equal(2.0, result[1, 0]);
            Assert.Equal(5.0, result[1, 1]);
            Assert.Equal(3.0, result[2, 0]);
            Assert.Equal(6.0, result[2, 1]);
        }

        [Fact]
        public void CreateIdentityMatrix_ReturnsCorrectIdentity()
        {
            // Act
            var identity = Matrix<double>.CreateIdentityMatrix<double>(3);

            // Assert
            Assert.Equal(3, identity.Rows);
            Assert.Equal(3, identity.Columns);
            Assert.Equal(1.0, identity[0, 0]);
            Assert.Equal(0.0, identity[0, 1]);
            Assert.Equal(0.0, identity[0, 2]);
            Assert.Equal(0.0, identity[1, 0]);
            Assert.Equal(1.0, identity[1, 1]);
            Assert.Equal(0.0, identity[1, 2]);
            Assert.Equal(0.0, identity[2, 0]);
            Assert.Equal(0.0, identity[2, 1]);
            Assert.Equal(1.0, identity[2, 2]);
        }

        [Fact]
        public void CreateIdentityMatrix_SizeOne_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => Matrix<double>.CreateIdentityMatrix<double>(1));
        }

        [Fact]
        public void CreateIdentityMatrix_SizeZero_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => Matrix<double>.CreateIdentityMatrix<double>(0));
        }

        [Fact]
        public void GetRow_ReturnsCorrectVector()
        {
            // Arrange
            var matrix = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 },
                { 7.0, 8.0, 9.0 }
            });

            // Act
            var row = matrix.GetRow(1);

            // Assert
            Assert.Equal(3, row.Length);
            Assert.Equal(4.0, row[0]);
            Assert.Equal(5.0, row[1]);
            Assert.Equal(6.0, row[2]);
        }

        [Fact]
        public void GetColumn_ReturnsCorrectVector()
        {
            // Arrange
            var matrix = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 },
                { 7.0, 8.0, 9.0 }
            });

            // Act
            var column = matrix.GetColumn(1);

            // Assert
            Assert.Equal(3, column.Length);
            Assert.Equal(2.0, column[0]);
            Assert.Equal(5.0, column[1]);
            Assert.Equal(8.0, column[2]);
        }

        [Fact]
        public void SetRow_UpdatesCorrectly()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 3);
            var newRow = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            matrix.SetRow(1, newRow);

            // Assert
            Assert.Equal(1.0, matrix[1, 0]);
            Assert.Equal(2.0, matrix[1, 1]);
            Assert.Equal(3.0, matrix[1, 2]);
        }

        [Fact]
        public void SetColumn_UpdatesCorrectly()
        {
            // Arrange
            var matrix = new Matrix<double>(3, 3);
            var newColumn = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

            // Act
            matrix.SetColumn(1, newColumn);

            // Assert
            Assert.Equal(1.0, matrix[0, 1]);
            Assert.Equal(2.0, matrix[1, 1]);
            Assert.Equal(3.0, matrix[2, 1]);
        }

        [Fact]
        public void Clone_CreatesDeepCopy()
        {
            // Arrange
            var original = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0 },
                { 3.0, 4.0 }
            });

            // Act
            var clone = original.Clone();
            clone[0, 0] = 999.0;

            // Assert
            Assert.Equal(1.0, original[0, 0]);
            Assert.Equal(999.0, clone[0, 0]);
            Assert.Equal(original.Rows, clone.Rows);
            Assert.Equal(original.Columns, clone.Columns);
        }

        [Fact]
        public void Determinant_2x2Matrix_ReturnsCorrectValue()
        {
            // Arrange
            var matrix = new Matrix<double>(new double[,]
            {
                { 3.0, 8.0 },
                { 4.0, 6.0 }
            });

            // Act
            var det = matrix.Determinant();

            // Assert
            // det = 3*6 - 8*4 = 18 - 32 = -14
            Assert.Equal(-14.0, det, 5);
        }

        [Fact]
        public void Determinant_3x3Matrix_ReturnsCorrectValue()
        {
            // Arrange
            var matrix = new Matrix<double>(new double[,]
            {
                { 6.0, 1.0, 1.0 },
                { 4.0, -2.0, 5.0 },
                { 2.0, 8.0, 7.0 }
            });

            // Act
            var det = matrix.Determinant();

            // Assert
            // det = 6*(-2*7 - 5*8) - 1*(4*7 - 5*2) + 1*(4*8 - (-2)*2)
            //     = 6*(-54) - 1*(18) + 1*(36)
            //     = -324 - 18 + 36 = -306
            Assert.Equal(-306.0, det, 5);
        }

        [Fact]
        public void Inverse_2x2Matrix_ReturnsCorrectInverse()
        {
            // Arrange
            var matrix = new Matrix<double>(new double[,]
            {
                { 4.0, 7.0 },
                { 2.0, 6.0 }
            });

            // Act
            var inverse = matrix.Inverse();

            // Assert
            var identity = matrix.Multiply(inverse);
            Assert.Equal(1.0, identity[0, 0], 5);
            Assert.Equal(0.0, identity[0, 1], 5);
            Assert.Equal(0.0, identity[1, 0], 5);
            Assert.Equal(1.0, identity[1, 1], 5);
        }

        [Fact]
        public void ElementwiseMultiply_TwoMatrices_ReturnsCorrectResult()
        {
            // Arrange
            var m1 = new Matrix<double>(new double[,]
            {
                { 2.0, 3.0 },
                { 4.0, 5.0 }
            });
            var m2 = new Matrix<double>(new double[,]
            {
                { 6.0, 7.0 },
                { 8.0, 9.0 }
            });

            // Act
            var result = m1.ElementwiseMultiply(m2);

            // Assert
            Assert.Equal(12.0, result[0, 0]);
            Assert.Equal(21.0, result[0, 1]);
            Assert.Equal(32.0, result[1, 0]);
            Assert.Equal(45.0, result[1, 1]);
        }

        [Fact]
        public void Sum_ReturnsCorrectTotal()
        {
            // Arrange
            var matrix = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });

            // Act
            var result = matrix.Sum();

            // Assert
            // 1 + 2 + 3 + 4 + 5 + 6 = 21
            Assert.Equal(21.0, result);
        }

        [Fact]
        public void Mean_ReturnsCorrectAverage()
        {
            // Arrange
            var matrix = new Matrix<double>(new double[,]
            {
                { 2.0, 4.0 },
                { 6.0, 8.0 }
            });

            // Act
            var result = matrix.Mean();

            // Assert
            // (2 + 4 + 6 + 8) / 4 = 20 / 4 = 5.0
            Assert.Equal(5.0, result);
        }

        [Fact]
        public void Apply_AppliesFunctionToEachElement()
        {
            // Arrange
            var matrix = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0 },
                { 3.0, 4.0 }
            });

            // Act
            var result = matrix.Apply(x => x * 2.0);

            // Assert
            Assert.Equal(2.0, result[0, 0]);
            Assert.Equal(4.0, result[0, 1]);
            Assert.Equal(6.0, result[1, 0]);
            Assert.Equal(8.0, result[1, 1]);
        }

        [Fact]
        public void IntMatrix_Constructor_WorksCorrectly()
        {
            // Arrange & Act
            var matrix = new Matrix<int>(new int[,]
            {
                { 1, 2, 3 },
                { 4, 5, 6 }
            });

            // Assert
            Assert.Equal(2, matrix.Rows);
            Assert.Equal(3, matrix.Columns);
            Assert.Equal(1, matrix[0, 0]);
            Assert.Equal(6, matrix[1, 2]);
        }

        [Fact]
        public void IntMatrix_Add_WorksCorrectly()
        {
            // Arrange
            var m1 = new Matrix<int>(new int[,] { { 1, 2 }, { 3, 4 } });
            var m2 = new Matrix<int>(new int[,] { { 5, 6 }, { 7, 8 } });

            // Act
            var result = m1.Add(m2);

            // Assert
            Assert.Equal(6, result[0, 0]);
            Assert.Equal(8, result[0, 1]);
            Assert.Equal(10, result[1, 0]);
            Assert.Equal(12, result[1, 1]);
        }

        [Fact]
        public void FloatMatrix_Constructor_WorksCorrectly()
        {
            // Arrange & Act
            var matrix = new Matrix<float>(new float[,]
            {
                { 1.0f, 2.0f },
                { 3.0f, 4.0f }
            });

            // Assert
            Assert.Equal(2, matrix.Rows);
            Assert.Equal(2, matrix.Columns);
            Assert.Equal(1.0f, matrix[0, 0]);
            Assert.Equal(4.0f, matrix[1, 1]);
        }

        [Fact]
        public void CreateMatrix_StaticMethod_WorksCorrectly()
        {
            // Act
            var matrix = Matrix<double>.CreateMatrix<double>(3, 4);

            // Assert
            Assert.Equal(3, matrix.Rows);
            Assert.Equal(4, matrix.Columns);
        }

        [Fact]
        public void MultiplyVector_ReturnsCorrectResult()
        {
            // Arrange
            var matrix = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0, 3.0 },
                { 4.0, 5.0, 6.0 }
            });
            var vector = new Vector<double>(new double[] { 2.0, 3.0, 4.0 });

            // Act
            var result = matrix.MultiplyVector(vector);

            // Assert
            // [1*2 + 2*3 + 3*4] = [2 + 6 + 12] = [20]
            // [4*2 + 5*3 + 6*4] = [8 + 15 + 24] = [47]
            Assert.Equal(2, result.Length);
            Assert.Equal(20.0, result[0]);
            Assert.Equal(47.0, result[1]);
        }

        [Fact]
        public void ToArray_Returns2DArray()
        {
            // Arrange
            var matrix = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0 },
                { 3.0, 4.0 }
            });

            // Act
            var array = matrix.ToArray();

            // Assert
            Assert.Equal(1.0, array[0, 0]);
            Assert.Equal(2.0, array[0, 1]);
            Assert.Equal(3.0, array[1, 0]);
            Assert.Equal(4.0, array[1, 1]);
        }
    }
}
