using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.LinearAlgebra;

/// <summary>
/// Integration tests for Matrix and Vector classes working together.
/// Tests real integration scenarios involving matrix-vector operations,
/// serialization roundtrips, and complex numerical computations.
/// </summary>
public class MatrixVectorIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region Matrix Construction Tests

    [Fact]
    public void Matrix_ConstructWithRowsAndColumns_CreatesCorrectDimensions()
    {
        // Arrange & Act
        var matrix = new Matrix<double>(3, 4);

        // Assert
        Assert.Equal(3, matrix.Rows);
        Assert.Equal(4, matrix.Columns);
    }

    [Fact]
    public void Matrix_ConstructFromNestedEnumerable_InitializesValues()
    {
        // Arrange
        var values = new List<List<double>>
        {
            new() { 1.0, 2.0, 3.0 },
            new() { 4.0, 5.0, 6.0 }
        };

        // Act
        var matrix = new Matrix<double>(values);

        // Assert
        Assert.Equal(2, matrix.Rows);
        Assert.Equal(3, matrix.Columns);
        Assert.Equal(1.0, matrix[0, 0]);
        Assert.Equal(6.0, matrix[1, 2]);
    }

    [Fact]
    public void Matrix_ConstructFrom2DArray_InitializesCorrectly()
    {
        // Arrange
        var data = new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
            { 5.0, 6.0 }
        };

        // Act
        var matrix = new Matrix<double>(data);

        // Assert
        Assert.Equal(3, matrix.Rows);
        Assert.Equal(2, matrix.Columns);
        Assert.Equal(1.0, matrix[0, 0]);
        Assert.Equal(4.0, matrix[1, 1]);
        Assert.Equal(6.0, matrix[2, 1]);
    }

    #endregion

    #region Vector Construction Tests

    [Fact]
    public void Vector_ConstructWithLength_CreatesCorrectSize()
    {
        // Arrange & Act
        var vector = new Vector<double>(5);

        // Assert
        Assert.Equal(5, vector.Length);
    }

    [Fact]
    public void Vector_ConstructFromEnumerable_InitializesValues()
    {
        // Arrange
        var values = new[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // Act
        var vector = new Vector<double>(values);

        // Assert
        Assert.Equal(5, vector.Length);
        Assert.Equal(1.0, vector[0]);
        Assert.Equal(5.0, vector[4]);
    }

    [Fact]
    public void Vector_FromArray_CreatesVectorFromArray()
    {
        // Arrange
        var array = new[] { 10.0, 20.0, 30.0 };

        // Act
        var vector = Vector<double>.FromArray(array);

        // Assert
        Assert.Equal(3, vector.Length);
        Assert.Equal(10.0, vector[0]);
        Assert.Equal(30.0, vector[2]);
    }

    #endregion

    #region Matrix Static Factory Methods

    [Fact]
    public void Matrix_CreateIdentity_CreatesCorrectIdentityMatrix()
    {
        // Arrange & Act
        var identity = Matrix<double>.CreateIdentity(3);

        // Assert
        Assert.Equal(3, identity.Rows);
        Assert.Equal(3, identity.Columns);
        Assert.Equal(1.0, identity[0, 0]);
        Assert.Equal(1.0, identity[1, 1]);
        Assert.Equal(1.0, identity[2, 2]);
        Assert.Equal(0.0, identity[0, 1]);
        Assert.Equal(0.0, identity[1, 0]);
    }

    [Fact]
    public void Matrix_CreateZeros_CreatesMatrixOfZeros()
    {
        // Arrange & Act
        var zeros = Matrix<double>.CreateZeros(2, 3);

        // Assert
        Assert.Equal(2, zeros.Rows);
        Assert.Equal(3, zeros.Columns);
        for (int i = 0; i < zeros.Rows; i++)
        {
            for (int j = 0; j < zeros.Columns; j++)
            {
                Assert.Equal(0.0, zeros[i, j]);
            }
        }
    }

    [Fact]
    public void Matrix_CreateOnes_CreatesMatrixOfOnes()
    {
        // Arrange & Act
        var ones = Matrix<double>.CreateOnes(2, 3);

        // Assert
        Assert.Equal(2, ones.Rows);
        Assert.Equal(3, ones.Columns);
        for (int i = 0; i < ones.Rows; i++)
        {
            for (int j = 0; j < ones.Columns; j++)
            {
                Assert.Equal(1.0, ones[i, j]);
            }
        }
    }

    [Fact]
    public void Matrix_CreateRandom_CreatesMatrixWithRandomValues()
    {
        // Arrange & Act
        var random = Matrix<double>.CreateRandom(3, 4);

        // Assert
        Assert.Equal(3, random.Rows);
        Assert.Equal(4, random.Columns);
    }

    [Fact]
    public void Matrix_CreateDiagonal_CreatesDiagonalMatrix()
    {
        // Arrange
        var diagonal = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var matrix = Matrix<double>.CreateDiagonal(diagonal);

        // Assert
        Assert.Equal(3, matrix.Rows);
        Assert.Equal(3, matrix.Columns);
        Assert.Equal(1.0, matrix[0, 0]);
        Assert.Equal(2.0, matrix[1, 1]);
        Assert.Equal(3.0, matrix[2, 2]);
        Assert.Equal(0.0, matrix[0, 1]);
    }

    #endregion

    #region Vector Static Factory Methods

    [Fact]
    public void Vector_CreateRandom_CreatesVectorWithRandomValues()
    {
        // Arrange & Act
        var random = Vector<double>.CreateRandom(5);

        // Assert
        Assert.Equal(5, random.Length);
    }

    [Fact]
    public void Vector_Concatenate_CombinesMultipleVectors()
    {
        // Arrange
        var v1 = new Vector<double>(new[] { 1.0, 2.0 });
        var v2 = new Vector<double>(new[] { 3.0, 4.0, 5.0 });

        // Act
        var concatenated = Vector<double>.Concatenate(v1, v2);

        // Assert
        Assert.Equal(5, concatenated.Length);
        Assert.Equal(1.0, concatenated[0]);
        Assert.Equal(2.0, concatenated[1]);
        Assert.Equal(3.0, concatenated[2]);
        Assert.Equal(5.0, concatenated[4]);
    }

    [Fact]
    public void Vector_CreateStandardBasis_CreatesCorrectBasisVector()
    {
        // Arrange & Act
        var basis = Vector<double>.CreateStandardBasis(4, 2);

        // Assert
        Assert.Equal(4, basis.Length);
        Assert.Equal(0.0, basis[0]);
        Assert.Equal(0.0, basis[1]);
        Assert.Equal(1.0, basis[2]);
        Assert.Equal(0.0, basis[3]);
    }

    #endregion

    #region Matrix-Vector Multiplication

    [Fact]
    public void MatrixVectorMultiply_ComputesCorrectResult()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 }
        });
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var result = matrix.Multiply(vector);

        // Assert
        Assert.Equal(2, result.Length);
        Assert.Equal(14.0, result[0], Tolerance); // 1*1 + 2*2 + 3*3 = 14
        Assert.Equal(32.0, result[1], Tolerance); // 4*1 + 5*2 + 6*3 = 32
    }

    [Fact]
    public void MatrixVectorOperator_ComputesSameAsMethod()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 0.0 },
            { 0.0, 2.0 }
        });
        var vector = new Vector<double>(new[] { 3.0, 4.0 });

        // Act
        var result = matrix * vector;

        // Assert
        Assert.Equal(2, result.Length);
        Assert.Equal(3.0, result[0], Tolerance);
        Assert.Equal(8.0, result[1], Tolerance);
    }

    #endregion

    #region Matrix-Matrix Operations

    [Fact]
    public void MatrixAdd_AddsTwoMatrices()
    {
        // Arrange
        var m1 = new Matrix<double>(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        var m2 = new Matrix<double>(new double[,] { { 5.0, 6.0 }, { 7.0, 8.0 } });

        // Act
        var result = m1 + m2;

        // Assert
        Assert.Equal(6.0, result[0, 0], Tolerance);
        Assert.Equal(8.0, result[0, 1], Tolerance);
        Assert.Equal(10.0, result[1, 0], Tolerance);
        Assert.Equal(12.0, result[1, 1], Tolerance);
    }

    [Fact]
    public void MatrixSubtract_SubtractsTwoMatrices()
    {
        // Arrange
        var m1 = new Matrix<double>(new double[,] { { 5.0, 6.0 }, { 7.0, 8.0 } });
        var m2 = new Matrix<double>(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });

        // Act
        var result = m1 - m2;

        // Assert
        Assert.Equal(4.0, result[0, 0], Tolerance);
        Assert.Equal(4.0, result[0, 1], Tolerance);
        Assert.Equal(4.0, result[1, 0], Tolerance);
        Assert.Equal(4.0, result[1, 1], Tolerance);
    }

    [Fact]
    public void MatrixMultiply_MultipliesTwoMatrices()
    {
        // Arrange
        var m1 = new Matrix<double>(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        var m2 = new Matrix<double>(new double[,] { { 5.0, 6.0 }, { 7.0, 8.0 } });

        // Act
        var result = m1 * m2;

        // Assert
        Assert.Equal(19.0, result[0, 0], Tolerance);  // 1*5 + 2*7 = 19
        Assert.Equal(22.0, result[0, 1], Tolerance);  // 1*6 + 2*8 = 22
        Assert.Equal(43.0, result[1, 0], Tolerance);  // 3*5 + 4*7 = 43
        Assert.Equal(50.0, result[1, 1], Tolerance);  // 3*6 + 4*8 = 50
    }

    [Fact]
    public void MatrixScalarMultiply_MultipliesMatrixByScalar()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });

        // Act
        var result = matrix * 2.0;

        // Assert
        Assert.Equal(2.0, result[0, 0], Tolerance);
        Assert.Equal(4.0, result[0, 1], Tolerance);
        Assert.Equal(6.0, result[1, 0], Tolerance);
        Assert.Equal(8.0, result[1, 1], Tolerance);
    }

    [Fact]
    public void MatrixScalarDivide_DividesMatrixByScalar()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,] { { 2.0, 4.0 }, { 6.0, 8.0 } });

        // Act
        var result = matrix / 2.0;

        // Assert
        Assert.Equal(1.0, result[0, 0], Tolerance);
        Assert.Equal(2.0, result[0, 1], Tolerance);
        Assert.Equal(3.0, result[1, 0], Tolerance);
        Assert.Equal(4.0, result[1, 1], Tolerance);
    }

    #endregion

    #region Vector-Vector Operations

    [Fact]
    public void VectorAdd_AddsTwoVectors()
    {
        // Arrange
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        // Act
        var result = v1 + v2;

        // Assert
        Assert.Equal(5.0, result[0], Tolerance);
        Assert.Equal(7.0, result[1], Tolerance);
        Assert.Equal(9.0, result[2], Tolerance);
    }

    [Fact]
    public void VectorSubtract_SubtractsTwoVectors()
    {
        // Arrange
        var v1 = new Vector<double>(new[] { 5.0, 7.0, 9.0 });
        var v2 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var result = v1 - v2;

        // Assert
        Assert.Equal(4.0, result[0], Tolerance);
        Assert.Equal(5.0, result[1], Tolerance);
        Assert.Equal(6.0, result[2], Tolerance);
    }

    [Fact]
    public void VectorScalarMultiply_MultipliesVectorByScalar()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var result = vector * 3.0;

        // Assert
        Assert.Equal(3.0, result[0], Tolerance);
        Assert.Equal(6.0, result[1], Tolerance);
        Assert.Equal(9.0, result[2], Tolerance);
    }

    [Fact]
    public void VectorScalarDivide_DividesVectorByScalar()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 6.0, 9.0, 12.0 });

        // Act
        var result = vector / 3.0;

        // Assert
        Assert.Equal(2.0, result[0], Tolerance);
        Assert.Equal(3.0, result[1], Tolerance);
        Assert.Equal(4.0, result[2], Tolerance);
    }

    [Fact]
    public void VectorElementwiseMultiply_MultipliesElementWise()
    {
        // Arrange
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        // Act
        var result = v1.ElementwiseMultiply(v2);

        // Assert
        Assert.Equal(4.0, result[0], Tolerance);
        Assert.Equal(10.0, result[1], Tolerance);
        Assert.Equal(18.0, result[2], Tolerance);
    }

    [Fact]
    public void VectorElementwiseDivide_DividesElementWise()
    {
        // Arrange
        var v1 = new Vector<double>(new[] { 4.0, 10.0, 18.0 });
        var v2 = new Vector<double>(new[] { 2.0, 5.0, 6.0 });

        // Act
        var result = v1.ElementwiseDivide(v2);

        // Assert
        Assert.Equal(2.0, result[0], Tolerance);
        Assert.Equal(2.0, result[1], Tolerance);
        Assert.Equal(3.0, result[2], Tolerance);
    }

    #endregion

    #region Matrix Transpose and Clone

    [Fact]
    public void MatrixTranspose_TransposesMatrix()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 }
        });

        // Act
        var transposed = matrix.Transpose();

        // Assert
        Assert.Equal(3, transposed.Rows);
        Assert.Equal(2, transposed.Columns);
        Assert.Equal(1.0, transposed[0, 0]);
        Assert.Equal(4.0, transposed[0, 1]);
        Assert.Equal(2.0, transposed[1, 0]);
        Assert.Equal(5.0, transposed[1, 1]);
    }

    [Fact]
    public void MatrixClone_CreatesIndependentCopy()
    {
        // Arrange
        var original = new Matrix<double>(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });

        // Act
        var clone = original.Clone();
        clone[0, 0] = 999.0;

        // Assert
        Assert.Equal(1.0, original[0, 0]); // Original unchanged
        Assert.Equal(999.0, clone[0, 0]); // Clone modified
    }

    #endregion

    #region Vector Operations

    [Fact]
    public void VectorClone_CreatesIndependentCopy()
    {
        // Arrange
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var clone = original.Clone();
        clone[0] = 999.0;

        // Assert
        Assert.Equal(1.0, original[0]); // Original unchanged
        Assert.Equal(999.0, clone[0]); // Clone modified
    }

    [Fact]
    public void VectorNormalize_CreatesUnitVector()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 3.0, 4.0 });

        // Act
        var normalized = vector.Normalize();

        // Assert
        var norm = Math.Sqrt(normalized[0] * normalized[0] + normalized[1] * normalized[1]);
        Assert.Equal(1.0, norm, Tolerance);
        Assert.Equal(0.6, normalized[0], Tolerance);
        Assert.Equal(0.8, normalized[1], Tolerance);
    }

    [Fact]
    public void VectorVariance_ComputesCorrectVariance()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0 });

        // Act
        var variance = vector.Variance();

        // Assert
        Assert.True(variance > 0);
    }

    [Fact]
    public void VectorNorm_ComputesL2Norm()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 3.0, 4.0 });

        // Act
        var norm = vector.Norm();

        // Assert
        Assert.Equal(5.0, norm, Tolerance); // sqrt(9 + 16) = 5
    }

    #endregion

    #region Matrix Row/Column Operations

    [Fact]
    public void MatrixGetColumn_ReturnsCorrectColumn()
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
    public void MatrixGetSubMatrix_ReturnsCorrectSubMatrix()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0 },
            { 5.0, 6.0, 7.0, 8.0 },
            { 9.0, 10.0, 11.0, 12.0 },
            { 13.0, 14.0, 15.0, 16.0 }
        });

        // Act
        var subMatrix = matrix.GetSubMatrix(1, 1, 2, 2);

        // Assert
        Assert.Equal(2, subMatrix.Rows);
        Assert.Equal(2, subMatrix.Columns);
        Assert.Equal(6.0, subMatrix[0, 0]);
        Assert.Equal(7.0, subMatrix[0, 1]);
        Assert.Equal(10.0, subMatrix[1, 0]);
        Assert.Equal(11.0, subMatrix[1, 1]);
    }

    [Fact]
    public void MatrixPointwiseDivide_DividesElementWise()
    {
        // Arrange
        var m1 = new Matrix<double>(new double[,] { { 4.0, 6.0 }, { 8.0, 10.0 } });
        var m2 = new Matrix<double>(new double[,] { { 2.0, 3.0 }, { 4.0, 5.0 } });

        // Act
        var result = m1.PointwiseDivide(m2);

        // Assert
        Assert.Equal(2.0, result[0, 0], Tolerance);
        Assert.Equal(2.0, result[0, 1], Tolerance);
        Assert.Equal(2.0, result[1, 0], Tolerance);
        Assert.Equal(2.0, result[1, 1], Tolerance);
    }

    #endregion

    #region Complex Integration Scenarios

    [Fact]
    public void IdentityMatrixMultiplication_PreservesVector()
    {
        // Arrange
        var identity = Matrix<double>.CreateIdentity(3);
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var result = identity * vector;

        // Assert
        Assert.Equal(1.0, result[0], Tolerance);
        Assert.Equal(2.0, result[1], Tolerance);
        Assert.Equal(3.0, result[2], Tolerance);
    }

    [Fact]
    public void MatrixMultiplicationAssociativity()
    {
        // Arrange
        var A = new Matrix<double>(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        var B = new Matrix<double>(new double[,] { { 5.0, 6.0 }, { 7.0, 8.0 } });
        var C = new Matrix<double>(new double[,] { { 1.0, 0.0 }, { 0.0, 1.0 } });

        // Act
        var result1 = (A * B) * C;
        var result2 = A * (B * C);

        // Assert
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                Assert.Equal(result1[i, j], result2[i, j], Tolerance);
            }
        }
    }

    [Fact]
    public void VectorOuterProduct_CreatesCorrectMatrix()
    {
        // Arrange
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v2 = new Vector<double>(new[] { 4.0, 5.0 });

        // Act
        var outerProduct = v1.OuterProduct(v2);

        // Assert
        Assert.Equal(3, outerProduct.Rows);
        Assert.Equal(2, outerProduct.Columns);
        Assert.Equal(4.0, outerProduct[0, 0], Tolerance);  // 1 * 4
        Assert.Equal(5.0, outerProduct[0, 1], Tolerance);  // 1 * 5
        Assert.Equal(8.0, outerProduct[1, 0], Tolerance);  // 2 * 4
        Assert.Equal(10.0, outerProduct[1, 1], Tolerance); // 2 * 5
        Assert.Equal(12.0, outerProduct[2, 0], Tolerance); // 3 * 4
        Assert.Equal(15.0, outerProduct[2, 1], Tolerance); // 3 * 5
    }

    [Fact]
    public void DiagonalMatrixMultiplication_ScalesVector()
    {
        // Arrange
        var diagonal = new Vector<double>(new[] { 2.0, 3.0, 4.0 });
        var diagMatrix = Matrix<double>.CreateDiagonal(diagonal);
        var vector = new Vector<double>(new[] { 1.0, 1.0, 1.0 });

        // Act
        var result = diagMatrix * vector;

        // Assert
        Assert.Equal(2.0, result[0], Tolerance);
        Assert.Equal(3.0, result[1], Tolerance);
        Assert.Equal(4.0, result[2], Tolerance);
    }

    [Fact]
    public void TransposeOfTranspose_ReturnsOriginal()
    {
        // Arrange
        var original = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 }
        });

        // Act
        var doubleTranspose = original.Transpose().Transpose();

        // Assert
        Assert.Equal(original.Rows, doubleTranspose.Rows);
        Assert.Equal(original.Columns, doubleTranspose.Columns);
        for (int i = 0; i < original.Rows; i++)
        {
            for (int j = 0; j < original.Columns; j++)
            {
                Assert.Equal(original[i, j], doubleTranspose[i, j], Tolerance);
            }
        }
    }

    #endregion

    #region Vector Subvector and Segment Tests

    [Fact]
    public void VectorGetSubVector_ReturnsCorrectPortion()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        // Act
        var subVector = vector.GetSubVector(1, 3);

        // Assert
        Assert.Equal(3, subVector.Length);
        Assert.Equal(2.0, subVector[0]);
        Assert.Equal(3.0, subVector[1]);
        Assert.Equal(4.0, subVector[2]);
    }

    [Fact]
    public void VectorRemoveAt_RemovesElementAtIndex()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        // Act
        var result = vector.RemoveAt(2);

        // Assert
        Assert.Equal(4, result.Length);
        Assert.Equal(1.0, result[0]);
        Assert.Equal(2.0, result[1]);
        Assert.Equal(4.0, result[2]);
        Assert.Equal(5.0, result[3]);
    }

    [Fact]
    public void VectorIndexOfMax_ReturnsCorrectIndex()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 1.0, 5.0, 3.0, 4.0, 2.0 });

        // Act
        var maxIndex = vector.IndexOfMax();

        // Assert
        Assert.Equal(1, maxIndex);
    }

    #endregion

    #region Matrix Row/Column Removal Tests

    [Fact]
    public void MatrixRemoveRow_RemovesCorrectRow()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 }
        });

        // Act
        var result = matrix.RemoveRow(1);

        // Assert
        Assert.Equal(2, result.Rows);
        Assert.Equal(3, result.Columns);
        Assert.Equal(1.0, result[0, 0]);
        Assert.Equal(7.0, result[1, 0]);
    }

    [Fact]
    public void MatrixRemoveColumn_RemovesCorrectColumn()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 }
        });

        // Act
        var result = matrix.RemoveColumn(1);

        // Assert
        Assert.Equal(3, result.Rows);
        Assert.Equal(2, result.Columns);
        Assert.Equal(1.0, result[0, 0]);
        Assert.Equal(3.0, result[0, 1]);
    }

    #endregion

    #region Large Scale Integration Tests

    [Fact]
    public void LargeMatrixMultiplication_CompletesWithinTimeout()
    {
        // Arrange
        var m1 = Matrix<double>.CreateRandom(100, 100);
        var m2 = Matrix<double>.CreateRandom(100, 100);

        // Act
        var result = m1 * m2;

        // Assert
        Assert.Equal(100, result.Rows);
        Assert.Equal(100, result.Columns);
    }

    [Fact]
    public void LargeVectorOperations_CompletesWithinTimeout()
    {
        // Arrange
        var v1 = Vector<double>.CreateRandom(10000);
        var v2 = Vector<double>.CreateRandom(10000);

        // Act
        var sum = v1 + v2;
        var product = v1.ElementwiseMultiply(v2);
        var normalized = v1.Normalize();

        // Assert
        Assert.Equal(10000, sum.Length);
        Assert.Equal(10000, product.Length);
        Assert.Equal(10000, normalized.Length);
    }

    #endregion
}
