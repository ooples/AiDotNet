using AiDotNet.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

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

    [Fact(Timeout = 120000)]
    public async Task Matrix_ConstructWithRowsAndColumns_CreatesCorrectDimensions()
    {
        // Arrange & Act
        var matrix = new Matrix<double>(3, 4);

        // Assert
        Assert.Equal(3, matrix.Rows);
        Assert.Equal(4, matrix.Columns);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_ConstructFromNestedEnumerable_InitializesValues()
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

    [Fact(Timeout = 120000)]
    public async Task Matrix_ConstructFrom2DArray_InitializesCorrectly()
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

    [Fact(Timeout = 120000)]
    public async Task Vector_ConstructWithLength_CreatesCorrectSize()
    {
        // Arrange & Act
        var vector = new Vector<double>(5);

        // Assert
        Assert.Equal(5, vector.Length);
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_ConstructFromEnumerable_InitializesValues()
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

    [Fact(Timeout = 120000)]
    public async Task Vector_FromArray_CreatesVectorFromArray()
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

    [Fact(Timeout = 120000)]
    public async Task Matrix_CreateIdentity_CreatesCorrectIdentityMatrix()
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

    [Fact(Timeout = 120000)]
    public async Task Matrix_CreateZeros_CreatesMatrixOfZeros()
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

    [Fact(Timeout = 120000)]
    public async Task Matrix_CreateOnes_CreatesMatrixOfOnes()
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

    [Fact(Timeout = 120000)]
    public async Task Matrix_CreateRandom_CreatesMatrixWithRandomValues()
    {
        // Arrange & Act
        var random = Matrix<double>.CreateRandom(3, 4);

        // Assert
        Assert.Equal(3, random.Rows);
        Assert.Equal(4, random.Columns);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_CreateDiagonal_CreatesDiagonalMatrix()
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

    [Fact(Timeout = 120000)]
    public async Task Vector_CreateRandom_CreatesVectorWithRandomValues()
    {
        // Arrange & Act
        var random = Vector<double>.CreateRandom(5);

        // Assert
        Assert.Equal(5, random.Length);
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_Concatenate_CombinesMultipleVectors()
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

    [Fact(Timeout = 120000)]
    public async Task Vector_CreateStandardBasis_CreatesCorrectBasisVector()
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

    [Fact(Timeout = 120000)]
    public async Task MatrixVectorMultiply_ComputesCorrectResult()
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

    [Fact(Timeout = 120000)]
    public async Task MatrixVectorOperator_ComputesSameAsMethod()
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

    [Fact(Timeout = 120000)]
    public async Task MatrixAdd_AddsTwoMatrices()
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

    [Fact(Timeout = 120000)]
    public async Task MatrixSubtract_SubtractsTwoMatrices()
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

    [Fact(Timeout = 120000)]
    public async Task MatrixMultiply_MultipliesTwoMatrices()
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

    [Fact(Timeout = 120000)]
    public async Task MatrixScalarMultiply_MultipliesMatrixByScalar()
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

    [Fact(Timeout = 120000)]
    public async Task MatrixScalarDivide_DividesMatrixByScalar()
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

    [Fact(Timeout = 120000)]
    public async Task VectorAdd_AddsTwoVectors()
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

    [Fact(Timeout = 120000)]
    public async Task VectorSubtract_SubtractsTwoVectors()
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

    [Fact(Timeout = 120000)]
    public async Task VectorScalarMultiply_MultipliesVectorByScalar()
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

    [Fact(Timeout = 120000)]
    public async Task VectorScalarDivide_DividesVectorByScalar()
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

    [Fact(Timeout = 120000)]
    public async Task VectorElementwiseMultiply_MultipliesElementWise()
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

    [Fact(Timeout = 120000)]
    public async Task VectorElementwiseDivide_DividesElementWise()
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

    [Fact(Timeout = 120000)]
    public async Task MatrixTranspose_TransposesMatrix()
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

    [Fact(Timeout = 120000)]
    public async Task MatrixClone_CreatesIndependentCopy()
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

    [Fact(Timeout = 120000)]
    public async Task VectorClone_CreatesIndependentCopy()
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

    [Fact(Timeout = 120000)]
    public async Task VectorNormalize_CreatesUnitVector()
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

    [Fact(Timeout = 120000)]
    public async Task VectorVariance_ComputesCorrectVariance()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0 });

        // Act
        var variance = vector.Variance();

        // Assert
        Assert.True(variance > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task VectorNorm_ComputesL2Norm()
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

    [Fact(Timeout = 120000)]
    public async Task MatrixGetColumn_ReturnsCorrectColumn()
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

    [Fact(Timeout = 120000)]
    public async Task MatrixGetSubMatrix_ReturnsCorrectSubMatrix()
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

    [Fact(Timeout = 120000)]
    public async Task MatrixPointwiseDivide_DividesElementWise()
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

    [Fact(Timeout = 120000)]
    public async Task IdentityMatrixMultiplication_PreservesVector()
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

    [Fact(Timeout = 120000)]
    public async Task MatrixMultiplicationAssociativity()
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

    [Fact(Timeout = 120000)]
    public async Task VectorOuterProduct_CreatesCorrectMatrix()
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

    [Fact(Timeout = 120000)]
    public async Task DiagonalMatrixMultiplication_ScalesVector()
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

    [Fact(Timeout = 120000)]
    public async Task TransposeOfTranspose_ReturnsOriginal()
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

    [Fact(Timeout = 120000)]
    public async Task VectorGetSubVector_ReturnsCorrectPortion()
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

    [Fact(Timeout = 120000)]
    public async Task VectorRemoveAt_RemovesElementAtIndex()
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

    [Fact(Timeout = 120000)]
    public async Task VectorIndexOfMax_ReturnsCorrectIndex()
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

    [Fact(Timeout = 120000)]
    public async Task MatrixRemoveRow_RemovesCorrectRow()
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

    [Fact(Timeout = 120000)]
    public async Task MatrixRemoveColumn_RemovesCorrectColumn()
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

    [Fact(Timeout = 120000)]
    public async Task LargeMatrixMultiplication_CompletesWithinTimeout()
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

    [Fact(Timeout = 120000)]
    public async Task LargeVectorOperations_CompletesWithinTimeout()
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

    #region Vector Additional Factory Methods

    [Fact(Timeout = 120000)]
    public async Task Vector_FromEnumerable_CreatesVectorFromEnumerable()
    {
        // Arrange
        IEnumerable<double> values = new List<double> { 1.0, 2.0, 3.0, 4.0 };

        // Act
        var vector = Vector<double>.FromEnumerable(values);

        // Assert
        Assert.Equal(4, vector.Length);
        Assert.Equal(1.0, vector[0]);
        Assert.Equal(4.0, vector[3]);
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_FromList_CreatesVectorFromList()
    {
        // Arrange
        var list = new List<double> { 5.0, 10.0, 15.0 };

        // Act
        var vector = Vector<double>.FromList(list);

        // Assert
        Assert.Equal(3, vector.Length);
        Assert.Equal(5.0, vector[0]);
        Assert.Equal(15.0, vector[2]);
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_Range_CreatesSequentialValues()
    {
        // Arrange & Act
        var vector = Vector<double>.Range(5, 4);

        // Assert
        Assert.Equal(4, vector.Length);
        Assert.Equal(5.0, vector[0], Tolerance);
        Assert.Equal(6.0, vector[1], Tolerance);
        Assert.Equal(7.0, vector[2], Tolerance);
        Assert.Equal(8.0, vector[3], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_Empty_CreatesEmptyVector()
    {
        // Arrange & Act
        var vector = Vector<double>.Empty();

        // Assert
        Assert.True(vector.IsEmpty || vector.Length == 0);
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_Zeros_CreatesZeroVector()
    {
        // Arrange
        var template = new Vector<double>(1);

        // Act
        var zeros = template.Zeros(5) as Vector<double>;

        // Assert
        Assert.NotNull(zeros);
        Assert.Equal(5, zeros.Length);
        for (int i = 0; i < zeros.Length; i++)
        {
            Assert.Equal(0.0, zeros[i]);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_Ones_CreatesOnesVector()
    {
        // Arrange
        var template = new Vector<double>(1);

        // Act
        var ones = template.Ones(5) as Vector<double>;

        // Assert
        Assert.NotNull(ones);
        Assert.Equal(5, ones.Length);
        for (int i = 0; i < ones.Length; i++)
        {
            Assert.Equal(1.0, ones[i]);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_Default_CreatesVectorWithDefaultValue()
    {
        // Arrange
        var template = new Vector<double>(1);

        // Act
        var defaultVec = template.Default(4, 7.5) as Vector<double>;

        // Assert
        Assert.NotNull(defaultVec);
        Assert.Equal(4, defaultVec.Length);
        for (int i = 0; i < defaultVec.Length; i++)
        {
            Assert.Equal(7.5, defaultVec[i]);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_CreateDefault_CreatesVectorWithDefaultValue()
    {
        // Arrange & Act
        var vector = VectorBase<double>.CreateDefault(5, 3.14);

        // Assert
        Assert.Equal(5, vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            Assert.Equal(3.14, vector[i]);
        }
    }

    #endregion

    #region Vector Functional Methods

    [Fact(Timeout = 120000)]
    public async Task Vector_Where_FiltersElements()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        // Act
        var filtered = vector.Where(x => x > 2.5);

        // Assert
        Assert.Equal(3, filtered.Length);
        Assert.Equal(3.0, filtered[0]);
        Assert.Equal(4.0, filtered[1]);
        Assert.Equal(5.0, filtered[2]);
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_Select_TransformsElements()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var transformed = vector.Select(x => x * 2);

        // Assert
        Assert.Equal(3, transformed.Length);
        Assert.Equal(2.0, transformed[0], Tolerance);
        Assert.Equal(4.0, transformed[1], Tolerance);
        Assert.Equal(6.0, transformed[2], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_Transform_AppliesFunction()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 1.0, 4.0, 9.0 });

        // Act
        var transformed = vector.Transform<double>(Math.Sqrt);

        // Assert
        Assert.Equal(3, transformed.Length);
        Assert.Equal(1.0, transformed[0], Tolerance);
        Assert.Equal(2.0, transformed[1], Tolerance);
        Assert.Equal(3.0, transformed[2], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_TransformWithIndex_AppliesFunctionWithIndex()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 10.0, 20.0, 30.0 });

        // Act
        var transformed = vector.Transform<double>((value, index) => value + index);

        // Assert
        Assert.Equal(3, transformed.Length);
        Assert.Equal(10.0, transformed[0], Tolerance); // 10 + 0
        Assert.Equal(21.0, transformed[1], Tolerance); // 20 + 1
        Assert.Equal(32.0, transformed[2], Tolerance); // 30 + 2
    }

    #endregion

    #region Vector Aggregation Methods

    [Fact(Timeout = 120000)]
    public async Task Vector_Mean_ComputesCorrectMean()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 2.0, 4.0, 6.0, 8.0 });

        // Act
        var mean = vector.Mean();

        // Assert
        Assert.Equal(5.0, mean, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_Sum_ComputesCorrectSum()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

        // Act
        var sum = vector.Sum();

        // Assert
        Assert.Equal(10.0, sum, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_L2Norm_ComputesCorrectNorm()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 3.0, 4.0 });

        // Act
        var l2Norm = vector.L2Norm();

        // Assert
        Assert.Equal(5.0, l2Norm, Tolerance); // sqrt(9 + 16) = 5
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_NonZeroCount_CountsNonZeros()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 0.0, 1.0, 0.0, 2.0, 3.0, 0.0 });

        // Act
        var count = vector.NonZeroCount();

        // Assert
        Assert.Equal(3, count);
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_NonZeroIndices_ReturnsCorrectIndices()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 0.0, 1.0, 0.0, 2.0, 0.0 });

        // Act
        var indices = vector.NonZeroIndices().ToList();

        // Assert
        Assert.Equal(2, indices.Count);
        Assert.Contains(1, indices);
        Assert.Contains(3, indices);
    }

    #endregion

    #region Vector Indexing and Search Methods

    [Fact(Timeout = 120000)]
    public async Task Vector_IndexOf_FindsElement()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        // Act
        var index = vector.IndexOf(3.0);

        // Assert
        Assert.Equal(2, index);
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_IndexOf_ReturnsMinusOneWhenNotFound()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var index = vector.IndexOf(99.0);

        // Assert
        Assert.Equal(-1, index);
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_BinarySearch_FindsElementInSortedVector()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        // Act
        var index = vector.BinarySearch(3.0);

        // Assert
        Assert.Equal(2, index);
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_SetValue_SetsValueAndReturnsVector()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var result = vector.SetValue(1, 99.0);

        // Assert
        Assert.Equal(99.0, result[1]);
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_Fill_FillsWithValue()
    {
        // Arrange
        var vector = new Vector<double>(5);

        // Act
        vector.Fill(7.0);

        // Assert
        for (int i = 0; i < vector.Length; i++)
        {
            Assert.Equal(7.0, vector[i]);
        }
    }

    #endregion

    #region Vector Segment and Range Methods

    [Fact(Timeout = 120000)]
    public async Task Vector_Subvector_ReturnsCorrectSubvector()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        // Act
        var subvector = vector.Subvector(1, 3);

        // Assert
        Assert.Equal(3, subvector.Length);
        Assert.Equal(2.0, subvector[0]);
        Assert.Equal(3.0, subvector[1]);
        Assert.Equal(4.0, subvector[2]);
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_GetRange_ReturnsCorrectRange()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });

        // Act
        var range = vector.GetRange(2, 2);

        // Assert
        Assert.Equal(2, range.Length);
        Assert.Equal(30.0, range[0]);
        Assert.Equal(40.0, range[1]);
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_GetSegment_ReturnsCorrectSegment()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        // Act
        var segment = vector.GetSegment(1, 3);

        // Assert
        Assert.Equal(3, segment.Length);
        Assert.Equal(2.0, segment[0]);
        Assert.Equal(4.0, segment[2]);
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_GetElements_ReturnsElementsAtIndices()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });
        var indices = new[] { 0, 2, 4 };

        // Act
        var elements = vector.GetElements(indices);

        // Assert
        Assert.Equal(3, elements.Length);
        Assert.Equal(10.0, elements[0]);
        Assert.Equal(30.0, elements[1]);
        Assert.Equal(50.0, elements[2]);
    }

    #endregion

    #region Vector Conversion Methods

    [Fact(Timeout = 120000)]
    public async Task Vector_ToArray_ReturnsCorrectArray()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var array = vector.ToArray();

        // Assert
        Assert.Equal(3, array.Length);
        Assert.Equal(1.0, array[0]);
        Assert.Equal(3.0, array[2]);
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_AsSpan_ReturnsSpan()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var span = vector.AsSpan();

        // Assert
        Assert.Equal(3, span.Length);
        Assert.Equal(1.0, span[0]);
        Assert.Equal(3.0, span[2]);
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_Transpose_CreatesColumnMatrix()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var matrix = vector.Transpose();

        // Assert
        Assert.Equal(3, matrix.Rows);
        Assert.Equal(1, matrix.Columns);
        Assert.Equal(1.0, matrix[0, 0]);
        Assert.Equal(2.0, matrix[1, 0]);
        Assert.Equal(3.0, matrix[2, 0]);
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_AppendAsMatrix_CreatesMatrixWithAppendedValue()
    {
        // Arrange - Standard ML: append bias term to feature vector
        // Vector [1, 2] + value 3 → row matrix [[1, 2, 3]]
        var vector = new Vector<double>(new[] { 1.0, 2.0 });

        // Act
        var matrix = vector.AppendAsMatrix(3.0);

        // Assert - Result is 1×(N+1) row matrix
        Assert.Equal(1, matrix.Rows);
        Assert.Equal(3, matrix.Columns);
        Assert.Equal(1.0, matrix[0, 0]);
        Assert.Equal(2.0, matrix[0, 1]);
        Assert.Equal(3.0, matrix[0, 2]);
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_ImplicitConversionToArray_Works()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        double[] array = vector;

        // Assert
        Assert.Equal(3, array.Length);
        Assert.Equal(1.0, array[0]);
    }

    #endregion

    #region Vector Serialization Tests

    [Fact(Timeout = 120000)]
    public async Task Vector_SerializeDeserialize_RoundTripsCorrectly()
    {
        // Arrange
        var original = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        // Act
        var serialized = original.Serialize();
        var deserialized = Vector<double>.Deserialize(serialized);

        // Assert
        Assert.Equal(original.Length, deserialized.Length);
        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], deserialized[i], Tolerance);
        }
    }

    #endregion

    #region Matrix Additional Factory Methods

    [Fact(Timeout = 120000)]
    public async Task Matrix_Empty_CreatesEmptyMatrix()
    {
        // Arrange & Act
        var matrix = Matrix<double>.Empty();

        // Assert
        Assert.NotNull(matrix);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_FromVector_CreatesMatrixFromVector()
    {
        // Arrange
        var vector = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var matrix = Matrix<double>.FromVector(vector);

        // Assert
        Assert.Equal(3, matrix.Rows);
        Assert.Equal(1, matrix.Columns);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_FromRows_CreatesMatrixFromRowVectors()
    {
        // Arrange
        var row1 = new double[] { 1.0, 2.0, 3.0 };
        var row2 = new double[] { 4.0, 5.0, 6.0 };

        // Act
        var matrix = Matrix<double>.FromRows(row1, row2);

        // Assert
        Assert.Equal(2, matrix.Rows);
        Assert.Equal(3, matrix.Columns);
        Assert.Equal(1.0, matrix[0, 0]);
        Assert.Equal(6.0, matrix[1, 2]);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_FromColumns_CreatesMatrixFromColumnVectors()
    {
        // Arrange
        var col1 = new double[] { 1.0, 4.0 };
        var col2 = new double[] { 2.0, 5.0 };
        var col3 = new double[] { 3.0, 6.0 };

        // Act
        var matrix = Matrix<double>.FromColumns(col1, col2, col3);

        // Assert
        Assert.Equal(2, matrix.Rows);
        Assert.Equal(3, matrix.Columns);
        Assert.Equal(1.0, matrix[0, 0]);
        Assert.Equal(2.0, matrix[0, 1]);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_FromRowVectors_CreatesMatrixFromEnumerable()
    {
        // Arrange
        var rows = new List<List<double>>
        {
            new() { 1.0, 2.0 },
            new() { 3.0, 4.0 },
            new() { 5.0, 6.0 }
        };

        // Act
        var matrix = Matrix<double>.FromRowVectors(rows);

        // Assert
        Assert.Equal(3, matrix.Rows);
        Assert.Equal(2, matrix.Columns);
        Assert.Equal(5.0, matrix[2, 0]);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_FromColumnVectors_CreatesMatrixFromEnumerable()
    {
        // Arrange
        var cols = new List<List<double>>
        {
            new() { 1.0, 2.0, 3.0 },
            new() { 4.0, 5.0, 6.0 }
        };

        // Act
        var matrix = Matrix<double>.FromColumnVectors(cols);

        // Assert
        Assert.Equal(3, matrix.Rows);
        Assert.Equal(2, matrix.Columns);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_CreateDefault_CreatesMatrixWithDefaultValue()
    {
        // Arrange & Act
        var matrix = Matrix<double>.CreateDefault(3, 4, 2.5);

        // Assert
        Assert.Equal(3, matrix.Rows);
        Assert.Equal(4, matrix.Columns);
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                Assert.Equal(2.5, matrix[i, j]);
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_BlockDiagonal_CreatesBlockDiagonalMatrix()
    {
        // Arrange
        var m1 = new Matrix<double>(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        var m2 = new Matrix<double>(new double[,] { { 5.0 } });

        // Act
        var blockDiag = Matrix<double>.BlockDiagonal(m1, m2);

        // Assert
        Assert.Equal(3, blockDiag.Rows);
        Assert.Equal(3, blockDiag.Columns);
        Assert.Equal(1.0, blockDiag[0, 0]);
        Assert.Equal(5.0, blockDiag[2, 2]);
        Assert.Equal(0.0, blockDiag[0, 2]);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_OuterProduct_ComputesCorrectOuterProduct()
    {
        // Arrange
        var a = new Vector<double>(new[] { 1.0, 2.0 });
        var b = new Vector<double>(new[] { 3.0, 4.0, 5.0 });

        // Act
        var outer = Matrix<double>.OuterProduct(a, b);

        // Assert
        Assert.Equal(2, outer.Rows);
        Assert.Equal(3, outer.Columns);
        Assert.Equal(3.0, outer[0, 0], Tolerance);  // 1 * 3
        Assert.Equal(10.0, outer[1, 2], Tolerance); // 2 * 5
    }

    #endregion

    #region Matrix Row/Column Segment Methods

    [Fact(Timeout = 120000)]
    public async Task Matrix_GetColumnSegment_ReturnsCorrectSegment()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
            { 5.0, 6.0 },
            { 7.0, 8.0 }
        });

        // Act
        var segment = matrix.GetColumnSegment(0, 1, 2);

        // Assert
        Assert.Equal(2, segment.Length);
        Assert.Equal(3.0, segment[0]);
        Assert.Equal(5.0, segment[1]);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_GetRowSegment_ReturnsCorrectSegment()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0 },
            { 5.0, 6.0, 7.0, 8.0 }
        });

        // Act
        var segment = matrix.GetRowSegment(0, 1, 2);

        // Assert
        Assert.Equal(2, segment.Length);
        Assert.Equal(2.0, segment[0]);
        Assert.Equal(3.0, segment[1]);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_SetSubMatrix_SetsSubMatrixCorrectly()
    {
        // Arrange
        var matrix = Matrix<double>.CreateZeros(4, 4);
        var subMatrix = new Matrix<double>(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });

        // Act
        matrix.SetSubMatrix(1, 1, subMatrix);

        // Assert
        Assert.Equal(1.0, matrix[1, 1]);
        Assert.Equal(2.0, matrix[1, 2]);
        Assert.Equal(3.0, matrix[2, 1]);
        Assert.Equal(4.0, matrix[2, 2]);
        Assert.Equal(0.0, matrix[0, 0]);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_Slice_ReturnsCorrectSlice()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 },
            { 10.0, 11.0, 12.0 }
        });

        // Act
        var slice = matrix.Slice(1, 2);

        // Assert
        Assert.Equal(2, slice.Rows);
        Assert.Equal(3, slice.Columns);
        Assert.Equal(4.0, slice[0, 0]);
        Assert.Equal(9.0, slice[1, 2]);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_GetRows_ReturnsRowsAtIndices()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
            { 5.0, 6.0 },
            { 7.0, 8.0 }
        });

        // Act
        var selectedRows = matrix.GetRows(new[] { 0, 2 });

        // Assert
        Assert.Equal(2, selectedRows.Rows);
        Assert.Equal(2, selectedRows.Columns);
        Assert.Equal(1.0, selectedRows[0, 0]);
        Assert.Equal(5.0, selectedRows[1, 0]);
    }

    #endregion

    #region Matrix Conversion Methods

    [Fact(Timeout = 120000)]
    public async Task Matrix_ToColumnVector_ConvertsToColumnVector()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        });

        // Act
        var vector = matrix.ToColumnVector();

        // Assert
        Assert.Equal(4, vector.Length);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_ToRowVector_ConvertsToRowVector()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        });

        // Act
        var vector = matrix.ToRowVector();

        // Assert
        Assert.Equal(4, vector.Length);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_GetColumns_ReturnsAllColumns()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 }
        });

        // Act
        var columns = matrix.GetColumns().ToList();

        // Assert
        Assert.Equal(3, columns.Count);
        Assert.Equal(2, columns[0].Length);
        Assert.Equal(1.0, columns[0][0]);
        Assert.Equal(4.0, columns[0][1]);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_GetRows_ReturnsAllRows()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 }
        });

        // Act
        var rows = matrix.GetRows().ToList();

        // Assert
        Assert.Equal(2, rows.Count);
        Assert.Equal(3, rows[0].Length);
        Assert.Equal(1.0, rows[0][0]);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_GetRowSpan_ReturnsSpanForRow()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 }
        });

        // Act
        var span = matrix.GetRowSpan(1);

        // Assert
        Assert.Equal(3, span.Length);
        Assert.Equal(4.0, span[0]);
        Assert.Equal(6.0, span[2]);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_GetRowReadOnlySpan_ReturnsReadOnlySpan()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 }
        });

        // Act
        var span = matrix.GetRowReadOnlySpan(0);

        // Assert
        Assert.Equal(3, span.Length);
        Assert.Equal(1.0, span[0]);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_GetColumnAsArray_ReturnsColumnArray()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
            { 5.0, 6.0 }
        });

        // Act
        var column = matrix.GetColumnAsArray(1);

        // Assert
        Assert.Equal(3, column.Length);
        Assert.Equal(2.0, column[0]);
        Assert.Equal(4.0, column[1]);
        Assert.Equal(6.0, column[2]);
    }

    #endregion

    #region Matrix Aggregation Methods

    [Fact(Timeout = 120000)]
    public async Task Matrix_RowWiseMax_ReturnsMaxPerRow()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 5.0, 3.0 },
            { 9.0, 2.0, 4.0 }
        });

        // Act
        var maxes = matrix.RowWiseMax();

        // Assert
        Assert.Equal(2, maxes.Length);
        Assert.Equal(5.0, maxes[0], Tolerance);
        Assert.Equal(9.0, maxes[1], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_RowWiseSum_ReturnsSumPerRow()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 }
        });

        // Act
        var sums = matrix.RowWiseSum();

        // Assert
        Assert.Equal(2, sums.Length);
        Assert.Equal(6.0, sums[0], Tolerance);
        Assert.Equal(15.0, sums[1], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_Transform_AppliesFunctionToAllElements()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 4.0 },
            { 9.0, 16.0 }
        });

        // Act
        var transformed = matrix.Transform((val, row, col) => Math.Sqrt(val));

        // Assert
        Assert.Equal(1.0, transformed[0, 0], Tolerance);
        Assert.Equal(2.0, transformed[0, 1], Tolerance);
        Assert.Equal(3.0, transformed[1, 0], Tolerance);
        Assert.Equal(4.0, transformed[1, 1], Tolerance);
    }

    #endregion

    #region Matrix Serialization Tests

    [Fact(Timeout = 120000)]
    public async Task Matrix_SerializeDeserialize_RoundTripsCorrectly()
    {
        // Arrange
        var original = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 }
        });

        // Act
        var serialized = original.Serialize();
        var deserialized = Matrix<double>.Deserialize(serialized);

        // Assert
        Assert.Equal(original.Rows, deserialized.Rows);
        Assert.Equal(original.Columns, deserialized.Columns);
        for (int i = 0; i < original.Rows; i++)
        {
            for (int j = 0; j < original.Columns; j++)
            {
                Assert.Equal(original[i, j], deserialized[i, j], Tolerance);
            }
        }
    }

    #endregion

    #region Matrix Division Operator Tests

    [Fact(Timeout = 120000)]
    public async Task Matrix_DivisionOperator_DividesTwoMatrices()
    {
        // Arrange
        var m1 = new Matrix<double>(new double[,] { { 4.0, 6.0 }, { 8.0, 10.0 } });
        var m2 = new Matrix<double>(new double[,] { { 2.0, 2.0 }, { 4.0, 5.0 } });

        // Act
        var result = m1 / m2;

        // Assert
        Assert.Equal(2.0, result[0, 0], Tolerance);
        Assert.Equal(3.0, result[0, 1], Tolerance);
        Assert.Equal(2.0, result[1, 0], Tolerance);
        Assert.Equal(2.0, result[1, 1], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_DivideMethod_DividesByScalar()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,] { { 4.0, 8.0 }, { 12.0, 16.0 } });

        // Act
        var result = matrix.Divide(4.0);

        // Assert
        Assert.Equal(1.0, result[0, 0], Tolerance);
        Assert.Equal(2.0, result[0, 1], Tolerance);
        Assert.Equal(3.0, result[1, 0], Tolerance);
        Assert.Equal(4.0, result[1, 1], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_DivideMethod_DividesByMatrix()
    {
        // Arrange
        var m1 = new Matrix<double>(new double[,] { { 10.0, 20.0 }, { 30.0, 40.0 } });
        var m2 = new Matrix<double>(new double[,] { { 2.0, 4.0 }, { 5.0, 8.0 } });

        // Act
        var result = m1.Divide(m2);

        // Assert
        Assert.Equal(5.0, result[0, 0], Tolerance);
        Assert.Equal(5.0, result[0, 1], Tolerance);
        Assert.Equal(6.0, result[1, 0], Tolerance);
        Assert.Equal(5.0, result[1, 1], Tolerance);
    }

    #endregion

    #region Matrix SetRow/SetColumn Tests

    [Fact(Timeout = 120000)]
    public async Task Matrix_SetRow_SetsRowCorrectly()
    {
        // Arrange
        var matrix = Matrix<double>.CreateZeros(3, 3);
        var newRow = new Vector<double>(new[] { 7.0, 8.0, 9.0 });

        // Act
        matrix.SetRow(1, newRow);

        // Assert
        Assert.Equal(7.0, matrix[1, 0]);
        Assert.Equal(8.0, matrix[1, 1]);
        Assert.Equal(9.0, matrix[1, 2]);
        Assert.Equal(0.0, matrix[0, 0]); // Other rows unchanged
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_SetColumn_SetsColumnCorrectly()
    {
        // Arrange
        var matrix = Matrix<double>.CreateZeros(3, 3);
        var newCol = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        matrix.SetColumn(1, newCol);

        // Assert
        Assert.Equal(1.0, matrix[0, 1]);
        Assert.Equal(2.0, matrix[1, 1]);
        Assert.Equal(3.0, matrix[2, 1]);
        Assert.Equal(0.0, matrix[0, 0]); // Other columns unchanged
    }

    #endregion

    #region Matrix Diagonal and SubMatrix Tests

    [Fact(Timeout = 120000)]
    public async Task Matrix_Diagonal_ReturnsDiagonalElements()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 }
        });

        // Act
        var diagonal = matrix.Diagonal();

        // Assert
        Assert.Equal(3, diagonal.Length);
        Assert.Equal(1.0, diagonal[0]);
        Assert.Equal(5.0, diagonal[1]);
        Assert.Equal(9.0, diagonal[2]);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_SubMatrix_ReturnsCorrectSubMatrix()
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
        var subMatrix = matrix.SubMatrix(1, 1, 2, 2);

        // Assert
        Assert.Equal(2, subMatrix.Rows);
        Assert.Equal(2, subMatrix.Columns);
        Assert.Equal(6.0, subMatrix[0, 0]);
        Assert.Equal(7.0, subMatrix[0, 1]);
        Assert.Equal(10.0, subMatrix[1, 0]);
        Assert.Equal(11.0, subMatrix[1, 1]);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_SubMatrixWithColumnIndices_ReturnsCorrectSubMatrix()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0 },
            { 5.0, 6.0, 7.0, 8.0 },
            { 9.0, 10.0, 11.0, 12.0 }
        });

        // Act - Get rows 0-1 (endRow=2 is exclusive), columns 0 and 2
        var subMatrix = matrix.SubMatrix(0, 2, new List<int> { 0, 2 });

        // Assert
        Assert.Equal(2, subMatrix.Rows);
        Assert.Equal(2, subMatrix.Columns);
        Assert.Equal(1.0, subMatrix[0, 0]);
        Assert.Equal(3.0, subMatrix[0, 1]);
        Assert.Equal(5.0, subMatrix[1, 0]);
        Assert.Equal(7.0, subMatrix[1, 1]);
    }

    #endregion

    #region Matrix ElementWise Operations

    [Fact(Timeout = 120000)]
    public async Task Matrix_ElementWiseMultiplyAndSum_ComputesCorrectly()
    {
        // Arrange - Frobenius inner product: sum(A .* B)
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
        var result = m1.ElementWiseMultiplyAndSum(m2);

        // Assert: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        Assert.Equal(70.0, result, Tolerance);
    }

    #endregion

    #region Matrix Span Operations

    [Fact(Timeout = 120000)]
    public async Task Matrix_AsSpan_ReturnsCorrectSpan()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        });

        // Act
        var span = matrix.AsSpan();

        // Assert - Matrix is stored in row-major order
        Assert.Equal(4, span.Length);
        Assert.Equal(1.0, span[0]);
        Assert.Equal(2.0, span[1]);
        Assert.Equal(3.0, span[2]);
        Assert.Equal(4.0, span[3]);
    }

    #endregion

    #region Matrix GetRow Tests

    [Fact(Timeout = 120000)]
    public async Task Matrix_GetRow_ReturnsCorrectRow()
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

    #endregion

    #region Matrix Add with Tensor

    [Fact(Timeout = 120000)]
    public async Task Matrix_AddTensor_AddsCorrectly()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        });
        var tensor = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 10.0, 20.0, 30.0, 40.0 }));

        // Act
        var result = matrix.Add(tensor);

        // Assert
        Assert.Equal(11.0, result[0, 0], Tolerance);
        Assert.Equal(22.0, result[0, 1], Tolerance);
        Assert.Equal(33.0, result[1, 0], Tolerance);
        Assert.Equal(44.0, result[1, 1], Tolerance);
    }

    #endregion

    #region Vector Static Properties

    [Fact(Timeout = 120000)]
    public async Task Vector_IsCpuAccelerated_ReturnsBoolean()
    {
        // Act - just verify it doesn't throw and returns a valid boolean
        var isCpuAccelerated = Vector<double>.IsCpuAccelerated;

        // Assert - just checking it's a valid value
        Assert.True(isCpuAccelerated || !isCpuAccelerated); // Always true, just checking it runs
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_IsGpuAccelerated_ReturnsBoolean()
    {
        // Act
        var isGpuAccelerated = Vector<double>.IsGpuAccelerated;

        // Assert
        Assert.True(isGpuAccelerated || !isGpuAccelerated);
    }

    [Fact(Timeout = 120000)]
    public async Task Vector_SimdVectorCount_ReturnsPositive()
    {
        // Act
        var simdCount = Vector<double>.SimdVectorCount;

        // Assert - Should be >= 1 (at least single element)
        Assert.True(simdCount >= 1);
    }

    #endregion

    #region Matrix Factory Methods

    [Fact(Timeout = 120000)]
    public async Task Matrix_Ones_CreatesMatrixOfOnes()
    {
        // Arrange
        var matrix = new Matrix<double>(3, 4);

        // Act - Ones requires rows and cols parameters
        var result = matrix.Ones(3, 4);

        // Assert
        Assert.Equal(3, result.Rows);
        Assert.Equal(4, result.Columns);
        for (int i = 0; i < result.Rows; i++)
            for (int j = 0; j < result.Columns; j++)
                Assert.Equal(1.0, result[i, j]);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_Zeros_CreatesMatrixOfZeros()
    {
        // Arrange
        var matrix = new Matrix<double>(3, 4);
        // Fill with non-zero values first
        for (int i = 0; i < matrix.Rows; i++)
            for (int j = 0; j < matrix.Columns; j++)
                matrix[i, j] = 99.0;

        // Act - Zeros requires rows and cols parameters
        var result = matrix.Zeros(3, 4);

        // Assert
        Assert.Equal(3, result.Rows);
        Assert.Equal(4, result.Columns);
        for (int i = 0; i < result.Rows; i++)
            for (int j = 0; j < result.Columns; j++)
                Assert.Equal(0.0, result[i, j]);
    }

    [Fact(Timeout = 120000)]
    public async Task Matrix_CreateMatrix_GenericFactory_CreatesCorrectDimensions()
    {
        // Act - CreateMatrix is static
        var result = Matrix<double>.CreateMatrix<float>(4, 5);

        // Assert
        Assert.Equal(4, result.Rows);
        Assert.Equal(5, result.Columns);
    }

    #endregion
}
