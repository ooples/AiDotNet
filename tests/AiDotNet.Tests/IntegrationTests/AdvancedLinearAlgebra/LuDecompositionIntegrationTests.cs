using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Enums.AlgorithmTypes;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AdvancedLinearAlgebra;

/// <summary>
/// Integration tests for LU decomposition that verify mathematical correctness.
/// These tests verify: P*A = L*U, L is lower triangular with 1s on diagonal, U is upper triangular.
/// </summary>
public class LuDecompositionIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region Helper Methods

    private static Matrix<double> CreateTestMatrix(int size, int seed = 42)
    {
        var random = new Random(seed);
        var matrix = new Matrix<double>(size, size);
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                matrix[i, j] = random.NextDouble() * 10 - 5;
            }
        }
        return matrix;
    }

    private static Matrix<double> ApplyPermutation(Matrix<double> matrix, Vector<int> p)
    {
        int n = matrix.Rows;
        var result = new Matrix<double>(n, matrix.Columns);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                result[i, j] = matrix[p[i], j];
            }
        }
        return result;
    }

    private static bool IsLowerTriangular(Matrix<double> matrix)
    {
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = i + 1; j < matrix.Columns; j++)
            {
                if (Math.Abs(matrix[i, j]) > Tolerance)
                    return false;
            }
        }
        return true;
    }

    private static bool IsUpperTriangular(Matrix<double> matrix)
    {
        for (int i = 1; i < matrix.Rows; i++)
        {
            for (int j = 0; j < i && j < matrix.Columns; j++)
            {
                if (Math.Abs(matrix[i, j]) > Tolerance)
                    return false;
            }
        }
        return true;
    }

    private static bool HasUnitDiagonal(Matrix<double> matrix)
    {
        int minDim = Math.Min(matrix.Rows, matrix.Columns);
        for (int i = 0; i < minDim; i++)
        {
            if (Math.Abs(matrix[i, i] - 1.0) > Tolerance)
                return false;
        }
        return true;
    }

    private static double MaxAbsDiff(Matrix<double> a, Matrix<double> b)
    {
        double maxDiff = 0;
        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < a.Columns; j++)
            {
                maxDiff = Math.Max(maxDiff, Math.Abs(a[i, j] - b[i, j]));
            }
        }
        return maxDiff;
    }

    #endregion

    #region Decomposition Property Tests

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(10)]
    public void LuDecomposition_PartialPivoting_PA_Equals_LU(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size);

        // Act
        var lu = new LuDecomposition<double>(A, LuAlgorithmType.PartialPivoting);

        // Assert - Verify P*A = L*U
        var PA = ApplyPermutation(A, lu.P);
        var LU = lu.L.Multiply(lu.U);
        double maxDiff = MaxAbsDiff(PA, LU);

        Assert.True(maxDiff < Tolerance,
            $"P*A should equal L*U. Max difference: {maxDiff}");
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(5)]
    public void LuDecomposition_PartialPivoting_L_IsLowerTriangular(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size);

        // Act
        var lu = new LuDecomposition<double>(A, LuAlgorithmType.PartialPivoting);

        // Assert
        Assert.True(IsLowerTriangular(lu.L), "L should be lower triangular");
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(5)]
    public void LuDecomposition_PartialPivoting_L_HasUnitDiagonal(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size);

        // Act
        var lu = new LuDecomposition<double>(A, LuAlgorithmType.PartialPivoting);

        // Assert
        Assert.True(HasUnitDiagonal(lu.L), "L should have 1s on the diagonal");
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(5)]
    public void LuDecomposition_PartialPivoting_U_IsUpperTriangular(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size);

        // Act
        var lu = new LuDecomposition<double>(A, LuAlgorithmType.PartialPivoting);

        // Assert
        Assert.True(IsUpperTriangular(lu.U), "U should be upper triangular");
    }

    [Theory]
    [InlineData(LuAlgorithmType.Doolittle)]
    [InlineData(LuAlgorithmType.Crout)]
    [InlineData(LuAlgorithmType.PartialPivoting)]
    [InlineData(LuAlgorithmType.CompletePivoting)]
    public void LuDecomposition_AllAlgorithms_ProduceValidDecomposition(LuAlgorithmType algorithm)
    {
        // Arrange
        var A = CreateTestMatrix(4, seed: 123);

        // Act
        var lu = new LuDecomposition<double>(A, algorithm);

        // Assert - L*U should reconstruct original (or permuted) matrix
        var LU = lu.L.Multiply(lu.U);

        // For algorithms with pivoting, compare P*A to L*U
        // For algorithms without pivoting, compare A to L*U directly
        Matrix<double> expected = algorithm == LuAlgorithmType.PartialPivoting ||
                                   algorithm == LuAlgorithmType.CompletePivoting
            ? ApplyPermutation(A, lu.P)
            : A;

        double maxDiff = MaxAbsDiff(expected, LU);
        Assert.True(maxDiff < 1e-6,
            $"Algorithm {algorithm}: Max difference = {maxDiff}");
    }

    #endregion

    #region Linear System Solving Tests

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(10)]
    public void LuDecomposition_Solve_ProducesCorrectSolution(int size)
    {
        // Arrange - Create a system with known solution
        var A = CreateTestMatrix(size, seed: 42);
        var xExpected = new Vector<double>(size);
        for (int i = 0; i < size; i++)
            xExpected[i] = i + 1.0;

        var b = A.Multiply(xExpected);

        // Act
        var lu = new LuDecomposition<double>(A, LuAlgorithmType.PartialPivoting);
        var xComputed = lu.Solve(b);

        // Assert - Verify A*x_computed ≈ b
        var bComputed = A.Multiply(xComputed);
        for (int i = 0; i < size; i++)
        {
            Assert.True(Math.Abs(bComputed[i] - b[i]) < 1e-8,
                $"A*x should equal b. Component {i}: expected {b[i]}, got {bComputed[i]}");
        }
    }

    [Fact]
    public void LuDecomposition_Solve_IdentityMatrix_ReturnsSameVector()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(4);
        var b = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

        // Act
        var lu = new LuDecomposition<double>(I);
        var x = lu.Solve(b);

        // Assert
        for (int i = 0; i < 4; i++)
        {
            Assert.True(Math.Abs(x[i] - b[i]) < Tolerance,
                $"Solution for identity matrix should be the input vector. Index {i}");
        }
    }

    [Fact]
    public void LuDecomposition_Solve_DiagonalMatrix_CorrectDivision()
    {
        // Arrange - Diagonal matrix with values 1, 2, 3, 4
        var D = new Matrix<double>(4, 4);
        for (int i = 0; i < 4; i++)
            D[i, i] = i + 1.0;

        var b = new Vector<double>(new[] { 2.0, 6.0, 12.0, 20.0 });
        // Expected solution: x[i] = b[i] / D[i,i] = {2, 3, 4, 5}

        // Act
        var lu = new LuDecomposition<double>(D);
        var x = lu.Solve(b);

        // Assert
        var expected = new[] { 2.0, 3.0, 4.0, 5.0 };
        for (int i = 0; i < 4; i++)
        {
            Assert.True(Math.Abs(x[i] - expected[i]) < Tolerance,
                $"Index {i}: expected {expected[i]}, got {x[i]}");
        }
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void LuDecomposition_2x2Matrix_CorrectDecomposition()
    {
        // Arrange - Simple 2x2 matrix
        var A = new Matrix<double>(2, 2);
        A[0, 0] = 4; A[0, 1] = 3;
        A[1, 0] = 6; A[1, 1] = 3;

        // Act
        var lu = new LuDecomposition<double>(A, LuAlgorithmType.PartialPivoting);

        // Assert
        var PA = ApplyPermutation(A, lu.P);
        var LU = lu.L.Multiply(lu.U);
        double maxDiff = MaxAbsDiff(PA, LU);

        Assert.True(maxDiff < Tolerance,
            $"2x2 decomposition failed. Max difference: {maxDiff}");
    }

    [Fact]
    public void LuDecomposition_SymmetricMatrix_ValidDecomposition()
    {
        // Arrange - Symmetric positive definite matrix
        var A = new Matrix<double>(3, 3);
        A[0, 0] = 4; A[0, 1] = 2; A[0, 2] = 1;
        A[1, 0] = 2; A[1, 1] = 5; A[1, 2] = 2;
        A[2, 0] = 1; A[2, 1] = 2; A[2, 2] = 6;

        // Act
        var lu = new LuDecomposition<double>(A);

        // Assert
        var PA = ApplyPermutation(A, lu.P);
        var LU = lu.L.Multiply(lu.U);
        double maxDiff = MaxAbsDiff(PA, LU);

        Assert.True(maxDiff < Tolerance,
            $"Symmetric matrix decomposition failed. Max difference: {maxDiff}");
    }

    [Fact]
    public void LuDecomposition_NonSquareMatrix_ThrowsArgumentException()
    {
        // Arrange
        var A = new Matrix<double>(3, 4);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new LuDecomposition<double>(A));
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void LuDecomposition_IllConditionedMatrix_StillDecomposes()
    {
        // Arrange - Create a moderately ill-conditioned matrix (Hilbert-like)
        int n = 4;
        var A = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                A[i, j] = 1.0 / (i + j + 1);
            }
        }

        // Act
        var lu = new LuDecomposition<double>(A, LuAlgorithmType.PartialPivoting);

        // Assert - Even for ill-conditioned matrices, decomposition should be valid
        var PA = ApplyPermutation(A, lu.P);
        var LU = lu.L.Multiply(lu.U);
        double maxDiff = MaxAbsDiff(PA, LU);

        // Use looser tolerance for ill-conditioned matrices
        Assert.True(maxDiff < 1e-6,
            $"Ill-conditioned matrix decomposition failed. Max difference: {maxDiff}");
    }

    [Fact]
    public void LuDecomposition_LargeMatrix_CorrectDecomposition()
    {
        // Arrange
        int size = 50;
        var A = CreateTestMatrix(size, seed: 999);

        // Act
        var lu = new LuDecomposition<double>(A, LuAlgorithmType.PartialPivoting);

        // Assert
        var PA = ApplyPermutation(A, lu.P);
        var LU = lu.L.Multiply(lu.U);
        double maxDiff = MaxAbsDiff(PA, LU);

        Assert.True(maxDiff < 1e-8,
            $"Large matrix decomposition failed. Max difference: {maxDiff}");
    }

    #endregion
}
