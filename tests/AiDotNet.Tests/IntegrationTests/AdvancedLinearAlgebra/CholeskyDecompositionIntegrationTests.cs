using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Enums.AlgorithmTypes;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AdvancedLinearAlgebra;

/// <summary>
/// Integration tests for Cholesky decomposition that verify mathematical correctness.
/// These tests verify: A = L * L^T for symmetric positive definite matrices, L is lower triangular.
/// </summary>
public class CholeskyDecompositionIntegrationTests
{
    private const double Tolerance = 1e-10;
    private const double LooseTolerance = 1e-8;

    #region Helper Methods

    /// <summary>
    /// Creates a symmetric positive definite matrix using A^T * A + I technique.
    /// </summary>
    private static Matrix<double> CreateSpdMatrix(int size, int seed = 42)
    {
        var random = new Random(seed);
        var A = new Matrix<double>(size, size);

        // Create a random matrix
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                A[i, j] = random.NextDouble() * 2 - 1;
            }
        }

        // A^T * A is guaranteed to be positive semi-definite
        // Adding identity makes it positive definite
        var AtA = A.Transpose().Multiply(A);
        for (int i = 0; i < size; i++)
        {
            AtA[i, i] += size; // Add n*I to ensure positive definiteness
        }

        return AtA;
    }

    /// <summary>
    /// Creates a known symmetric positive definite matrix for exact testing.
    /// </summary>
    private static Matrix<double> CreateKnownSpdMatrix()
    {
        // A well-known SPD matrix
        var A = new Matrix<double>(3, 3);
        A[0, 0] = 4; A[0, 1] = 2; A[0, 2] = 1;
        A[1, 0] = 2; A[1, 1] = 5; A[1, 2] = 2;
        A[2, 0] = 1; A[2, 1] = 2; A[2, 2] = 6;
        return A;
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

    private static bool IsSymmetric(Matrix<double> matrix)
    {
        if (matrix.Rows != matrix.Columns)
            return false;

        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = i + 1; j < matrix.Columns; j++)
            {
                if (Math.Abs(matrix[i, j] - matrix[j, i]) > Tolerance)
                    return false;
            }
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
    public void CholeskyDecomposition_A_Equals_LLT(int size)
    {
        // Arrange
        var A = CreateSpdMatrix(size);

        // Act
        var chol = new CholeskyDecomposition<double>(A);

        // Assert - Verify A = L * L^T
        var LLt = chol.L.Multiply(chol.L.Transpose());
        double maxDiff = MaxAbsDiff(A, LLt);

        Assert.True(maxDiff < LooseTolerance,
            $"A should equal L*L^T. Max difference: {maxDiff}");
    }

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(5)]
    public void CholeskyDecomposition_L_IsLowerTriangular(int size)
    {
        // Arrange
        var A = CreateSpdMatrix(size);

        // Act
        var chol = new CholeskyDecomposition<double>(A);

        // Assert
        Assert.True(IsLowerTriangular(chol.L),
            "L should be lower triangular");
    }

    [Fact]
    public void CholeskyDecomposition_L_HasPositiveDiagonal()
    {
        // Arrange
        var A = CreateSpdMatrix(5);

        // Act
        var chol = new CholeskyDecomposition<double>(A);

        // Assert - Diagonal elements should be positive
        for (int i = 0; i < chol.L.Rows; i++)
        {
            Assert.True(chol.L[i, i] > 0,
                $"Diagonal element L[{i},{i}] = {chol.L[i, i]} should be positive");
        }
    }

    [Theory]
    [InlineData(CholeskyAlgorithmType.Crout)]
    [InlineData(CholeskyAlgorithmType.Banachiewicz)]
    [InlineData(CholeskyAlgorithmType.LDL)]
    [InlineData(CholeskyAlgorithmType.BlockCholesky)]
    public void CholeskyDecomposition_AllAlgorithms_ProduceValidDecomposition(CholeskyAlgorithmType algorithm)
    {
        // Arrange
        var A = CreateSpdMatrix(5, seed: 123);

        // Act
        var chol = new CholeskyDecomposition<double>(A, algorithm);

        // Assert
        var LLt = chol.L.Multiply(chol.L.Transpose());
        double maxDiff = MaxAbsDiff(A, LLt);

        Assert.True(maxDiff < LooseTolerance,
            $"Algorithm {algorithm}: Max difference = {maxDiff}");
    }

    #endregion

    #region Linear System Solving Tests

    [Theory]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(10)]
    public void CholeskyDecomposition_Solve_ProducesCorrectSolution(int size)
    {
        // Arrange
        var A = CreateSpdMatrix(size, seed: 42);
        var xExpected = new Vector<double>(size);
        for (int i = 0; i < size; i++)
            xExpected[i] = i + 1.0;

        var b = A.Multiply(xExpected);

        // Act
        var chol = new CholeskyDecomposition<double>(A);
        var xComputed = chol.Solve(b);

        // Assert
        var bComputed = A.Multiply(xComputed);
        for (int i = 0; i < size; i++)
        {
            Assert.True(Math.Abs(bComputed[i] - b[i]) < LooseTolerance,
                $"A*x should equal b. Component {i}: expected {b[i]}, got {bComputed[i]}");
        }
    }

    [Fact]
    public void CholeskyDecomposition_Solve_IdentityMatrix_ReturnsSameVector()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(4);
        var b = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

        // Act
        var chol = new CholeskyDecomposition<double>(I);
        var x = chol.Solve(b);

        // Assert
        for (int i = 0; i < 4; i++)
        {
            Assert.True(Math.Abs(x[i] - b[i]) < Tolerance,
                $"Solution for identity matrix should be the input vector. Index {i}");
        }
    }

    [Fact]
    public void CholeskyDecomposition_Solve_DiagonalMatrix_CorrectDivision()
    {
        // Arrange - Diagonal SPD matrix with positive diagonal values
        var D = new Matrix<double>(4, 4);
        D[0, 0] = 1; D[1, 1] = 4; D[2, 2] = 9; D[3, 3] = 16;

        var b = new Vector<double>(new[] { 1.0, 8.0, 27.0, 64.0 });
        // Expected solution: x[i] = b[i] / D[i,i] = {1, 2, 3, 4}

        // Act
        var chol = new CholeskyDecomposition<double>(D);
        var x = chol.Solve(b);

        // Assert
        var expected = new[] { 1.0, 2.0, 3.0, 4.0 };
        for (int i = 0; i < 4; i++)
        {
            Assert.True(Math.Abs(x[i] - expected[i]) < Tolerance,
                $"Index {i}: expected {expected[i]}, got {x[i]}");
        }
    }

    #endregion

    #region Known Matrix Tests

    [Fact]
    public void CholeskyDecomposition_KnownMatrix_CorrectL()
    {
        // Arrange - Use a known SPD matrix with calculable Cholesky factor
        var A = CreateKnownSpdMatrix();

        // Act
        var chol = new CholeskyDecomposition<double>(A);

        // Assert - Verify L * L^T = A
        var LLt = chol.L.Multiply(chol.L.Transpose());
        double maxDiff = MaxAbsDiff(A, LLt);

        Assert.True(maxDiff < Tolerance,
            $"Known matrix decomposition. Max difference: {maxDiff}");

        // Verify L is lower triangular
        Assert.True(IsLowerTriangular(chol.L), "L should be lower triangular");
    }

    [Fact]
    public void CholeskyDecomposition_IdentityMatrix_L_Is_Identity()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(4);

        // Act
        var chol = new CholeskyDecomposition<double>(I);

        // Assert - L should be identity
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.True(Math.Abs(chol.L[i, j] - expected) < Tolerance,
                    $"L[{i},{j}] should be {expected}, got {chol.L[i, j]}");
            }
        }
    }

    #endregion

    #region Error Handling Tests

    [Fact]
    public void CholeskyDecomposition_NonSquareMatrix_ThrowsArgumentException()
    {
        // Arrange
        var A = new Matrix<double>(3, 4);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new CholeskyDecomposition<double>(A));
    }

    [Fact]
    public void CholeskyDecomposition_NonSymmetricMatrix_ThrowsArgumentException()
    {
        // Arrange - Create a non-symmetric matrix
        var A = new Matrix<double>(3, 3);
        A[0, 0] = 4; A[0, 1] = 2; A[0, 2] = 1;
        A[1, 0] = 3; A[1, 1] = 5; A[1, 2] = 2; // A[1,0] != A[0,1]
        A[2, 0] = 1; A[2, 1] = 2; A[2, 2] = 6;

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new CholeskyDecomposition<double>(A));
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void CholeskyDecomposition_WellConditionedMatrix_HighAccuracy()
    {
        // Arrange - Create a well-conditioned SPD matrix
        var A = CreateSpdMatrix(10, seed: 42);

        // Act
        var chol = new CholeskyDecomposition<double>(A);

        // Assert
        var LLt = chol.L.Multiply(chol.L.Transpose());
        double maxDiff = MaxAbsDiff(A, LLt);

        Assert.True(maxDiff < 1e-12,
            $"Well-conditioned matrix should have very high accuracy. Max difference: {maxDiff}");
    }

    [Fact]
    public void CholeskyDecomposition_CovarianceMatrix_ValidDecomposition()
    {
        // Arrange - Simulate a covariance matrix (common use case)
        int n = 5;
        var random = new Random(42);
        var data = new double[100, n];

        // Generate random data
        for (int i = 0; i < 100; i++)
            for (int j = 0; j < n; j++)
                data[i, j] = random.NextDouble() * 10;

        // Compute sample covariance matrix
        var cov = new Matrix<double>(n, n);
        var means = new double[n];

        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < 100; i++)
                means[j] += data[i, j];
            means[j] /= 100;
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int k = 0; k < 100; k++)
                    cov[i, j] += (data[k, i] - means[i]) * (data[k, j] - means[j]);
                cov[i, j] /= 99; // Sample covariance
            }
        }

        // Make sure it's positive definite by adding small diagonal
        for (int i = 0; i < n; i++)
            cov[i, i] += 0.01;

        // Act
        var chol = new CholeskyDecomposition<double>(cov);

        // Assert
        var LLt = chol.L.Multiply(chol.L.Transpose());
        double maxDiff = MaxAbsDiff(cov, LLt);

        Assert.True(maxDiff < LooseTolerance,
            $"Covariance matrix decomposition. Max difference: {maxDiff}");
    }

    #endregion

    #region Uniqueness Tests

    [Fact]
    public void CholeskyDecomposition_IsUnique()
    {
        // Arrange
        var A = CreateSpdMatrix(5, seed: 123);

        // Act - Perform decomposition twice
        var chol1 = new CholeskyDecomposition<double>(A);
        var chol2 = new CholeskyDecomposition<double>(A);

        // Assert - Both should produce the same L
        double maxDiff = MaxAbsDiff(chol1.L, chol2.L);

        Assert.True(maxDiff < Tolerance,
            $"Cholesky decomposition should be unique. Max difference: {maxDiff}");
    }

    #endregion
}
