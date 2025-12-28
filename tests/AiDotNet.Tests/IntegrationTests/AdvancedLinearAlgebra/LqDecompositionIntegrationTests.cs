using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Enums.AlgorithmTypes;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AdvancedLinearAlgebra;

/// <summary>
/// Integration tests for LQ decomposition that verify mathematical correctness.
/// These tests verify: A = L*Q, L is lower triangular, Q is orthogonal.
/// </summary>
public class LqDecompositionIntegrationTests
{
    private const double Tolerance = 1e-8;
    private const double LooseTolerance = 1e-4;

    #region Helper Methods

    private static Matrix<double> CreateTestMatrix(int rows, int cols, int seed = 42)
    {
        var random = new Random(seed);
        var matrix = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = random.NextDouble() * 10 - 5;
            }
        }
        return matrix;
    }

    private static double MaxAbsDiff(Matrix<double> a, Matrix<double> b)
    {
        if (a.Rows != b.Rows || a.Columns != b.Columns)
            return double.MaxValue;

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

    private static bool IsOrthogonal(Matrix<double> Q, double tolerance)
    {
        // For Q to be orthogonal, Q * Q^T should be identity
        var QQt = Q.Multiply(Q.Transpose());
        int n = QQt.Rows;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                if (Math.Abs(QQt[i, j] - expected) > tolerance)
                    return false;
            }
        }
        return true;
    }

    private static bool IsLowerTriangular(Matrix<double> L, double tolerance)
    {
        // Lower triangular has zeros above the diagonal
        int minDim = Math.Min(L.Rows, L.Columns);
        for (int i = 0; i < minDim; i++)
        {
            for (int j = i + 1; j < L.Columns; j++)
            {
                if (Math.Abs(L[i, j]) > tolerance)
                    return false;
            }
        }
        return true;
    }

    #endregion

    #region Reconstruction Tests (A = L*Q)

    [Theory]
    [InlineData(3, 3)]
    [InlineData(4, 4)]
    [InlineData(3, 5)]
    [InlineData(5, 3)]
    public void LqDecomposition_Householder_Reconstruction(int rows, int cols)
    {
        // Arrange
        var A = CreateTestMatrix(rows, cols);

        // Act
        var lq = new LqDecomposition<double>(A, LqAlgorithmType.Householder);

        // Assert - Verify A = L*Q
        var reconstructed = lq.L.Multiply(lq.Q);
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"A should equal L*Q. Max difference: {maxDiff}");
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void LqDecomposition_Givens_Reconstruction(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size, size, seed: 123);

        // Act
        var lq = new LqDecomposition<double>(A, LqAlgorithmType.Givens);

        // Assert
        var reconstructed = lq.L.Multiply(lq.Q);
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"Givens: A should equal L*Q. Max difference: {maxDiff}");
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void LqDecomposition_GramSchmidt_Reconstruction(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size, size, seed: 456);

        // Act
        var lq = new LqDecomposition<double>(A, LqAlgorithmType.GramSchmidt);

        // Assert
        var reconstructed = lq.L.Multiply(lq.Q);
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"GramSchmidt: A should equal L*Q. Max difference: {maxDiff}");
    }

    #endregion

    #region L Matrix Structure Tests

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void LqDecomposition_L_IsLowerTriangular(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size, size);

        // Act
        var lq = new LqDecomposition<double>(A);

        // Assert - L should be lower triangular
        Assert.True(IsLowerTriangular(lq.L, LooseTolerance),
            "L should be lower triangular (zeros above diagonal)");
    }

    #endregion

    #region Orthogonality Tests

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void LqDecomposition_Q_IsOrthogonal(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size, size);

        // Act
        var lq = new LqDecomposition<double>(A);

        // Assert - Q * Q^T should be identity
        Assert.True(IsOrthogonal(lq.Q, LooseTolerance),
            "Q should be orthogonal (Q * Q^T = I)");
    }

    #endregion

    #region Special Matrix Tests

    [Fact]
    public void LqDecomposition_IdentityMatrix_ValidDecomposition()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(4);

        // Act
        var lq = new LqDecomposition<double>(I);

        // Assert - L should be lower triangular
        Assert.True(IsLowerTriangular(lq.L, Tolerance),
            "Identity matrix L should be lower triangular");

        // Verify reconstruction
        var reconstructed = lq.L.Multiply(lq.Q);
        double maxDiff = MaxAbsDiff(I, reconstructed);
        Assert.True(maxDiff < LooseTolerance,
            $"Identity matrix reconstruction failed. Max diff: {maxDiff}");
    }

    [Fact]
    public void LqDecomposition_DiagonalMatrix_ValidDecomposition()
    {
        // Arrange
        var D = new Matrix<double>(4, 4);
        D[0, 0] = 4; D[1, 1] = 3; D[2, 2] = 2; D[3, 3] = 1;

        // Act
        var lq = new LqDecomposition<double>(D);

        // Assert
        Assert.True(IsLowerTriangular(lq.L, Tolerance),
            "Diagonal matrix L should be lower triangular");
    }

    [Fact]
    public void LqDecomposition_LowerTriangularMatrix_PreservesStructure()
    {
        // Arrange - Lower triangular matrix
        var L_input = new Matrix<double>(4, 4);
        L_input[0, 0] = 4;
        L_input[1, 0] = 2; L_input[1, 1] = 3;
        L_input[2, 0] = 1; L_input[2, 1] = 2; L_input[2, 2] = 2;
        L_input[3, 0] = 0; L_input[3, 1] = 1; L_input[3, 2] = 1; L_input[3, 3] = 1;

        // Act
        var lq = new LqDecomposition<double>(L_input);

        // Assert
        Assert.True(IsLowerTriangular(lq.L, LooseTolerance),
            "Lower triangular input L should produce lower triangular L");
    }

    [Fact]
    public void LqDecomposition_ZeroMatrix_ValidDecomposition()
    {
        // Arrange
        var Z = new Matrix<double>(3, 3); // All zeros

        // Act
        var lq = new LqDecomposition<double>(Z);

        // Assert - L should be lower triangular (all zeros is valid)
        Assert.True(IsLowerTriangular(lq.L, Tolerance),
            "Zero matrix L should be lower triangular");
    }

    #endregion

    #region Algorithm Comparison Tests

    [Theory]
    [InlineData(LqAlgorithmType.Householder)]
    [InlineData(LqAlgorithmType.Givens)]
    [InlineData(LqAlgorithmType.GramSchmidt)]
    public void LqDecomposition_AllAlgorithms_ProduceValidDecomposition(LqAlgorithmType algorithm)
    {
        // Arrange
        var A = CreateTestMatrix(4, 4, seed: 42);

        // Act
        var lq = new LqDecomposition<double>(A, algorithm);

        // Assert - Basic validity checks
        Assert.Equal(4, lq.L.Rows);
        Assert.Equal(4, lq.Q.Rows);

        // No NaN or Inf values
        for (int i = 0; i < lq.L.Rows; i++)
        {
            for (int j = 0; j < lq.L.Columns; j++)
            {
                Assert.False(double.IsNaN(lq.L[i, j]),
                    $"Algorithm {algorithm}: L has NaN at [{i},{j}]");
            }
            for (int j = 0; j < lq.Q.Columns; j++)
            {
                Assert.False(double.IsNaN(lq.Q[i, j]),
                    $"Algorithm {algorithm}: Q has NaN at [{i},{j}]");
            }
        }
    }

    #endregion

    #region Rectangular Matrix Tests

    [Fact]
    public void LqDecomposition_TallMatrix_ValidDecomposition()
    {
        // Arrange - More rows than columns
        var A = CreateTestMatrix(6, 3);

        // Act
        var lq = new LqDecomposition<double>(A);

        // Assert
        var reconstructed = lq.L.Multiply(lq.Q);
        double maxDiff = MaxAbsDiff(A, reconstructed);
        Assert.True(maxDiff < LooseTolerance,
            $"Tall matrix reconstruction failed. Max diff: {maxDiff}");
    }

    [Fact]
    public void LqDecomposition_WideMatrix_ValidDecomposition()
    {
        // Arrange - More columns than rows
        var A = CreateTestMatrix(3, 6);

        // Act
        var lq = new LqDecomposition<double>(A);

        // Assert
        var reconstructed = lq.L.Multiply(lq.Q);
        double maxDiff = MaxAbsDiff(A, reconstructed);
        Assert.True(maxDiff < LooseTolerance,
            $"Wide matrix reconstruction failed. Max diff: {maxDiff}");
    }

    #endregion

    #region Solve Tests

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void LqDecomposition_Solve_ProducesCorrectSolution(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size, size, seed: 42);
        var xExpected = new Vector<double>(size);
        for (int i = 0; i < size; i++)
            xExpected[i] = i + 1.0;

        var b = A.Multiply(xExpected);

        // Act
        var lq = new LqDecomposition<double>(A);
        var xComputed = lq.Solve(b);

        // Assert - Verify A*x_computed ≈ b
        var bComputed = A.Multiply(xComputed);
        for (int i = 0; i < size; i++)
        {
            Assert.True(Math.Abs(bComputed[i] - b[i]) < LooseTolerance,
                $"A*x should equal b. Component {i}: expected {b[i]}, got {bComputed[i]}");
        }
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void LqDecomposition_LargeMatrix_ValidDecomposition()
    {
        // Arrange
        var A = CreateTestMatrix(10, 10, seed: 999);

        // Act
        var lq = new LqDecomposition<double>(A);

        // Assert
        Assert.True(IsLowerTriangular(lq.L, LooseTolerance),
            "Large matrix L should be lower triangular");

        var reconstructed = lq.L.Multiply(lq.Q);
        double maxDiff = MaxAbsDiff(A, reconstructed);
        Assert.True(maxDiff < LooseTolerance,
            $"Large matrix reconstruction failed. Max diff: {maxDiff}");
    }

    #endregion
}
