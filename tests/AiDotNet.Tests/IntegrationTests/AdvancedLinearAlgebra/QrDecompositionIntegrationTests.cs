using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Enums.AlgorithmTypes;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AdvancedLinearAlgebra;

/// <summary>
/// Integration tests for QR decomposition that verify mathematical correctness.
/// These tests verify: A = Q*R, Q is orthogonal (Q^T*Q = I), R is upper triangular.
/// </summary>
public class QrDecompositionIntegrationTests
{
    private const double Tolerance = 1e-10;
    private const double LooseTolerance = 1e-6;

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

    private static bool IsUpperTriangular(Matrix<double> matrix)
    {
        for (int i = 1; i < matrix.Rows; i++)
        {
            for (int j = 0; j < Math.Min(i, matrix.Columns); j++)
            {
                if (Math.Abs(matrix[i, j]) > LooseTolerance)
                    return false;
            }
        }
        return true;
    }

    private static bool IsOrthogonal(Matrix<double> Q, double tolerance)
    {
        // Check Q^T * Q = I
        var QtQ = Q.Transpose().Multiply(Q);
        int n = QtQ.Rows;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                if (Math.Abs(QtQ[i, j] - expected) > tolerance)
                    return false;
            }
        }
        return true;
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

    private static double FrobeniusNorm(Matrix<double> m)
    {
        double sum = 0;
        for (int i = 0; i < m.Rows; i++)
        {
            for (int j = 0; j < m.Columns; j++)
            {
                sum += m[i, j] * m[i, j];
            }
        }
        return Math.Sqrt(sum);
    }

    #endregion

    #region Decomposition Property Tests

    [Theory]
    [InlineData(3, 3)]
    [InlineData(4, 4)]
    [InlineData(5, 3)]
    [InlineData(6, 4)]
    public void QrDecomposition_Householder_A_Equals_QR(int rows, int cols)
    {
        // Arrange
        var A = CreateTestMatrix(rows, cols);

        // Act
        var qr = new QrDecomposition<double>(A, QrAlgorithmType.Householder);

        // Assert - Verify A = Q*R
        var QR = qr.Q.Multiply(qr.R);
        double maxDiff = MaxAbsDiff(A, QR);

        Assert.True(maxDiff < LooseTolerance,
            $"A should equal Q*R. Max difference: {maxDiff}");
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void QrDecomposition_Householder_Q_IsOrthogonal(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size, size);

        // Act
        var qr = new QrDecomposition<double>(A, QrAlgorithmType.Householder);

        // Assert - Q^T * Q should be identity
        Assert.True(IsOrthogonal(qr.Q, LooseTolerance),
            "Q should be orthogonal (Q^T * Q = I)");
    }

    [Theory]
    [InlineData(3, 3)]
    [InlineData(4, 4)]
    [InlineData(5, 3)]
    public void QrDecomposition_Householder_R_IsUpperTriangular(int rows, int cols)
    {
        // Arrange
        var A = CreateTestMatrix(rows, cols);

        // Act
        var qr = new QrDecomposition<double>(A, QrAlgorithmType.Householder);

        // Assert
        Assert.True(IsUpperTriangular(qr.R),
            "R should be upper triangular");
    }

    [Theory]
    [InlineData(QrAlgorithmType.GramSchmidt)]
    [InlineData(QrAlgorithmType.Householder)]
    [InlineData(QrAlgorithmType.Givens)]
    [InlineData(QrAlgorithmType.ModifiedGramSchmidt)]
    [InlineData(QrAlgorithmType.IterativeGramSchmidt)]
    public void QrDecomposition_AllAlgorithms_ProduceValidDecomposition(QrAlgorithmType algorithm)
    {
        // Arrange
        var A = CreateTestMatrix(4, 4, seed: 123);

        // Act
        var qr = new QrDecomposition<double>(A, algorithm);

        // Assert - A should equal Q*R
        var QR = qr.Q.Multiply(qr.R);
        double maxDiff = MaxAbsDiff(A, QR);

        Assert.True(maxDiff < LooseTolerance,
            $"Algorithm {algorithm}: Max difference = {maxDiff}");
    }

    [Theory]
    [InlineData(QrAlgorithmType.GramSchmidt)]
    [InlineData(QrAlgorithmType.ModifiedGramSchmidt)]
    [InlineData(QrAlgorithmType.IterativeGramSchmidt)]
    public void QrDecomposition_GramSchmidtVariants_Q_HasOrthonormalColumns(QrAlgorithmType algorithm)
    {
        // Arrange
        var A = CreateTestMatrix(5, 3);

        // Act
        var qr = new QrDecomposition<double>(A, algorithm);

        // Assert - Check that columns of Q are orthonormal
        // For tall matrices, Q^T * Q should be identity (for thin Q)
        var QtQ = qr.Q.Transpose().Multiply(qr.Q);
        for (int i = 0; i < QtQ.Rows; i++)
        {
            for (int j = 0; j < QtQ.Columns; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.True(Math.Abs(QtQ[i, j] - expected) < LooseTolerance,
                    $"Algorithm {algorithm}: Q^T*Q[{i},{j}] = {QtQ[i, j]}, expected {expected}");
            }
        }
    }

    #endregion

    #region Linear System Solving Tests

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void QrDecomposition_Solve_ProducesCorrectSolution(int size)
    {
        // Arrange - Create a system with known solution
        var A = CreateTestMatrix(size, size, seed: 42);
        var xExpected = new Vector<double>(size);
        for (int i = 0; i < size; i++)
            xExpected[i] = i + 1.0;

        var b = A.Multiply(xExpected);

        // Act
        var qr = new QrDecomposition<double>(A, QrAlgorithmType.Householder);
        var xComputed = qr.Solve(b);

        // Assert - Verify A*x_computed ≈ b
        var bComputed = A.Multiply(xComputed);
        for (int i = 0; i < size; i++)
        {
            Assert.True(Math.Abs(bComputed[i] - b[i]) < LooseTolerance,
                $"A*x should equal b. Component {i}: expected {b[i]}, got {bComputed[i]}");
        }
    }

    [Fact]
    public void QrDecomposition_Solve_IdentityMatrix_ReturnsSameVector()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(4);
        var b = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

        // Act
        var qr = new QrDecomposition<double>(I);
        var x = qr.Solve(b);

        // Assert
        for (int i = 0; i < 4; i++)
        {
            Assert.True(Math.Abs(x[i] - b[i]) < LooseTolerance,
                $"Solution for identity matrix should be the input vector. Index {i}");
        }
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void QrDecomposition_2x2Matrix_CorrectDecomposition()
    {
        // Arrange
        var A = new Matrix<double>(2, 2);
        A[0, 0] = 3; A[0, 1] = 1;
        A[1, 0] = 4; A[1, 1] = 2;

        // Act
        var qr = new QrDecomposition<double>(A);

        // Assert
        var QR = qr.Q.Multiply(qr.R);
        double maxDiff = MaxAbsDiff(A, QR);

        Assert.True(maxDiff < LooseTolerance,
            $"2x2 decomposition failed. Max difference: {maxDiff}");
    }

    [Fact]
    public void QrDecomposition_TallMatrix_ValidDecomposition()
    {
        // Arrange - More rows than columns
        var A = CreateTestMatrix(6, 3);

        // Act
        var qr = new QrDecomposition<double>(A);

        // Assert
        var QR = qr.Q.Multiply(qr.R);
        double maxDiff = MaxAbsDiff(A, QR);

        Assert.True(maxDiff < LooseTolerance,
            $"Tall matrix decomposition failed. Max difference: {maxDiff}");
    }

    [Fact]
    public void QrDecomposition_IdentityMatrix_Q_Is_Identity()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(4);

        // Act
        var qr = new QrDecomposition<double>(I);

        // Assert - Q should be identity (up to sign)
        // R should also be identity
        for (int i = 0; i < 4; i++)
        {
            Assert.True(Math.Abs(Math.Abs(qr.Q[i, i]) - 1.0) < LooseTolerance,
                $"Diagonal of Q should be ±1. Q[{i},{i}] = {qr.Q[i, i]}");
            Assert.True(Math.Abs(Math.Abs(qr.R[i, i]) - 1.0) < LooseTolerance,
                $"Diagonal of R should be ±1. R[{i},{i}] = {qr.R[i, i]}");
        }
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void QrDecomposition_Householder_AccurateOnIllConditionedMatrix()
    {
        // Arrange - Create a Hilbert-like matrix that can cause numerical issues
        int n = 5;
        var A = new Matrix<double>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                A[i, j] = 1.0 / (i + j + 1); // Hilbert-like matrix
            }
        }

        // Act
        var qrHouseholder = new QrDecomposition<double>(A, QrAlgorithmType.Householder);

        // Assert - Householder should produce accurate decomposition
        var QR = qrHouseholder.Q.Multiply(qrHouseholder.R);
        double maxDiff = MaxAbsDiff(A, QR);

        Assert.True(maxDiff < LooseTolerance,
            $"Householder on ill-conditioned matrix. Max difference: {maxDiff}");

        // Verify orthogonality is preserved
        Assert.True(IsOrthogonal(qrHouseholder.Q, LooseTolerance),
            "Q should remain orthogonal even for ill-conditioned matrices");
    }

    [Fact]
    public void QrDecomposition_LargeMatrix_CorrectDecomposition()
    {
        // Arrange
        var A = CreateTestMatrix(30, 30, seed: 999);

        // Act
        var qr = new QrDecomposition<double>(A, QrAlgorithmType.Householder);

        // Assert
        var QR = qr.Q.Multiply(qr.R);
        double relativeError = MaxAbsDiff(A, QR) / FrobeniusNorm(A);

        Assert.True(relativeError < 1e-10,
            $"Large matrix: Relative error = {relativeError}");
    }

    #endregion

    #region Orthogonality Preservation Tests

    [Fact]
    public void QrDecomposition_Q_Columns_AreUnitVectors()
    {
        // Arrange
        var A = CreateTestMatrix(5, 5);

        // Act
        var qr = new QrDecomposition<double>(A, QrAlgorithmType.Householder);

        // Assert - Each column of Q should have unit norm
        for (int j = 0; j < qr.Q.Columns; j++)
        {
            double norm = 0;
            for (int i = 0; i < qr.Q.Rows; i++)
            {
                norm += qr.Q[i, j] * qr.Q[i, j];
            }
            norm = Math.Sqrt(norm);

            Assert.True(Math.Abs(norm - 1.0) < LooseTolerance,
                $"Column {j} of Q should have unit norm. Actual: {norm}");
        }
    }

    [Fact]
    public void QrDecomposition_Q_Columns_AreMutuallyOrthogonal()
    {
        // Arrange
        var A = CreateTestMatrix(5, 5);

        // Act
        var qr = new QrDecomposition<double>(A, QrAlgorithmType.Householder);

        // Assert - Different columns should be orthogonal (dot product ≈ 0)
        for (int i = 0; i < qr.Q.Columns; i++)
        {
            for (int j = i + 1; j < qr.Q.Columns; j++)
            {
                double dotProduct = 0;
                for (int k = 0; k < qr.Q.Rows; k++)
                {
                    dotProduct += qr.Q[k, i] * qr.Q[k, j];
                }

                Assert.True(Math.Abs(dotProduct) < LooseTolerance,
                    $"Columns {i} and {j} should be orthogonal. Dot product: {dotProduct}");
            }
        }
    }

    #endregion
}
