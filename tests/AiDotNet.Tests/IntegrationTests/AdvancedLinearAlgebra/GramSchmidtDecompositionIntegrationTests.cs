using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Enums.AlgorithmTypes;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AdvancedLinearAlgebra;

/// <summary>
/// Integration tests for Gram-Schmidt decomposition that verify mathematical correctness.
/// These tests verify: A = Q*R, Q has orthonormal columns, R is upper triangular.
/// </summary>
public class GramSchmidtDecompositionIntegrationTests
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

    #endregion

    #region Classical Gram-Schmidt Tests

    [Theory]
    [InlineData(3, 3)]
    [InlineData(4, 4)]
    [InlineData(5, 3)]
    [InlineData(6, 4)]
    public void GramSchmidt_Classical_A_Equals_QR(int rows, int cols)
    {
        // Arrange
        var A = CreateTestMatrix(rows, cols);

        // Act
        var gs = new GramSchmidtDecomposition<double>(A, GramSchmidtAlgorithmType.Classical);

        // Assert - Verify A = Q*R
        var QR = gs.Q.Multiply(gs.R);
        double maxDiff = MaxAbsDiff(A, QR);

        Assert.True(maxDiff < LooseTolerance,
            $"A should equal Q*R. Max difference: {maxDiff}");
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void GramSchmidt_Classical_Q_HasOrthonormalColumns(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size, size);

        // Act
        var gs = new GramSchmidtDecomposition<double>(A, GramSchmidtAlgorithmType.Classical);

        // Assert - Q^T * Q should be identity
        var QtQ = gs.Q.Transpose().Multiply(gs.Q);
        for (int i = 0; i < QtQ.Rows; i++)
        {
            for (int j = 0; j < QtQ.Columns; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.True(Math.Abs(QtQ[i, j] - expected) < LooseTolerance,
                    $"Q^T*Q[{i},{j}] = {QtQ[i, j]}, expected {expected}");
            }
        }
    }

    [Theory]
    [InlineData(3, 3)]
    [InlineData(4, 4)]
    [InlineData(5, 3)]
    public void GramSchmidt_Classical_R_IsUpperTriangular(int rows, int cols)
    {
        // Arrange
        var A = CreateTestMatrix(rows, cols);

        // Act
        var gs = new GramSchmidtDecomposition<double>(A, GramSchmidtAlgorithmType.Classical);

        // Assert
        Assert.True(IsUpperTriangular(gs.R),
            "R should be upper triangular");
    }

    #endregion

    #region Modified Gram-Schmidt Tests

    [Theory]
    [InlineData(3, 3)]
    [InlineData(4, 4)]
    [InlineData(5, 3)]
    public void GramSchmidt_Modified_A_Equals_QR(int rows, int cols)
    {
        // Arrange
        var A = CreateTestMatrix(rows, cols);

        // Act
        var gs = new GramSchmidtDecomposition<double>(A, GramSchmidtAlgorithmType.Modified);

        // Assert - Verify A = Q*R
        var QR = gs.Q.Multiply(gs.R);
        double maxDiff = MaxAbsDiff(A, QR);

        Assert.True(maxDiff < LooseTolerance,
            $"A should equal Q*R. Max difference: {maxDiff}");
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void GramSchmidt_Modified_Q_HasOrthonormalColumns(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size, size);

        // Act
        var gs = new GramSchmidtDecomposition<double>(A, GramSchmidtAlgorithmType.Modified);

        // Assert - Q^T * Q should be identity
        var QtQ = gs.Q.Transpose().Multiply(gs.Q);
        for (int i = 0; i < QtQ.Rows; i++)
        {
            for (int j = 0; j < QtQ.Columns; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.True(Math.Abs(QtQ[i, j] - expected) < LooseTolerance,
                    $"Modified: Q^T*Q[{i},{j}] = {QtQ[i, j]}, expected {expected}");
            }
        }
    }

    [Theory]
    [InlineData(GramSchmidtAlgorithmType.Classical)]
    [InlineData(GramSchmidtAlgorithmType.Modified)]
    public void GramSchmidt_AllAlgorithms_ProduceValidDecomposition(GramSchmidtAlgorithmType algorithm)
    {
        // Arrange
        var A = CreateTestMatrix(4, 4, seed: 123);

        // Act
        var gs = new GramSchmidtDecomposition<double>(A, algorithm);

        // Assert - A should equal Q*R
        var QR = gs.Q.Multiply(gs.R);
        double maxDiff = MaxAbsDiff(A, QR);

        Assert.True(maxDiff < LooseTolerance,
            $"Algorithm {algorithm}: Max difference = {maxDiff}");
    }

    #endregion

    #region Linear System Solving Tests

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void GramSchmidt_Solve_ProducesCorrectSolution(int size)
    {
        // Arrange - Create a system with known solution
        var A = CreateTestMatrix(size, size, seed: 42);
        var xExpected = new Vector<double>(size);
        for (int i = 0; i < size; i++)
            xExpected[i] = i + 1.0;

        var b = A.Multiply(xExpected);

        // Act
        var gs = new GramSchmidtDecomposition<double>(A, GramSchmidtAlgorithmType.Modified);
        var xComputed = gs.Solve(b);

        // Assert - Verify A*x_computed ≈ b
        var bComputed = A.Multiply(xComputed);
        for (int i = 0; i < size; i++)
        {
            Assert.True(Math.Abs(bComputed[i] - b[i]) < LooseTolerance,
                $"A*x should equal b. Component {i}: expected {b[i]}, got {bComputed[i]}");
        }
    }

    [Fact]
    public void GramSchmidt_Solve_IdentityMatrix_ReturnsSameVector()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(4);
        var b = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

        // Act
        var gs = new GramSchmidtDecomposition<double>(I);
        var x = gs.Solve(b);

        // Assert
        for (int i = 0; i < 4; i++)
        {
            Assert.True(Math.Abs(x[i] - b[i]) < LooseTolerance,
                $"Solution for identity matrix should be the input vector. Index {i}");
        }
    }

    #endregion

    #region Matrix Inversion Tests

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void GramSchmidt_Invert_ProducesCorrectInverse(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size, size, seed: 42);

        // Act
        var gs = new GramSchmidtDecomposition<double>(A, GramSchmidtAlgorithmType.Modified);
        var AInv = gs.Invert();

        // Assert - A * A^-1 should equal I
        var product = A.Multiply(AInv);
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.True(Math.Abs(product[i, j] - expected) < LooseTolerance,
                    $"(A * A^-1)[{i},{j}] = {product[i, j]}, expected {expected}");
            }
        }
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void GramSchmidt_2x2Matrix_CorrectDecomposition()
    {
        // Arrange
        var A = new Matrix<double>(2, 2);
        A[0, 0] = 3; A[0, 1] = 1;
        A[1, 0] = 4; A[1, 1] = 2;

        // Act
        var gs = new GramSchmidtDecomposition<double>(A);

        // Assert
        var QR = gs.Q.Multiply(gs.R);
        double maxDiff = MaxAbsDiff(A, QR);

        Assert.True(maxDiff < LooseTolerance,
            $"2x2 decomposition failed. Max difference: {maxDiff}");
    }

    [Fact]
    public void GramSchmidt_TallMatrix_ValidDecomposition()
    {
        // Arrange - More rows than columns
        var A = CreateTestMatrix(6, 3);

        // Act
        var gs = new GramSchmidtDecomposition<double>(A);

        // Assert
        var QR = gs.Q.Multiply(gs.R);
        double maxDiff = MaxAbsDiff(A, QR);

        Assert.True(maxDiff < LooseTolerance,
            $"Tall matrix decomposition failed. Max difference: {maxDiff}");
    }

    [Fact]
    public void GramSchmidt_IdentityMatrix_DecomposesToIdentities()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(4);

        // Act
        var gs = new GramSchmidtDecomposition<double>(I);

        // Assert - Q and R should both be identity (up to sign)
        for (int i = 0; i < 4; i++)
        {
            Assert.True(Math.Abs(Math.Abs(gs.Q[i, i]) - 1.0) < LooseTolerance,
                $"Diagonal of Q should be +/-1. Q[{i},{i}] = {gs.Q[i, i]}");
            Assert.True(Math.Abs(Math.Abs(gs.R[i, i]) - 1.0) < LooseTolerance,
                $"Diagonal of R should be +/-1. R[{i},{i}] = {gs.R[i, i]}");
        }
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void GramSchmidt_Modified_HandlesIllConditionedMatrix()
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
        var gsModified = new GramSchmidtDecomposition<double>(A, GramSchmidtAlgorithmType.Modified);

        // Assert - Modified should produce accurate decomposition
        var QR = gsModified.Q.Multiply(gsModified.R);
        double maxDiff = MaxAbsDiff(A, QR);

        Assert.True(maxDiff < LooseTolerance,
            $"Modified Gram-Schmidt on ill-conditioned matrix. Max difference: {maxDiff}");
    }

    [Fact]
    public void GramSchmidt_Q_Columns_AreUnitVectors()
    {
        // Arrange
        var A = CreateTestMatrix(5, 5);

        // Act
        var gs = new GramSchmidtDecomposition<double>(A, GramSchmidtAlgorithmType.Modified);

        // Assert - Each column of Q should have unit norm
        for (int j = 0; j < gs.Q.Columns; j++)
        {
            double norm = 0;
            for (int i = 0; i < gs.Q.Rows; i++)
            {
                norm += gs.Q[i, j] * gs.Q[i, j];
            }
            norm = Math.Sqrt(norm);

            Assert.True(Math.Abs(norm - 1.0) < LooseTolerance,
                $"Column {j} of Q should have unit norm. Actual: {norm}");
        }
    }

    [Fact]
    public void GramSchmidt_Q_Columns_AreMutuallyOrthogonal()
    {
        // Arrange
        var A = CreateTestMatrix(5, 5);

        // Act
        var gs = new GramSchmidtDecomposition<double>(A, GramSchmidtAlgorithmType.Modified);

        // Assert - Different columns should be orthogonal (dot product ≈ 0)
        for (int i = 0; i < gs.Q.Columns; i++)
        {
            for (int j = i + 1; j < gs.Q.Columns; j++)
            {
                double dotProduct = 0;
                for (int k = 0; k < gs.Q.Rows; k++)
                {
                    dotProduct += gs.Q[k, i] * gs.Q[k, j];
                }

                Assert.True(Math.Abs(dotProduct) < LooseTolerance,
                    $"Columns {i} and {j} should be orthogonal. Dot product: {dotProduct}");
            }
        }
    }

    #endregion
}
