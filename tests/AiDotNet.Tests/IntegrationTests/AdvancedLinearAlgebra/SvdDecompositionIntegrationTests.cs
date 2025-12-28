using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Enums.AlgorithmTypes;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AdvancedLinearAlgebra;

/// <summary>
/// Integration tests for SVD decomposition that verify mathematical correctness.
/// These tests verify: A = U*S*V^T, U and V are orthogonal, singular values are non-negative and sorted.
/// </summary>
public class SvdDecompositionIntegrationTests
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

    private static Matrix<double> ReconstructFromSvd(Matrix<double> U, Vector<double> S, Matrix<double> Vt)
    {
        int m = U.Rows;
        int n = Vt.Columns;
        int k = S.Length;

        var result = new Matrix<double>(m, n);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int l = 0; l < k; l++)
                {
                    sum += U[i, l] * S[l] * Vt[l, j];
                }
                result[i, j] = sum;
            }
        }
        return result;
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

    private static bool AreSingularValuesNonNegative(Vector<double> S)
    {
        for (int i = 0; i < S.Length; i++)
        {
            if (S[i] < -Tolerance) // Allow small numerical errors
                return false;
        }
        return true;
    }

    private static bool AreSingularValuesSorted(Vector<double> S)
    {
        for (int i = 0; i < S.Length - 1; i++)
        {
            if (S[i] < S[i + 1] - Tolerance) // Allow small numerical errors
                return false;
        }
        return true;
    }

    #endregion

    #region Reconstruction Tests (A = U*S*V^T)

    [Theory]
    [InlineData(3, 3)]
    [InlineData(4, 4)]
    [InlineData(5, 3)]
    [InlineData(3, 5)]
    public void SvdDecomposition_GolubReinsch_Reconstruction(int rows, int cols)
    {
        // Arrange
        var A = CreateTestMatrix(rows, cols);

        // Act
        var svd = new SvdDecomposition<double>(A, SvdAlgorithmType.GolubReinsch);

        // Assert - Verify A = U*S*V^T
        var reconstructed = ReconstructFromSvd(svd.U, svd.S, svd.Vt);
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"A should equal U*S*V^T. Max difference: {maxDiff}");
    }

    [Theory]
    [InlineData(3, 3)]
    [InlineData(4, 4)]
    public void SvdDecomposition_Jacobi_Reconstruction(int rows, int cols)
    {
        // Arrange
        var A = CreateTestMatrix(rows, cols, seed: 123);

        // Act
        var svd = new SvdDecomposition<double>(A, SvdAlgorithmType.Jacobi);

        // Assert
        var reconstructed = ReconstructFromSvd(svd.U, svd.S, svd.Vt);
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"Jacobi: A should equal U*S*V^T. Max difference: {maxDiff}");
    }

    [Theory]
    [InlineData(SvdAlgorithmType.GolubReinsch)]
    [InlineData(SvdAlgorithmType.Jacobi)]
    [InlineData(SvdAlgorithmType.PowerIteration)]
    public void SvdDecomposition_AllAlgorithms_ProduceValidDecomposition(SvdAlgorithmType algorithm)
    {
        // Arrange
        var A = CreateTestMatrix(4, 4, seed: 42);

        // Act
        var svd = new SvdDecomposition<double>(A, algorithm);

        // Assert - All singular values should be non-negative
        Assert.True(AreSingularValuesNonNegative(svd.S),
            $"Algorithm {algorithm}: Singular values should be non-negative");

        // Singular values should be sorted in descending order
        Assert.True(AreSingularValuesSorted(svd.S),
            $"Algorithm {algorithm}: Singular values should be sorted in descending order");
    }

    #endregion

    #region Orthogonality Tests

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void SvdDecomposition_U_IsOrthogonal(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size, size);

        // Act
        var svd = new SvdDecomposition<double>(A, SvdAlgorithmType.GolubReinsch);

        // Assert - U^T * U should be identity
        Assert.True(IsOrthogonal(svd.U, LooseTolerance),
            $"U should be orthogonal (U^T * U = I)");
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void SvdDecomposition_Vt_IsOrthogonal(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size, size);

        // Act
        var svd = new SvdDecomposition<double>(A, SvdAlgorithmType.GolubReinsch);

        // Assert - Vt * Vt^T should be identity (since Vt = V^T)
        var VtVtt = svd.Vt.Multiply(svd.Vt.Transpose());
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.True(Math.Abs(VtVtt[i, j] - expected) < LooseTolerance,
                    $"V^T should be orthogonal. Vt*Vt^T[{i},{j}] = {VtVtt[i, j]}, expected {expected}");
            }
        }
    }

    #endregion

    #region Singular Value Property Tests

    [Fact]
    public void SvdDecomposition_SingularValues_AreNonNegative()
    {
        // Arrange
        var A = CreateTestMatrix(5, 5);

        // Act
        var svd = new SvdDecomposition<double>(A);

        // Assert
        for (int i = 0; i < svd.S.Length; i++)
        {
            Assert.True(svd.S[i] >= -Tolerance,
                $"Singular value S[{i}] = {svd.S[i]} should be non-negative");
        }
    }

    [Fact]
    public void SvdDecomposition_SingularValues_AreSortedDescending()
    {
        // Arrange
        var A = CreateTestMatrix(5, 5);

        // Act
        var svd = new SvdDecomposition<double>(A);

        // Assert
        for (int i = 0; i < svd.S.Length - 1; i++)
        {
            Assert.True(svd.S[i] >= svd.S[i + 1] - Tolerance,
                $"Singular values should be sorted. S[{i}] = {svd.S[i]}, S[{i + 1}] = {svd.S[i + 1]}");
        }
    }

    [Fact]
    public void SvdDecomposition_IdentityMatrix_HasAllOnes()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(4);

        // Act
        var svd = new SvdDecomposition<double>(I);

        // Assert - All singular values should be 1
        for (int i = 0; i < svd.S.Length; i++)
        {
            Assert.True(Math.Abs(svd.S[i] - 1.0) < Tolerance,
                $"Identity matrix singular value S[{i}] = {svd.S[i]}, expected 1.0");
        }
    }

    [Fact]
    public void SvdDecomposition_DiagonalMatrix_HasCorrectSingularValues()
    {
        // Arrange - Diagonal matrix with values 4, 3, 2, 1
        var D = new Matrix<double>(4, 4);
        D[0, 0] = 4; D[1, 1] = 3; D[2, 2] = 2; D[3, 3] = 1;

        // Act
        var svd = new SvdDecomposition<double>(D);

        // Assert - Singular values should be 4, 3, 2, 1 (sorted descending)
        var expected = new[] { 4.0, 3.0, 2.0, 1.0 };
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(Math.Abs(svd.S[i] - expected[i]) < Tolerance,
                $"Diagonal matrix S[{i}] = {svd.S[i]}, expected {expected[i]}");
        }
    }

    #endregion

    #region Solve Tests

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void SvdDecomposition_Solve_ProducesCorrectSolution(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size, size, seed: 42);
        var xExpected = new Vector<double>(size);
        for (int i = 0; i < size; i++)
            xExpected[i] = i + 1.0;

        var b = A.Multiply(xExpected);

        // Act
        var svd = new SvdDecomposition<double>(A);
        var xComputed = svd.Solve(b);

        // Assert - Verify A*x_computed ≈ b
        var bComputed = A.Multiply(xComputed);
        for (int i = 0; i < size; i++)
        {
            Assert.True(Math.Abs(bComputed[i] - b[i]) < LooseTolerance,
                $"A*x should equal b. Component {i}: expected {b[i]}, got {bComputed[i]}");
        }
    }

    [Fact]
    public void SvdDecomposition_Solve_IdentityMatrix_ReturnsSameVector()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(4);
        var b = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

        // Act
        var svd = new SvdDecomposition<double>(I);
        var x = svd.Solve(b);

        // Assert
        for (int i = 0; i < 4; i++)
        {
            Assert.True(Math.Abs(x[i] - b[i]) < Tolerance,
                $"Solution for identity matrix should be the input vector. Index {i}: expected {b[i]}, got {x[i]}");
        }
    }

    #endregion

    #region Rectangular Matrix Tests

    [Fact]
    public void SvdDecomposition_TallMatrix_ValidDecomposition()
    {
        // Arrange - More rows than columns
        var A = CreateTestMatrix(6, 3);

        // Act
        var svd = new SvdDecomposition<double>(A);

        // Assert
        Assert.Equal(6, svd.U.Rows);
        Assert.Equal(3, svd.S.Length);
        Assert.Equal(3, svd.Vt.Columns);

        // Verify reconstruction
        var reconstructed = ReconstructFromSvd(svd.U, svd.S, svd.Vt);
        double maxDiff = MaxAbsDiff(A, reconstructed);
        Assert.True(maxDiff < LooseTolerance,
            $"Tall matrix reconstruction failed. Max difference: {maxDiff}");
    }

    [Fact]
    public void SvdDecomposition_WideMatrix_ValidDecomposition()
    {
        // Arrange - More columns than rows
        var A = CreateTestMatrix(3, 6);

        // Act
        var svd = new SvdDecomposition<double>(A);

        // Assert
        Assert.Equal(3, svd.U.Rows);
        Assert.Equal(3, svd.S.Length);
        Assert.Equal(6, svd.Vt.Columns);

        // Verify reconstruction
        var reconstructed = ReconstructFromSvd(svd.U, svd.S, svd.Vt);
        double maxDiff = MaxAbsDiff(A, reconstructed);
        Assert.True(maxDiff < LooseTolerance,
            $"Wide matrix reconstruction failed. Max difference: {maxDiff}");
    }

    #endregion

    #region Special Matrix Tests

    [Fact]
    public void SvdDecomposition_ZeroMatrix_HasZeroSingularValues()
    {
        // Arrange
        var Z = new Matrix<double>(3, 3); // All zeros

        // Act
        var svd = new SvdDecomposition<double>(Z);

        // Assert - All singular values should be 0
        for (int i = 0; i < svd.S.Length; i++)
        {
            Assert.True(Math.Abs(svd.S[i]) < Tolerance,
                $"Zero matrix singular value S[{i}] = {svd.S[i]}, expected 0");
        }
    }

    [Fact]
    public void SvdDecomposition_RankOneMatrix_HasOneSingularValue()
    {
        // Arrange - Rank 1 matrix: u * v^T
        var u = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var v = new Vector<double>(new[] { 4.0, 5.0, 6.0 });
        var A = new Matrix<double>(3, 3);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                A[i, j] = u[i] * v[j];

        // Act
        var svd = new SvdDecomposition<double>(A);

        // Assert - Only one significant singular value
        Assert.True(svd.S[0] > 1.0, "First singular value should be significant");
        Assert.True(Math.Abs(svd.S[1]) < LooseTolerance, $"Second singular value should be ~0, got {svd.S[1]}");
        Assert.True(Math.Abs(svd.S[2]) < LooseTolerance, $"Third singular value should be ~0, got {svd.S[2]}");
    }

    [Fact]
    public void SvdDecomposition_SymmetricMatrix_ValidDecomposition()
    {
        // Arrange - Create symmetric matrix A^T * A
        var B = CreateTestMatrix(4, 4, seed: 123);
        var A = B.Transpose().Multiply(B);

        // Act
        var svd = new SvdDecomposition<double>(A);

        // Assert
        var reconstructed = ReconstructFromSvd(svd.U, svd.S, svd.Vt);
        double maxDiff = MaxAbsDiff(A, reconstructed);
        Assert.True(maxDiff < LooseTolerance,
            $"Symmetric matrix reconstruction failed. Max difference: {maxDiff}");
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void SvdDecomposition_LargeMatrix_CorrectDecomposition()
    {
        // Arrange
        var A = CreateTestMatrix(20, 20, seed: 999);

        // Act
        var svd = new SvdDecomposition<double>(A);

        // Assert
        var reconstructed = ReconstructFromSvd(svd.U, svd.S, svd.Vt);
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"Large matrix reconstruction failed. Max difference: {maxDiff}");
    }

    #endregion
}
