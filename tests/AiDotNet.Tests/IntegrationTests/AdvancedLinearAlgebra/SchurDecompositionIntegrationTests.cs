using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Enums.AlgorithmTypes;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AdvancedLinearAlgebra;

/// <summary>
/// Integration tests for Schur decomposition that verify mathematical correctness.
/// These tests verify: A = U*S*U^T, U is orthogonal, S is quasi-upper triangular.
/// </summary>
public class SchurDecompositionIntegrationTests
{
    private const double Tolerance = 1e-8;
    private const double LooseTolerance = 1e-4;

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

    private static Matrix<double> CreateSymmetricMatrix(int size, int seed = 42)
    {
        var random = new Random(seed);
        var matrix = new Matrix<double>(size, size);
        for (int i = 0; i < size; i++)
        {
            for (int j = i; j < size; j++)
            {
                double value = random.NextDouble() * 10 - 5;
                matrix[i, j] = value;
                matrix[j, i] = value;
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

    private static bool IsQuasiUpperTriangular(Matrix<double> S, double tolerance)
    {
        // Quasi-upper triangular allows 2x2 blocks on diagonal (for complex eigenvalues)
        int n = S.Rows;
        for (int i = 2; i < n; i++)
        {
            for (int j = 0; j < i - 1; j++)
            {
                if (Math.Abs(S[i, j]) > tolerance)
                    return false;
            }
        }
        return true;
    }

    #endregion

    #region Reconstruction Tests (A = U*S*U^T)

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void SchurDecomposition_QR_Reconstruction(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size);

        // Act
        var schur = new SchurDecomposition<double>(A, SchurAlgorithmType.QR);

        // Assert - Verify A = U*S*U^T
        var reconstructed = schur.UnitaryMatrix
            .Multiply(schur.SchurMatrix)
            .Multiply(schur.UnitaryMatrix.Transpose());
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"A should equal U*S*U^T. Max difference: {maxDiff}");
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void SchurDecomposition_Francis_Reconstruction(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size, seed: 123);

        // Act
        var schur = new SchurDecomposition<double>(A, SchurAlgorithmType.Francis);

        // Assert
        var reconstructed = schur.UnitaryMatrix
            .Multiply(schur.SchurMatrix)
            .Multiply(schur.UnitaryMatrix.Transpose());
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance,
            $"Francis: A should equal U*S*U^T. Max difference: {maxDiff}");
    }

    #endregion

    #region Orthogonality Tests

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void SchurDecomposition_UnitaryMatrix_IsOrthogonal(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size);

        // Act
        var schur = new SchurDecomposition<double>(A);

        // Assert - U^T * U should be identity
        Assert.True(IsOrthogonal(schur.UnitaryMatrix, LooseTolerance),
            "Unitary matrix should be orthogonal (U^T * U = I)");
    }

    #endregion

    #region Triangular Structure Tests

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void SchurDecomposition_SchurMatrix_IsQuasiUpperTriangular(int size)
    {
        // Arrange
        var A = CreateTestMatrix(size);

        // Act
        var schur = new SchurDecomposition<double>(A);

        // Assert - S should be quasi-upper triangular
        Assert.True(IsQuasiUpperTriangular(schur.SchurMatrix, LooseTolerance),
            "Schur matrix should be quasi-upper triangular");
    }

    [Fact]
    public void SchurDecomposition_SymmetricMatrix_HasDiagonalSchurMatrix()
    {
        // Arrange - Symmetric matrices have real eigenvalues, so Schur form is diagonal
        var A = CreateSymmetricMatrix(4);

        // Act
        var schur = new SchurDecomposition<double>(A);

        // Assert - For symmetric matrices, Schur matrix should be nearly diagonal
        double offDiagMax = 0;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                if (i != j)
                {
                    offDiagMax = Math.Max(offDiagMax, Math.Abs(schur.SchurMatrix[i, j]));
                }
            }
        }

        Assert.True(offDiagMax < LooseTolerance,
            $"Symmetric matrix should have diagonal Schur form. Max off-diagonal: {offDiagMax}");
    }

    #endregion

    #region Special Matrix Tests

    [Fact]
    public void SchurDecomposition_IdentityMatrix_PreservesIdentity()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(4);

        // Act
        var schur = new SchurDecomposition<double>(I);

        // Assert - Schur of identity should be identity
        double maxDiff = MaxAbsDiff(I, schur.SchurMatrix);
        Assert.True(maxDiff < Tolerance,
            $"Schur of identity should be identity. Max difference: {maxDiff}");
    }

    [Fact]
    public void SchurDecomposition_DiagonalMatrix_PreservesDiagonal()
    {
        // Arrange
        var D = new Matrix<double>(4, 4);
        D[0, 0] = 4; D[1, 1] = 3; D[2, 2] = 2; D[3, 3] = 1;

        // Act
        var schur = new SchurDecomposition<double>(D);

        // Assert - Diagonal of Schur should contain same values (possibly reordered)
        var diagValues = new List<double>();
        for (int i = 0; i < 4; i++)
        {
            diagValues.Add(schur.SchurMatrix[i, i]);
        }
        diagValues.Sort();
        var expected = new List<double> { 1.0, 2.0, 3.0, 4.0 };

        for (int i = 0; i < 4; i++)
        {
            Assert.True(Math.Abs(diagValues[i] - expected[i]) < LooseTolerance,
                $"Diagonal value mismatch: got {diagValues[i]}, expected {expected[i]}");
        }
    }

    #endregion

    #region Algorithm Comparison Tests

    [Theory]
    [InlineData(SchurAlgorithmType.QR)]
    [InlineData(SchurAlgorithmType.Francis)]
    [InlineData(SchurAlgorithmType.Implicit)]
    public void SchurDecomposition_AllAlgorithms_ProduceValidDecomposition(SchurAlgorithmType algorithm)
    {
        // Arrange
        var A = CreateTestMatrix(4, seed: 42);

        // Act
        var schur = new SchurDecomposition<double>(A, algorithm);

        // Assert - Basic validity checks
        Assert.Equal(4, schur.SchurMatrix.Rows);
        Assert.Equal(4, schur.SchurMatrix.Columns);
        Assert.Equal(4, schur.UnitaryMatrix.Rows);
        Assert.Equal(4, schur.UnitaryMatrix.Columns);

        // No NaN or Inf values
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                Assert.False(double.IsNaN(schur.SchurMatrix[i, j]),
                    $"Algorithm {algorithm}: SchurMatrix has NaN at [{i},{j}]");
                Assert.False(double.IsNaN(schur.UnitaryMatrix[i, j]),
                    $"Algorithm {algorithm}: UnitaryMatrix has NaN at [{i},{j}]");
            }
        }
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void SchurDecomposition_LargeMatrix_CorrectDecomposition()
    {
        // Arrange
        var A = CreateTestMatrix(10, seed: 999);

        // Act
        var schur = new SchurDecomposition<double>(A);

        // Assert
        var reconstructed = schur.UnitaryMatrix
            .Multiply(schur.SchurMatrix)
            .Multiply(schur.UnitaryMatrix.Transpose());
        double maxDiff = MaxAbsDiff(A, reconstructed);

        Assert.True(maxDiff < LooseTolerance * 10, // Looser tolerance for larger matrices
            $"Large matrix reconstruction failed. Max difference: {maxDiff}");
    }

    #endregion
}
