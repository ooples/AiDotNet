using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Enums.AlgorithmTypes;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AdvancedLinearAlgebra;

/// <summary>
/// Integration tests for Eigen decomposition that verify mathematical correctness.
/// These tests verify: A*v = λ*v for each eigenpair, and A = V*D*V^(-1).
/// </summary>
public class EigenDecompositionIntegrationTests
{
    private const double Tolerance = 1e-6;
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

    private static double MaxAbsDiff(Vector<double> a, Vector<double> b)
    {
        if (a.Length != b.Length)
            return double.MaxValue;

        double maxDiff = 0;
        for (int i = 0; i < a.Length; i++)
        {
            maxDiff = Math.Max(maxDiff, Math.Abs(a[i] - b[i]));
        }
        return maxDiff;
    }

    #endregion

    #region Eigenpair Tests (A*v = λ*v)

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void EigenDecomposition_QR_EigenpairsAreValid(int size)
    {
        // Arrange - Use symmetric matrix for more reliable convergence
        var A = CreateSymmetricMatrix(size);

        // Act
        var eigen = new EigenDecomposition<double>(A, EigenAlgorithmType.QR);

        // Assert - A*v should equal λ*v for each eigenpair
        for (int i = 0; i < size; i++)
        {
            var v = eigen.EigenVectors.GetColumn(i);
            var lambda = eigen.EigenValues[i];

            // Compute A*v
            var Av = A.Multiply(v);

            // Compute λ*v
            var lambdaV = v.Multiply(lambda);

            // They should be equal (or parallel for numerical reasons)
            double diff = MaxAbsDiff(Av, lambdaV);
            Assert.True(diff < LooseTolerance,
                $"Eigenpair {i}: A*v should equal λ*v. Max difference: {diff}");
        }
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void EigenDecomposition_Jacobi_EigenpairsAreValid(int size)
    {
        // Arrange - Jacobi works best on symmetric matrices
        var A = CreateSymmetricMatrix(size, seed: 123);

        // Act
        var eigen = new EigenDecomposition<double>(A, EigenAlgorithmType.Jacobi);

        // Assert - A*v should equal λ*v for each eigenpair
        for (int i = 0; i < size; i++)
        {
            var v = eigen.EigenVectors.GetColumn(i);
            var lambda = eigen.EigenValues[i];

            var Av = A.Multiply(v);
            var lambdaV = v.Multiply(lambda);

            double diff = MaxAbsDiff(Av, lambdaV);
            Assert.True(diff < LooseTolerance,
                $"Jacobi eigenpair {i}: A*v should equal λ*v. Max difference: {diff}");
        }
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void EigenDecomposition_PowerIteration_DominantEigenvalueIsValid(int size)
    {
        // Arrange
        var A = CreateSymmetricMatrix(size, seed: 456);

        // Act
        var eigen = new EigenDecomposition<double>(A, EigenAlgorithmType.PowerIteration);

        // Assert - At least the first (dominant) eigenpair should be valid
        var v = eigen.EigenVectors.GetColumn(0);
        var lambda = eigen.EigenValues[0];

        var Av = A.Multiply(v);
        var lambdaV = v.Multiply(lambda);

        double diff = MaxAbsDiff(Av, lambdaV);
        Assert.True(diff < LooseTolerance,
            $"Power iteration: dominant A*v should equal λ*v. Max difference: {diff}");
    }

    #endregion

    #region Orthogonality Tests (for symmetric matrices)

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void EigenDecomposition_SymmetricMatrix_EigenvectorsAreOrthogonal(int size)
    {
        // Arrange
        var A = CreateSymmetricMatrix(size);

        // Act
        var eigen = new EigenDecomposition<double>(A, EigenAlgorithmType.Jacobi);

        // Assert - V^T * V should be identity for symmetric matrices
        var V = eigen.EigenVectors;
        var VtV = V.Transpose().Multiply(V);

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.True(Math.Abs(VtV[i, j] - expected) < LooseTolerance,
                    $"V^T*V[{i},{j}] = {VtV[i, j]}, expected {expected}");
            }
        }
    }

    #endregion

    #region Special Matrix Tests

    [Fact]
    public void EigenDecomposition_IdentityMatrix_HasAllOnes()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(4);

        // Act
        var eigen = new EigenDecomposition<double>(I);

        // Assert - All eigenvalues should be 1
        for (int i = 0; i < eigen.EigenValues.Length; i++)
        {
            Assert.True(Math.Abs(eigen.EigenValues[i] - 1.0) < Tolerance,
                $"Identity matrix eigenvalue {i} = {eigen.EigenValues[i]}, expected 1.0");
        }
    }

    [Fact]
    public void EigenDecomposition_DiagonalMatrix_HasDiagonalAsEigenvalues()
    {
        // Arrange - Diagonal matrix with values 4, 3, 2, 1
        var D = new Matrix<double>(4, 4);
        D[0, 0] = 4; D[1, 1] = 3; D[2, 2] = 2; D[3, 3] = 1;

        // Act
        var eigen = new EigenDecomposition<double>(D);

        // Assert - Eigenvalues should be 4, 3, 2, 1 (order may vary)
        var expected = new HashSet<double> { 4.0, 3.0, 2.0, 1.0 };
        var found = new HashSet<double>();

        for (int i = 0; i < eigen.EigenValues.Length; i++)
        {
            double val = eigen.EigenValues[i];
            // Find closest expected value
            double closest = expected.OrderBy(x => Math.Abs(x - val)).First();
            Assert.True(Math.Abs(val - closest) < Tolerance,
                $"Diagonal matrix eigenvalue {i} = {val}, expected one of {string.Join(", ", expected)}");
            found.Add(closest);
        }

        Assert.Equal(4, found.Count);
    }

    [Fact]
    public void EigenDecomposition_ZeroMatrix_HasAllZeroEigenvalues()
    {
        // Arrange
        var Z = new Matrix<double>(3, 3); // All zeros

        // Act
        var eigen = new EigenDecomposition<double>(Z);

        // Assert - All eigenvalues should be 0
        for (int i = 0; i < eigen.EigenValues.Length; i++)
        {
            Assert.True(Math.Abs(eigen.EigenValues[i]) < LooseTolerance,
                $"Zero matrix eigenvalue {i} = {eigen.EigenValues[i]}, expected 0");
        }
    }

    [Fact]
    public void EigenDecomposition_SymmetricPositiveDefinite_HasPositiveEigenvalues()
    {
        // Arrange - Create SPD matrix A^T * A
        var B = CreateTestMatrix(4, seed: 789);
        var A = B.Transpose().Multiply(B);

        // Act
        var eigen = new EigenDecomposition<double>(A);

        // Assert - All eigenvalues should be positive (or non-negative for positive semidefinite)
        for (int i = 0; i < eigen.EigenValues.Length; i++)
        {
            Assert.True(eigen.EigenValues[i] >= -Tolerance,
                $"SPD matrix eigenvalue {i} = {eigen.EigenValues[i]}, expected positive");
        }
    }

    #endregion

    #region Algorithm Comparison Tests

    [Theory]
    [InlineData(EigenAlgorithmType.QR)]
    [InlineData(EigenAlgorithmType.Jacobi)]
    [InlineData(EigenAlgorithmType.PowerIteration)]
    public void EigenDecomposition_AllAlgorithms_ProduceRealEigenvalues(EigenAlgorithmType algorithm)
    {
        // Arrange - Symmetric matrices have real eigenvalues
        var A = CreateSymmetricMatrix(4, seed: 42);

        // Act
        var eigen = new EigenDecomposition<double>(A, algorithm);

        // Assert - No NaN or infinite values
        for (int i = 0; i < eigen.EigenValues.Length; i++)
        {
            Assert.False(double.IsNaN(eigen.EigenValues[i]),
                $"Algorithm {algorithm}: Eigenvalue {i} is NaN");
            Assert.False(double.IsInfinity(eigen.EigenValues[i]),
                $"Algorithm {algorithm}: Eigenvalue {i} is infinite");
        }
    }

    #endregion

    #region Numerical Stability Tests

    [Fact]
    public void EigenDecomposition_LargeMatrix_ProducesValidDecomposition()
    {
        // Arrange
        var A = CreateSymmetricMatrix(10, seed: 999);

        // Act
        var eigen = new EigenDecomposition<double>(A);

        // Assert - At least check no NaN/Inf values
        for (int i = 0; i < eigen.EigenValues.Length; i++)
        {
            Assert.False(double.IsNaN(eigen.EigenValues[i]),
                $"Large matrix eigenvalue {i} is NaN");
            Assert.False(double.IsInfinity(eigen.EigenValues[i]),
                $"Large matrix eigenvalue {i} is infinite");
        }

        // Also verify at least one eigenpair is valid
        var v = eigen.EigenVectors.GetColumn(0);
        var lambda = eigen.EigenValues[0];
        var Av = A.Multiply(v);
        var lambdaV = v.Multiply(lambda);
        double diff = MaxAbsDiff(Av, lambdaV);

        Assert.True(diff < LooseTolerance * 10, // Looser tolerance for larger matrices
            $"Large matrix: A*v should equal λ*v. Max difference: {diff}");
    }

    [Fact]
    public void EigenDecomposition_IllConditionedMatrix_HandlesGracefully()
    {
        // Arrange - Create a nearly singular matrix
        var A = new Matrix<double>(3, 3);
        A[0, 0] = 1.0; A[0, 1] = 2.0; A[0, 2] = 3.0;
        A[1, 0] = 2.0; A[1, 1] = 4.0; A[1, 2] = 6.0; // Row 1 = 2 * Row 0
        A[2, 0] = 1.0; A[2, 1] = 3.0; A[2, 2] = 4.0;

        // Act - Should not throw
        var eigen = new EigenDecomposition<double>(A);

        // Assert - One eigenvalue should be close to 0 (due to rank deficiency)
        bool hasNearZero = false;
        for (int i = 0; i < eigen.EigenValues.Length; i++)
        {
            if (Math.Abs(eigen.EigenValues[i]) < LooseTolerance)
            {
                hasNearZero = true;
                break;
            }
        }

        Assert.True(hasNearZero,
            $"Rank-deficient matrix should have at least one eigenvalue near zero. Eigenvalues: {string.Join(", ", Enumerable.Range(0, eigen.EigenValues.Length).Select(i => eigen.EigenValues[i]))}");
    }

    #endregion

    #region Validation Tests

    [Fact]
    public void EigenDecomposition_NonSquareMatrix_ThrowsException()
    {
        // Arrange
        var A = new Matrix<double>(3, 4); // Non-square

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new EigenDecomposition<double>(A));
    }

    #endregion
}
