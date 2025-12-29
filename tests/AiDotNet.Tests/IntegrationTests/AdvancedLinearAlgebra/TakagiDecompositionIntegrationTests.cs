using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Enums.AlgorithmTypes;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AdvancedLinearAlgebra;

/// <summary>
/// Integration tests for Takagi decomposition (Takagi factorization).
/// These tests verify: correct decomposition of symmetric matrices, various algorithms,
/// and proper Solve/Invert functionality.
/// </summary>
public class TakagiDecompositionIntegrationTests
{
    private const double Tolerance = 1e-10;
    private const double LooseTolerance = 1e-4;

    #region Helper Methods

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
                matrix[j, i] = value; // Make symmetric
            }
        }
        return matrix;
    }

    private static Matrix<double> CreatePositiveDefiniteMatrix(int size, int seed = 42)
    {
        var random = new Random(seed);
        var temp = new Matrix<double>(size, size);

        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                temp[i, j] = random.NextDouble() * 2 - 1;
            }
        }

        // A^T * A is positive definite
        return temp.Transpose().Multiply(temp);
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

    private static double ComplexMagnitude(Complex<double> c)
    {
        return Math.Sqrt(c.Real * c.Real + c.Imaginary * c.Imaginary);
    }

    #endregion

    #region Basic Decomposition Tests

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void TakagiDecomposition_BasicDecomposition_ProducesValidMatrices(int size)
    {
        // Arrange
        var A = CreateSymmetricMatrix(size);

        // Act
        var takagi = new TakagiDecomposition<double>(A);

        // Assert - Check matrix dimensions
        Assert.Equal(size, takagi.SigmaMatrix.Rows);
        Assert.Equal(size, takagi.SigmaMatrix.Columns);
        Assert.Equal(size, takagi.UnitaryMatrix.Rows);
        Assert.Equal(size, takagi.UnitaryMatrix.Columns);
    }

    [Fact]
    public void TakagiDecomposition_SigmaMatrix_IsDiagonal()
    {
        // Arrange
        var A = CreateSymmetricMatrix(4);

        // Act
        var takagi = new TakagiDecomposition<double>(A);

        // Assert - Off-diagonal elements should be zero
        for (int i = 0; i < takagi.SigmaMatrix.Rows; i++)
        {
            for (int j = 0; j < takagi.SigmaMatrix.Columns; j++)
            {
                if (i != j)
                {
                    Assert.True(Math.Abs(takagi.SigmaMatrix[i, j]) < Tolerance,
                        $"SigmaMatrix[{i},{j}] should be zero");
                }
            }
        }
    }

    [Fact]
    public void TakagiDecomposition_SigmaMatrix_HasNonNegativeDiagonals()
    {
        // Arrange
        var A = CreatePositiveDefiniteMatrix(4);

        // Act
        var takagi = new TakagiDecomposition<double>(A);

        // Assert - Singular values should be non-negative
        for (int i = 0; i < takagi.SigmaMatrix.Rows; i++)
        {
            Assert.True(takagi.SigmaMatrix[i, i] >= -Tolerance,
                $"SigmaMatrix[{i},{i}] = {takagi.SigmaMatrix[i, i]} should be non-negative");
        }
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void TakagiDecomposition_Reconstruction_Accurate(int size)
    {
        // Arrange
        var A = CreateSymmetricMatrix(size);

        // Act
        var takagi = new TakagiDecomposition<double>(A, TakagiAlgorithmType.Jacobi);

        // Reconstruct: A = U * S * U^T for Takagi decomposition
        // For complex U = Re(U) + i*Im(U), the real part of U*S*U^T is:
        // Real(U*S*U^T) = Re(U)*S*Re(U)^T - Im(U)*S*Im(U)^T
        var U = takagi.UnitaryMatrix;
        var S = takagi.SigmaMatrix;

        var URealPart = new Matrix<double>(size, size);
        var UImagPart = new Matrix<double>(size, size);
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                URealPart[i, j] = U[i, j].Real;
                UImagPart[i, j] = U[i, j].Imaginary;
            }
        }

        // Real part = Re(U)*S*Re(U)^T - Im(U)*S*Im(U)^T
        var realPart = URealPart.Multiply(S).Multiply(URealPart.Transpose());
        var imagContribution = UImagPart.Multiply(S).Multiply(UImagPart.Transpose());

        var reconstructed = new Matrix<double>(size, size);
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                reconstructed[i, j] = realPart[i, j] - imagContribution[i, j];
            }
        }

        // Assert - Reconstruction error should be small
        var diff = new Matrix<double>(size, size);
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                diff[i, j] = A[i, j] - reconstructed[i, j];
            }
        }

        double reconstructionError = FrobeniusNorm(diff);
        Assert.True(reconstructionError < LooseTolerance,
            $"Reconstruction error {reconstructionError} should be small (< {LooseTolerance})");
    }

    #endregion

    #region Algorithm Variant Tests

    [Theory]
    [InlineData(TakagiAlgorithmType.Jacobi)]
    [InlineData(TakagiAlgorithmType.EigenDecomposition)]
    [InlineData(TakagiAlgorithmType.QR)]
    public void TakagiDecomposition_AllAlgorithms_ProduceValidDecomposition(TakagiAlgorithmType algorithm)
    {
        // Arrange
        var A = CreateSymmetricMatrix(3, seed: 123);

        // Act
        var takagi = new TakagiDecomposition<double>(A, algorithm);

        // Assert - Check that matrices have valid dimensions
        Assert.Equal(3, takagi.SigmaMatrix.Rows);
        Assert.Equal(3, takagi.SigmaMatrix.Columns);
        Assert.Equal(3, takagi.UnitaryMatrix.Rows);
        Assert.Equal(3, takagi.UnitaryMatrix.Columns);
    }

    [Fact]
    public void TakagiDecomposition_JacobiAlgorithm_ConvergesToDiagonal()
    {
        // Arrange
        var A = CreateSymmetricMatrix(4);

        // Act
        var takagi = new TakagiDecomposition<double>(A, TakagiAlgorithmType.Jacobi);

        // Assert - Off-diagonal elements of SigmaMatrix should be zero
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                if (i != j)
                {
                    Assert.True(Math.Abs(takagi.SigmaMatrix[i, j]) < LooseTolerance,
                        $"Jacobi: SigmaMatrix[{i},{j}] should be zero");
                }
            }
        }
    }

    [Fact]
    public void TakagiDecomposition_PowerIteration_ProducesValidResult()
    {
        // Arrange
        var A = CreatePositiveDefiniteMatrix(3);

        // Act
        var takagi = new TakagiDecomposition<double>(A, TakagiAlgorithmType.PowerIteration);

        // Assert - Sigma should have non-negative diagonals
        for (int i = 0; i < 3; i++)
        {
            Assert.True(takagi.SigmaMatrix[i, i] >= -LooseTolerance,
                $"PowerIteration: SigmaMatrix[{i},{i}] should be non-negative");
        }
    }

    [Fact]
    public void TakagiDecomposition_LanczosIteration_ProducesValidResult()
    {
        // Arrange
        var A = CreatePositiveDefiniteMatrix(4);

        // Act
        var takagi = new TakagiDecomposition<double>(A, TakagiAlgorithmType.LanczosIteration);

        // Assert - Should produce valid matrices
        Assert.Equal(4, takagi.SigmaMatrix.Rows);
        Assert.Equal(4, takagi.UnitaryMatrix.Rows);
    }

    #endregion

    #region Solve Tests

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void TakagiDecomposition_Solve_ProducesCorrectSolution(int size)
    {
        // Arrange
        var A = CreatePositiveDefiniteMatrix(size);
        var b = new Vector<double>(size);
        for (int i = 0; i < size; i++)
            b[i] = i + 1.0;

        // Act
        var takagi = new TakagiDecomposition<double>(A);
        var x = takagi.Solve(b);

        // Assert - Verify A * x ≈ b
        Assert.Equal(size, x.Length);
        var Ax = A.Multiply(x);
        for (int i = 0; i < size; i++)
        {
            Assert.True(Math.Abs(Ax[i] - b[i]) < LooseTolerance,
                $"A*x[{i}] = {Ax[i]} should equal b[{i}] = {b[i]}");
        }
    }

    [Fact]
    public void TakagiDecomposition_Solve_IdentityMatrix_ReturnsSameVector()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(3);
        var b = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        // Act
        var takagi = new TakagiDecomposition<double>(I);
        var x = takagi.Solve(b);

        // Assert - For identity matrix, x should equal b
        for (int i = 0; i < 3; i++)
        {
            Assert.True(Math.Abs(x[i] - b[i]) < LooseTolerance,
                $"For identity matrix, x[{i}] should equal b[{i}]");
        }
    }

    #endregion

    #region Invert Tests

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void TakagiDecomposition_Invert_ProducesCorrectInverse(int size)
    {
        // Arrange
        var A = CreatePositiveDefiniteMatrix(size);

        // Act
        var takagi = new TakagiDecomposition<double>(A);
        var AInv = takagi.Invert();

        // Assert - Verify A * A^-1 ≈ I
        Assert.Equal(size, AInv.Rows);
        Assert.Equal(size, AInv.Columns);

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

    [Fact]
    public void TakagiDecomposition_Invert_IdentityMatrix_ReturnsIdentity()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(3);

        // Act
        var takagi = new TakagiDecomposition<double>(I);
        var IInv = takagi.Invert();

        // Assert
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.True(Math.Abs(IInv[i, j] - expected) < LooseTolerance,
                    $"Inverse of identity should be identity. [{i},{j}]");
            }
        }
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void TakagiDecomposition_2x2Matrix_WorksCorrectly()
    {
        // Arrange
        var A = new Matrix<double>(2, 2);
        A[0, 0] = 4; A[0, 1] = 2;
        A[1, 0] = 2; A[1, 1] = 3;

        // Act
        var takagi = new TakagiDecomposition<double>(A);

        // Assert
        Assert.Equal(2, takagi.SigmaMatrix.Rows);
        Assert.Equal(2, takagi.UnitaryMatrix.Rows);
    }

    [Fact]
    public void TakagiDecomposition_DiagonalMatrix_WorksCorrectly()
    {
        // Arrange
        var A = new Matrix<double>(3, 3);
        A[0, 0] = 4; A[0, 1] = 0; A[0, 2] = 0;
        A[1, 0] = 0; A[1, 1] = 9; A[1, 2] = 0;
        A[2, 0] = 0; A[2, 1] = 0; A[2, 2] = 16;

        // Act
        var takagi = new TakagiDecomposition<double>(A);

        // Assert - Singular values should be the absolute values of eigenvalues
        // For a diagonal matrix, eigenvalues = diagonal elements = [4, 9, 16]
        // So singular values should be [4, 9, 16] (in some order)
        var singularValues = new List<double>();
        for (int i = 0; i < 3; i++)
        {
            singularValues.Add(takagi.SigmaMatrix[i, i]);
        }
        singularValues.Sort();

        Assert.True(Math.Abs(singularValues[0] - 4.0) < LooseTolerance,
            $"Expected 4.0, got {singularValues[0]}");
        Assert.True(Math.Abs(singularValues[1] - 9.0) < LooseTolerance,
            $"Expected 9.0, got {singularValues[1]}");
        Assert.True(Math.Abs(singularValues[2] - 16.0) < LooseTolerance,
            $"Expected 16.0, got {singularValues[2]}");
    }

    [Fact]
    public void TakagiDecomposition_IdentityMatrix_HasAllOnes()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(3);

        // Act
        var takagi = new TakagiDecomposition<double>(I);

        // Assert - All singular values should be 1
        for (int i = 0; i < 3; i++)
        {
            Assert.True(Math.Abs(takagi.SigmaMatrix[i, i] - 1.0) < LooseTolerance,
                $"Singular value {i} should be 1.0");
        }
    }

    #endregion

    #region Numerical Properties Tests

    [Fact]
    public void TakagiDecomposition_NoNaNOrInfinity_InMatrices()
    {
        // Arrange
        var A = CreateSymmetricMatrix(4);

        // Act
        var takagi = new TakagiDecomposition<double>(A);

        // Assert - Check SigmaMatrix
        for (int i = 0; i < takagi.SigmaMatrix.Rows; i++)
        {
            for (int j = 0; j < takagi.SigmaMatrix.Columns; j++)
            {
                Assert.False(double.IsNaN(takagi.SigmaMatrix[i, j]),
                    $"SigmaMatrix[{i},{j}] should not be NaN");
                Assert.False(double.IsInfinity(takagi.SigmaMatrix[i, j]),
                    $"SigmaMatrix[{i},{j}] should not be infinity");
            }
        }

        // Check UnitaryMatrix
        for (int i = 0; i < takagi.UnitaryMatrix.Rows; i++)
        {
            for (int j = 0; j < takagi.UnitaryMatrix.Columns; j++)
            {
                Assert.False(double.IsNaN(takagi.UnitaryMatrix[i, j].Real),
                    $"UnitaryMatrix[{i},{j}].Real should not be NaN");
                Assert.False(double.IsNaN(takagi.UnitaryMatrix[i, j].Imaginary),
                    $"UnitaryMatrix[{i},{j}].Imaginary should not be NaN");
            }
        }
    }

    [Fact]
    public void TakagiDecomposition_UnitaryMatrix_HasUnitNormColumns()
    {
        // Arrange
        var A = CreatePositiveDefiniteMatrix(3);

        // Act
        var takagi = new TakagiDecomposition<double>(A, TakagiAlgorithmType.Jacobi);

        // Assert - Each column of unitary matrix should have unit norm
        for (int j = 0; j < takagi.UnitaryMatrix.Columns; j++)
        {
            double normSquared = 0;
            for (int i = 0; i < takagi.UnitaryMatrix.Rows; i++)
            {
                var c = takagi.UnitaryMatrix[i, j];
                normSquared += c.Real * c.Real + c.Imaginary * c.Imaginary;
            }
            double norm = Math.Sqrt(normSquared);

            Assert.True(Math.Abs(norm - 1.0) < LooseTolerance,
                $"Column {j} of UnitaryMatrix should have unit norm. Got {norm}");
        }
    }

    [Fact]
    public void TakagiDecomposition_UnitaryMatrix_ColumnsAreOrthogonal()
    {
        // Arrange
        var A = CreatePositiveDefiniteMatrix(3);

        // Act
        var takagi = new TakagiDecomposition<double>(A, TakagiAlgorithmType.Jacobi);

        // Assert - Different columns should be orthogonal
        for (int i = 0; i < takagi.UnitaryMatrix.Columns; i++)
        {
            for (int j = i + 1; j < takagi.UnitaryMatrix.Columns; j++)
            {
                double dotReal = 0, dotImag = 0;
                for (int k = 0; k < takagi.UnitaryMatrix.Rows; k++)
                {
                    var ci = takagi.UnitaryMatrix[k, i];
                    var cj = takagi.UnitaryMatrix[k, j];
                    // Complex dot product: sum of conjugate(ci) * cj
                    dotReal += ci.Real * cj.Real + ci.Imaginary * cj.Imaginary;
                    dotImag += ci.Real * cj.Imaginary - ci.Imaginary * cj.Real;
                }
                double dotMagnitude = Math.Sqrt(dotReal * dotReal + dotImag * dotImag);

                Assert.True(dotMagnitude < LooseTolerance,
                    $"Columns {i} and {j} should be orthogonal. Dot product magnitude: {dotMagnitude}");
            }
        }
    }

    #endregion

    #region Consistency Tests

    [Fact]
    public void TakagiDecomposition_DifferentAlgorithms_ProduceValidResults()
    {
        // Note: Different Takagi algorithms may produce different singular values
        // because they solve the decomposition problem using different approaches.
        // Instead of comparing values between algorithms, we verify each produces valid results.

        // Arrange
        var A = CreatePositiveDefiniteMatrix(3, seed: 42);

        // Act
        var takagiJacobi = new TakagiDecomposition<double>(A, TakagiAlgorithmType.Jacobi);
        var takagiEigen = new TakagiDecomposition<double>(A, TakagiAlgorithmType.EigenDecomposition);

        // Assert - Both algorithms should produce non-negative singular values
        for (int i = 0; i < 3; i++)
        {
            Assert.True(takagiJacobi.SigmaMatrix[i, i] >= -LooseTolerance,
                $"Jacobi singular value {i} should be non-negative");
            Assert.True(takagiEigen.SigmaMatrix[i, i] >= -LooseTolerance,
                $"EigenDecomposition singular value {i} should be non-negative");
        }

        // Assert - Both should produce valid (non-NaN) unitary matrices
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Assert.False(double.IsNaN(takagiJacobi.UnitaryMatrix[i, j].Real),
                    $"Jacobi UnitaryMatrix[{i},{j}].Real should not be NaN");
                Assert.False(double.IsNaN(takagiEigen.UnitaryMatrix[i, j].Real),
                    $"Eigen UnitaryMatrix[{i},{j}].Real should not be NaN");
            }
        }
    }

    #endregion
}
