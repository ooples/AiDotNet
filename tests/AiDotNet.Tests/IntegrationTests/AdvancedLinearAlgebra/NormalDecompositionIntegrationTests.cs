using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AdvancedLinearAlgebra;

/// <summary>
/// Integration tests for Normal Equation decomposition.
/// These tests verify: least squares solution for overdetermined systems,
/// pseudo-inverse computation, and proper handling of various matrix shapes.
/// </summary>
public class NormalDecompositionIntegrationTests
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

    private static Matrix<double> CreateFullRankMatrix(int rows, int cols, int seed = 42)
    {
        var random = new Random(seed);
        var matrix = new Matrix<double>(rows, cols);

        // Use integer values to avoid floating-point precision issues
        // when computing A^T*A (which must be perfectly symmetric for Cholesky)
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                // Use small integers to avoid overflow and maintain precision
                matrix[i, j] = random.Next(-5, 6);
            }
        }

        // Add diagonal dominance to ensure full rank and positive definiteness
        int minDim = Math.Min(rows, cols);
        for (int i = 0; i < minDim; i++)
        {
            matrix[i, i] += 10 + random.Next(1, 5);
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

    private static double VectorNorm(Vector<double> v)
    {
        double sum = 0;
        for (int i = 0; i < v.Length; i++)
        {
            sum += v[i] * v[i];
        }
        return Math.Sqrt(sum);
    }

    #endregion

    #region Basic Solve Tests

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void NormalDecomposition_Solve_SquareMatrix_ProducesCorrectSolution(int size)
    {
        // Arrange
        var A = CreateFullRankMatrix(size, size);
        var xExpected = new Vector<double>(size);
        for (int i = 0; i < size; i++)
            xExpected[i] = i + 1.0;

        var b = A.Multiply(xExpected);

        // Act
        var normal = new NormalDecomposition<double>(A);
        var xComputed = normal.Solve(b);

        // Assert - Verify A*x_computed ≈ b
        var bComputed = A.Multiply(xComputed);
        for (int i = 0; i < size; i++)
        {
            Assert.True(Math.Abs(bComputed[i] - b[i]) < LooseTolerance,
                $"A*x should equal b. Component {i}: expected {b[i]}, got {bComputed[i]}");
        }
    }

    [Theory]
    [InlineData(6, 3)]
    [InlineData(8, 4)]
    [InlineData(10, 5)]
    public void NormalDecomposition_Solve_OverdeterminedSystem_MinimizesResidual(int rows, int cols)
    {
        // Arrange - Create overdetermined system
        var A = CreateFullRankMatrix(rows, cols);
        var xExpected = new Vector<double>(cols);
        for (int i = 0; i < cols; i++)
            xExpected[i] = i + 1.0;

        var b = A.Multiply(xExpected);

        // Act
        var normal = new NormalDecomposition<double>(A);
        var xComputed = normal.Solve(b);

        // Assert - For consistent overdetermined systems, solution should be exact
        for (int i = 0; i < cols; i++)
        {
            Assert.True(Math.Abs(xComputed[i] - xExpected[i]) < LooseTolerance,
                $"Solution x[{i}]: expected {xExpected[i]}, got {xComputed[i]}");
        }
    }

    [Fact]
    public void NormalDecomposition_Solve_IdentityMatrix_ReturnsSameVector()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(4);
        var b = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

        // Act
        var normal = new NormalDecomposition<double>(I);
        var x = normal.Solve(b);

        // Assert
        for (int i = 0; i < 4; i++)
        {
            Assert.True(Math.Abs(x[i] - b[i]) < Tolerance,
                $"Solution for identity matrix should be the input vector. Index {i}");
        }
    }

    #endregion

    #region Least Squares Tests

    [Fact]
    public void NormalDecomposition_Solve_LeastSquaresFit_MinimizesResidual()
    {
        // Arrange - Create an inconsistent overdetermined system
        // The solution should minimize ||Ax - b||^2
        var A = new Matrix<double>(4, 2);
        A[0, 0] = 1; A[0, 1] = 0;
        A[1, 0] = 1; A[1, 1] = 1;
        A[2, 0] = 1; A[2, 1] = 2;
        A[3, 0] = 1; A[3, 1] = 3;

        // b values that don't lie exactly on a line
        var b = new Vector<double>(new[] { 1.1, 2.0, 2.9, 4.2 });

        // Act
        var normal = new NormalDecomposition<double>(A);
        var x = normal.Solve(b);

        // Assert - Verify the residual is minimized (normal equations are satisfied)
        // A^T * A * x = A^T * b
        var AtA = A.Transpose().Multiply(A);
        var Atb = A.Transpose().Multiply(b);
        var AtAx = AtA.Multiply(x);

        for (int i = 0; i < 2; i++)
        {
            Assert.True(Math.Abs(AtAx[i] - Atb[i]) < LooseTolerance,
                $"Normal equations should be satisfied. Index {i}");
        }
    }

    [Fact]
    public void NormalDecomposition_Solve_LinearRegression_ProducesValidFit()
    {
        // Arrange - Linear regression: y = a + bx
        // Points: (0, 1), (1, 3), (2, 5), (3, 7) should give y = 1 + 2x
        var A = new Matrix<double>(4, 2);
        A[0, 0] = 1; A[0, 1] = 0;
        A[1, 0] = 1; A[1, 1] = 1;
        A[2, 0] = 1; A[2, 1] = 2;
        A[3, 0] = 1; A[3, 1] = 3;

        var b = new Vector<double>(new[] { 1.0, 3.0, 5.0, 7.0 });

        // Act
        var normal = new NormalDecomposition<double>(A);
        var coefficients = normal.Solve(b);

        // Assert - Should get intercept ≈ 1, slope ≈ 2
        Assert.True(Math.Abs(coefficients[0] - 1.0) < LooseTolerance,
            $"Intercept should be 1.0, got {coefficients[0]}");
        Assert.True(Math.Abs(coefficients[1] - 2.0) < LooseTolerance,
            $"Slope should be 2.0, got {coefficients[1]}");
    }

    #endregion

    #region Matrix Inversion Tests

    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void NormalDecomposition_Invert_SquareMatrix_ProducesCorrectInverse(int size)
    {
        // Arrange
        var A = CreateFullRankMatrix(size, size);

        // Act
        var normal = new NormalDecomposition<double>(A);
        var AInv = normal.Invert();

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

    [Fact]
    public void NormalDecomposition_Invert_IdentityMatrix_ReturnsIdentity()
    {
        // Arrange
        var I = Matrix<double>.CreateIdentityMatrix(4);

        // Act
        var normal = new NormalDecomposition<double>(I);
        var IInv = normal.Invert();

        // Assert
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.True(Math.Abs(IInv[i, j] - expected) < LooseTolerance,
                    $"Inverse of identity should be identity. [{i},{j}]");
            }
        }
    }

    [Theory]
    [InlineData(6, 3)]
    [InlineData(8, 4)]
    public void NormalDecomposition_Invert_TallMatrix_ProducesPseudoInverse(int rows, int cols)
    {
        // Arrange
        var A = CreateFullRankMatrix(rows, cols);

        // Act
        var normal = new NormalDecomposition<double>(A);
        var APseudoInv = normal.Invert();

        // Assert - For full column rank matrices: A^+ * A = I (cols x cols)
        var product = APseudoInv.Multiply(A);
        for (int i = 0; i < cols; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double expected = (i == j) ? 1.0 : 0.0;
                Assert.True(Math.Abs(product[i, j] - expected) < LooseTolerance,
                    $"(A^+ * A)[{i},{j}] = {product[i, j]}, expected {expected}");
            }
        }
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void NormalDecomposition_2x2Matrix_WorksCorrectly()
    {
        // Arrange
        var A = new Matrix<double>(2, 2);
        A[0, 0] = 3; A[0, 1] = 1;
        A[1, 0] = 1; A[1, 1] = 2;

        var b = new Vector<double>(new[] { 5.0, 5.0 });

        // Act
        var normal = new NormalDecomposition<double>(A);
        var x = normal.Solve(b);

        // Assert - Verify solution
        var bComputed = A.Multiply(x);
        for (int i = 0; i < 2; i++)
        {
            Assert.True(Math.Abs(bComputed[i] - b[i]) < LooseTolerance);
        }
    }

    [Fact]
    public void NormalDecomposition_DiagonalMatrix_WorksCorrectly()
    {
        // Arrange
        var A = new Matrix<double>(3, 3);
        A[0, 0] = 2; A[0, 1] = 0; A[0, 2] = 0;
        A[1, 0] = 0; A[1, 1] = 3; A[1, 2] = 0;
        A[2, 0] = 0; A[2, 1] = 0; A[2, 2] = 4;

        var b = new Vector<double>(new[] { 6.0, 12.0, 20.0 });

        // Act
        var normal = new NormalDecomposition<double>(A);
        var x = normal.Solve(b);

        // Assert
        Assert.True(Math.Abs(x[0] - 3.0) < LooseTolerance);
        Assert.True(Math.Abs(x[1] - 4.0) < LooseTolerance);
        Assert.True(Math.Abs(x[2] - 5.0) < LooseTolerance);
    }

    #endregion

    #region Numerical Properties Tests

    [Fact]
    public void NormalDecomposition_Solve_AndInvert_AreConsistent()
    {
        // Arrange
        var A = CreateFullRankMatrix(4, 4);
        var b = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

        // Act
        var normal = new NormalDecomposition<double>(A);
        var xFromSolve = normal.Solve(b);
        var AInv = normal.Invert();
        var xFromInverse = AInv.Multiply(b);

        // Assert - Both methods should give same result
        for (int i = 0; i < 4; i++)
        {
            Assert.True(Math.Abs(xFromSolve[i] - xFromInverse[i]) < LooseTolerance,
                $"Solve and inverse multiplication should match. Index {i}");
        }
    }

    [Fact]
    public void NormalDecomposition_NoNaNOrInfinity_InResults()
    {
        // Arrange
        var A = CreateFullRankMatrix(5, 5);
        var b = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        // Act
        var normal = new NormalDecomposition<double>(A);
        var x = normal.Solve(b);
        var AInv = normal.Invert();

        // Assert - Check solution
        for (int i = 0; i < x.Length; i++)
        {
            Assert.False(double.IsNaN(x[i]), $"x[{i}] should not be NaN");
            Assert.False(double.IsInfinity(x[i]), $"x[{i}] should not be infinity");
        }

        // Check inverse
        for (int i = 0; i < AInv.Rows; i++)
        {
            for (int j = 0; j < AInv.Columns; j++)
            {
                Assert.False(double.IsNaN(AInv[i, j]), $"AInv[{i},{j}] should not be NaN");
                Assert.False(double.IsInfinity(AInv[i, j]), $"AInv[{i},{j}] should not be infinity");
            }
        }
    }

    [Fact]
    public void NormalDecomposition_ResidualIsOrthogonalToColumnSpace()
    {
        // Arrange - For a least squares solution, the residual r = b - Ax
        // should be orthogonal to the column space of A (i.e., A^T * r = 0)
        var A = CreateFullRankMatrix(6, 3);
        var b = new Vector<double>(6);
        for (int i = 0; i < 6; i++)
            b[i] = i + 1.0;

        // Act
        var normal = new NormalDecomposition<double>(A);
        var x = normal.Solve(b);

        // Compute residual
        var Ax = A.Multiply(x);
        var residual = new Vector<double>(6);
        for (int i = 0; i < 6; i++)
            residual[i] = b[i] - Ax[i];

        // Compute A^T * residual
        var AtResidual = A.Transpose().Multiply(residual);

        // Assert - A^T * residual should be approximately zero
        for (int i = 0; i < 3; i++)
        {
            Assert.True(Math.Abs(AtResidual[i]) < LooseTolerance,
                $"A^T * residual[{i}] should be 0, got {AtResidual[i]}");
        }
    }

    #endregion
}
