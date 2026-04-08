using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

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




    #endregion

    #region Least Squares Tests



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

    [Fact(Timeout = 120000)]
    public async Task NormalDecomposition_Invert_IdentityMatrix_ReturnsIdentity()
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



    #endregion

    #region Numerical Properties Tests




    #endregion
}
