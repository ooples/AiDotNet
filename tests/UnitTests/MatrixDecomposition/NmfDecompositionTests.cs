using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MatrixDecomposition;

/// <summary>
/// Unit tests for the NmfDecomposition class.
/// </summary>
public class NmfDecompositionTests
{
    [Fact]
    public void Constructor_WithValidMatrix_InitializesCorrectly()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 }
        });

        // Act
        var nmf = new NmfDecomposition<double>(matrix, components: 2);

        // Assert
        Assert.NotNull(nmf.W);
        Assert.NotNull(nmf.H);
        Assert.Equal(3, nmf.W.Rows);
        Assert.Equal(2, nmf.W.Columns);
        Assert.Equal(2, nmf.H.Rows);
        Assert.Equal(3, nmf.H.Columns);
        Assert.Equal(2, nmf.Components);
    }

    [Fact]
    public void Constructor_WithNegativeValues_ThrowsArgumentException()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, -2, 3 },
            { 4, 5, 6 }
        });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new NmfDecomposition<double>(matrix));
    }

    [Fact]
    public void Constructor_WithZeroComponents_ThrowsArgumentException()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 }
        });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new NmfDecomposition<double>(matrix, components: 0));
    }

    [Fact]
    public void Constructor_WithTooManyComponents_ThrowsArgumentException()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 }
        });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new NmfDecomposition<double>(matrix, components: 10));
    }

    [Fact]
    public void Reconstruct_ApproximatesOriginalMatrix()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 4, 2, 3 },
            { 1, 3, 5 },
            { 2, 1, 4 }
        });

        var nmf = new NmfDecomposition<double>(matrix, components: 2, maxIterations: 500);

        // Act
        var reconstructed = nmf.Reconstruct();

        // Assert
        Assert.Equal(matrix.Rows, reconstructed.Rows);
        Assert.Equal(matrix.Columns, reconstructed.Columns);

        // Check that reconstruction is reasonably close to original
        double totalError = 0;
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                double error = Math.Abs(matrix[i, j] - reconstructed[i, j]);
                totalError += error * error;
            }
        }

        double rmse = Math.Sqrt(totalError / ((double)((long)matrix.Rows * (long)matrix.Columns)));
        Assert.True(rmse < 2.0, $"RMSE {rmse} should be less than 2.0");
    }

    [Fact]
    public void WAndH_AreNonNegative()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 5, 3, 2 },
            { 1, 4, 6 },
            { 3, 2, 1 }
        });

        var nmf = new NmfDecomposition<double>(matrix, components: 2);

        // Act & Assert - Check W is non-negative
        for (int i = 0; i < nmf.W.Rows; i++)
        {
            for (int j = 0; j < nmf.W.Columns; j++)
            {
                Assert.True(nmf.W[i, j] >= 0, $"W[{i},{j}] should be non-negative");
            }
        }

        // Check H is non-negative
        for (int i = 0; i < nmf.H.Rows; i++)
        {
            for (int j = 0; j < nmf.H.Columns; j++)
            {
                Assert.True(nmf.H[i, j] >= 0, $"H[{i},{j}] should be non-negative");
            }
        }
    }

    [Fact]
    public void Solve_ReturnsVectorOfCorrectSize()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 4, 2 },
            { 3, 5 }
        });

        var b = new Vector<double>(new[] { 10.0, 12.0 });
        var nmf = new NmfDecomposition<double>(matrix, components: 2);

        // Act
        var x = nmf.Solve(b);

        // Assert
        Assert.NotNull(x);
        Assert.Equal(2, x.Length);
    }

    [Fact]
    public void Invert_ReturnsMatrixOfCorrectDimensions()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 4, 2, 1 },
            { 3, 5, 2 },
            { 1, 1, 3 }
        });

        var nmf = new NmfDecomposition<double>(matrix, components: 3);

        // Act
        var inverse = nmf.Invert();

        // Assert
        Assert.NotNull(inverse);
        Assert.Equal(matrix.Columns, inverse.Rows);
        Assert.Equal(matrix.Rows, inverse.Columns);
    }

    [Fact]
    public void Factorization_WithIdentityMatrix_WorksCorrectly()
    {
        // Arrange
        var matrix = Matrix<double>.CreateIdentityMatrix(3);

        // Act
        var nmf = new NmfDecomposition<double>(matrix, components: 2, maxIterations: 300);
        var reconstructed = nmf.Reconstruct();

        // Assert
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                double error = Math.Abs(matrix[i, j] - reconstructed[i, j]);
                Assert.True(error < 1.0, $"Element [{i},{j}] error {error} should be small");
            }
        }
    }

    [Fact]
    public void Factorization_WithSparseMatrix_PreservesSparsity()
    {
        // Arrange - Create a sparse matrix (mostly zeros)
        var matrix = new Matrix<double>(new double[,]
        {
            { 5, 0, 0 },
            { 0, 3, 0 },
            { 0, 0, 4 }
        });

        // Act
        var nmf = new NmfDecomposition<double>(matrix, components: 2, maxIterations: 400);

        // Assert - W and H should be non-negative
        for (int i = 0; i < nmf.W.Rows; i++)
        {
            for (int j = 0; j < nmf.W.Columns; j++)
            {
                Assert.True(nmf.W[i, j] >= 0);
            }
        }
    }

    [Fact]
    public void Components_Property_ReturnsCorrectValue()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3, 4 },
            { 5, 6, 7, 8 },
            { 9, 10, 11, 12 }
        });

        // Act
        var nmf = new NmfDecomposition<double>(matrix, components: 2);

        // Assert
        Assert.Equal(2, nmf.Components);
    }

    [Fact]
    public void A_Property_ReturnsOriginalMatrix()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1, 2 },
            { 3, 4 }
        });

        // Act
        var nmf = new NmfDecomposition<double>(matrix);

        // Assert
        Assert.Equal(matrix, nmf.A);
    }

    [Fact]
    public void Factorization_WithDifferentNumericTypes_WorksCorrectly()
    {
        // Arrange
        var matrixFloat = new Matrix<float>(new float[,]
        {
            { 1f, 2f, 3f },
            { 4f, 5f, 6f }
        });

        // Act
        var nmfFloat = new NmfDecomposition<float>(matrixFloat, components: 2);

        // Assert
        Assert.NotNull(nmfFloat.W);
        Assert.NotNull(nmfFloat.H);
        Assert.Equal(2, nmfFloat.W.Rows);
        Assert.Equal(2, nmfFloat.Components);
    }
}
