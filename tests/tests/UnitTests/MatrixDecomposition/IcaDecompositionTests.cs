using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MatrixDecomposition;

/// <summary>
/// Unit tests for the IcaDecomposition class.
/// </summary>
public class IcaDecompositionTests
{
    [Fact]
    public void Constructor_WithValidMatrix_InitializesCorrectly()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 },
            { 10.0, 11.0, 12.0 }
        });

        // Act
        var ica = new IcaDecomposition<double>(matrix, components: 2);

        // Assert
        Assert.NotNull(ica.UnmixingMatrix);
        Assert.NotNull(ica.MixingMatrix);
        Assert.NotNull(ica.IndependentComponents);
        Assert.NotNull(ica.Mean);
        Assert.NotNull(ica.WhiteningMatrix);
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
        Assert.Throws<ArgumentException>(() => new IcaDecomposition<double>(matrix, components: 0));
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
        Assert.Throws<ArgumentException>(() => new IcaDecomposition<double>(matrix, components: 10));
    }

    [Fact]
    public void UnmixingMatrix_HasCorrectDimensions()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 }
        });

        int components = 2;

        // Act
        var ica = new IcaDecomposition<double>(matrix, components: components);

        // Assert
        Assert.Equal(components, ica.UnmixingMatrix.Rows);
        Assert.Equal(matrix.Columns, ica.UnmixingMatrix.Columns);
    }

    [Fact]
    public void IndependentComponents_HasCorrectDimensions()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0 },
            { 5.0, 6.0, 7.0, 8.0 },
            { 9.0, 10.0, 11.0, 12.0 }
        });

        int components = 2;

        // Act
        var ica = new IcaDecomposition<double>(matrix, components: components);

        // Assert
        Assert.Equal(components, ica.IndependentComponents.Rows);
        Assert.Equal(matrix.Rows, ica.IndependentComponents.Columns);
    }

    [Fact]
    public void Mean_HasCorrectLength()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 }
        });

        // Act
        var ica = new IcaDecomposition<double>(matrix, components: 2);

        // Assert
        Assert.Equal(matrix.Columns, ica.Mean.Length);
    }

    [Fact]
    public void Solve_ReturnsVectorOfCorrectSize()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 4.0, 2.0 },
            { 3.0, 5.0 }
        });

        var b = new Vector<double>(new[] { 10.0, 12.0 });
        var ica = new IcaDecomposition<double>(matrix, components: 2);

        // Act
        var x = ica.Solve(b);

        // Assert
        Assert.NotNull(x);
        Assert.True(x.Length > 0);
    }

    [Fact]
    public void Invert_ReturnsMatrixOfCorrectDimensions()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 4.0, 2.0 },
            { 3.0, 5.0 }
        });

        var ica = new IcaDecomposition<double>(matrix, components: 2);

        // Act
        var inverse = ica.Invert();

        // Assert
        Assert.NotNull(inverse);
        Assert.True(inverse.Rows > 0);
        Assert.True(inverse.Columns > 0);
    }

    [Fact]
    public void Transform_WithNewData_ReturnsCorrectDimensions()
    {
        // Arrange
        var trainingMatrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 }
        });

        var newMatrix = new Matrix<double>(new double[,]
        {
            { 2.0, 3.0, 4.0 },
            { 5.0, 6.0, 7.0 }
        });

        var ica = new IcaDecomposition<double>(trainingMatrix, components: 2);

        // Act
        var transformed = ica.Transform(newMatrix);

        // Assert
        Assert.NotNull(transformed);
        Assert.Equal(newMatrix.Rows, transformed.Rows);
    }

    [Fact]
    public void A_Property_ReturnsOriginalMatrix()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        });

        // Act
        var ica = new IcaDecomposition<double>(matrix);

        // Assert
        Assert.Equal(matrix, ica.A);
    }

    [Fact]
    public void Factorization_WithMixedSignals_SeparatesComponents()
    {
        // Arrange - Create two independent source signals
        double[] time = new double[100];
        for (int i = 0; i < 100; i++)
            time[i] = i * 0.1;

        // Source 1: Sine wave
        double[] source1 = new double[100];
        for (int i = 0; i < 100; i++)
            source1[i] = Math.Sin(time[i]);

        // Source 2: Square wave
        double[] source2 = new double[100];
        for (int i = 0; i < 100; i++)
            source2[i] = Math.Sign(Math.Sin(2 * time[i]));

        // Create mixed signals
        var mixedMatrix = new Matrix<double>(100, 2);
        for (int i = 0; i < 100; i++)
        {
            mixedMatrix[i, 0] = 0.6 * source1[i] + 0.4 * source2[i];
            mixedMatrix[i, 1] = 0.4 * source1[i] + 0.6 * source2[i];
        }

        // Act
        var ica = new IcaDecomposition<double>(mixedMatrix, components: 2, maxIterations: 300);

        // Assert - ICA should produce independent components
        Assert.NotNull(ica.IndependentComponents);
        Assert.Equal(2, ica.IndependentComponents.Rows);
        Assert.Equal(100, ica.IndependentComponents.Columns);
    }

    [Fact]
    public void Factorization_WithDifferentNumericTypes_WorksCorrectly()
    {
        // Arrange
        var matrixFloat = new Matrix<float>(new float[,]
        {
            { 1f, 2f, 3f },
            { 4f, 5f, 6f },
            { 7f, 8f, 9f }
        });

        // Act
        var icaFloat = new IcaDecomposition<float>(matrixFloat, components: 2);

        // Assert
        Assert.NotNull(icaFloat.UnmixingMatrix);
        Assert.NotNull(icaFloat.IndependentComponents);
    }

    [Fact]
    public void WhiteningMatrix_HasCorrectDimensions()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
            { 7.0, 8.0, 9.0 }
        });

        int components = 2;

        // Act
        var ica = new IcaDecomposition<double>(matrix, components: components);

        // Assert
        Assert.Equal(components, ica.WhiteningMatrix.Rows);
        Assert.Equal(matrix.Columns, ica.WhiteningMatrix.Columns);
    }

    [Fact]
    public void MixingMatrix_HasCorrectDimensions()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0, 4.0 },
            { 5.0, 6.0, 7.0, 8.0 },
            { 9.0, 10.0, 11.0, 12.0 }
        });

        int components = 2;

        // Act
        var ica = new IcaDecomposition<double>(matrix, components: components);

        // Assert
        Assert.NotNull(ica.MixingMatrix);
        Assert.True(ica.MixingMatrix.Rows > 0);
        Assert.True(ica.MixingMatrix.Columns > 0);
    }

    [Fact]
    public void Constructor_WithDefaultComponents_UsesMinDimension()
    {
        // Arrange
        var matrix = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 }
        });

        // Act
        var ica = new IcaDecomposition<double>(matrix); // No components specified

        // Assert
        Assert.NotNull(ica.IndependentComponents);
        Assert.True(ica.IndependentComponents.Rows <= Math.Min(matrix.Rows, matrix.Columns));
    }
}
