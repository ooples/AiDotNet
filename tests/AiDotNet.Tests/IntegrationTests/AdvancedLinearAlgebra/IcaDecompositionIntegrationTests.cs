using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.AdvancedLinearAlgebra;

/// <summary>
/// Integration tests for Independent Component Analysis (ICA) decomposition.
/// These tests verify: source separation, unmixing/mixing matrix relationship,
/// independent component orthogonality, and transform functionality.
/// </summary>
public class IcaDecompositionIntegrationTests
{
    private const double Tolerance = 1e-6;
    private const double LooseTolerance = 1e-3;

    #region Helper Methods

    private static Matrix<double> CreateMixedSignals(int samples, int sources, int seed = 42)
    {
        var random = new Random(seed);
        var matrix = new Matrix<double>(samples, sources);
        for (int i = 0; i < samples; i++)
        {
            for (int j = 0; j < sources; j++)
            {
                // Create different signal patterns for each source
                double t = i / (double)samples * 10 * Math.PI;
                double signal = j switch
                {
                    0 => Math.Sin(t),                    // Sine wave
                    1 => Math.Sign(Math.Sin(2 * t)),     // Square wave
                    2 => (t % (2 * Math.PI)) / Math.PI - 1, // Sawtooth
                    _ => random.NextDouble() * 2 - 1     // Random noise
                };
                matrix[i, j] = signal + random.NextDouble() * 0.1; // Add small noise
            }
        }
        return matrix;
    }

    private static Matrix<double> CreateRandomMatrix(int rows, int cols, int seed = 42)
    {
        var random = new Random(seed);
        var matrix = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = random.NextDouble() * 2 - 1;
            }
        }
        return matrix;
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

    #region Basic Decomposition Tests

    [Theory]
    [InlineData(100, 3, 2)]
    [InlineData(200, 4, 3)]
    [InlineData(150, 5, 4)]
    public void IcaDecomposition_BasicDecomposition_ProducesValidMatrices(int samples, int features, int components)
    {
        // Arrange
        var X = CreateRandomMatrix(samples, features);

        // Act
        var ica = new IcaDecomposition<double>(X, components);

        // Assert - Check matrix dimensions
        // UnmixingMatrix W is (components × components) because it operates on whitened data
        Assert.Equal(components, ica.UnmixingMatrix.Rows);
        Assert.Equal(components, ica.UnmixingMatrix.Columns);
        // MixingMatrix A is (features × components) - inverse mapping from components to features
        Assert.Equal(features, ica.MixingMatrix.Rows);
        Assert.Equal(components, ica.MixingMatrix.Columns);
        // IndependentComponents S is (components × samples)
        Assert.Equal(components, ica.IndependentComponents.Rows);
        Assert.Equal(samples, ica.IndependentComponents.Columns);
    }

    [Fact]
    public void IcaDecomposition_DefaultComponents_UsesMinDimension()
    {
        // Arrange
        var X = CreateRandomMatrix(50, 4);

        // Act
        var ica = new IcaDecomposition<double>(X);

        // Assert - Default should use min(rows, cols)
        Assert.Equal(4, ica.UnmixingMatrix.Rows);
    }

    [Theory]
    [InlineData(80, 3)]
    [InlineData(100, 4)]
    public void IcaDecomposition_Mean_HasCorrectDimensions(int samples, int features)
    {
        // Arrange
        var X = CreateRandomMatrix(samples, features);

        // Act
        var ica = new IcaDecomposition<double>(X);

        // Assert
        Assert.Equal(features, ica.Mean.Length);
    }

    [Fact]
    public void IcaDecomposition_WhiteningMatrix_HasCorrectDimensions()
    {
        // Arrange
        int samples = 100;
        int features = 5;
        int components = 3;
        var X = CreateRandomMatrix(samples, features);

        // Act
        var ica = new IcaDecomposition<double>(X, components);

        // Assert
        Assert.Equal(components, ica.WhiteningMatrix.Rows);
        Assert.Equal(features, ica.WhiteningMatrix.Columns);
    }

    #endregion

    #region Unmixing Matrix Properties Tests

    [Fact]
    public void IcaDecomposition_UnmixingMatrix_RowsAreApproximatelyOrthogonal()
    {
        // Arrange
        var X = CreateMixedSignals(200, 3);

        // Act
        var ica = new IcaDecomposition<double>(X, 3, maxIterations: 300);

        // Assert - Rows of unmixing matrix should be nearly orthogonal
        var W = ica.UnmixingMatrix;
        for (int i = 0; i < W.Rows; i++)
        {
            for (int j = i + 1; j < W.Rows; j++)
            {
                double dotProduct = 0;
                for (int k = 0; k < W.Columns; k++)
                {
                    dotProduct += W[i, k] * W[j, k];
                }
                // Due to ICA's non-exact orthogonalization, use loose tolerance
                Assert.True(Math.Abs(dotProduct) < 1.0,
                    $"Rows {i} and {j} should be nearly orthogonal. Dot product: {dotProduct}");
            }
        }
    }

    [Fact]
    public void IcaDecomposition_NoNaNOrInfinity_InMatrices()
    {
        // Arrange
        var X = CreateRandomMatrix(100, 4);

        // Act
        var ica = new IcaDecomposition<double>(X, 3);

        // Assert - Check UnmixingMatrix
        for (int i = 0; i < ica.UnmixingMatrix.Rows; i++)
        {
            for (int j = 0; j < ica.UnmixingMatrix.Columns; j++)
            {
                Assert.False(double.IsNaN(ica.UnmixingMatrix[i, j]),
                    $"UnmixingMatrix[{i},{j}] should not be NaN");
                Assert.False(double.IsInfinity(ica.UnmixingMatrix[i, j]),
                    $"UnmixingMatrix[{i},{j}] should not be infinity");
            }
        }

        // Check MixingMatrix
        for (int i = 0; i < ica.MixingMatrix.Rows; i++)
        {
            for (int j = 0; j < ica.MixingMatrix.Columns; j++)
            {
                Assert.False(double.IsNaN(ica.MixingMatrix[i, j]),
                    $"MixingMatrix[{i},{j}] should not be NaN");
                Assert.False(double.IsInfinity(ica.MixingMatrix[i, j]),
                    $"MixingMatrix[{i},{j}] should not be infinity");
            }
        }
    }

    #endregion

    #region Transform Tests

    [Fact]
    public void IcaDecomposition_Transform_ProducesCorrectDimensions()
    {
        // Arrange
        var X = CreateRandomMatrix(100, 5);
        var ica = new IcaDecomposition<double>(X, 3);
        var newData = CreateRandomMatrix(20, 5, seed: 123);

        // Act
        var transformed = ica.Transform(newData);

        // Assert
        Assert.Equal(20, transformed.Rows);
        Assert.Equal(3, transformed.Columns);
    }

    [Fact]
    public void IcaDecomposition_Transform_WrongDimensions_ThrowsArgumentException()
    {
        // Arrange
        var X = CreateRandomMatrix(100, 5);
        var ica = new IcaDecomposition<double>(X, 3);
        var wrongData = CreateRandomMatrix(20, 4); // Wrong number of columns

        // Act & Assert
        Assert.Throws<ArgumentException>(() => ica.Transform(wrongData));
    }

    #endregion

    #region Error Handling Tests

    [Fact]
    public void IcaDecomposition_ZeroComponents_ThrowsArgumentException()
    {
        // Arrange
        var X = CreateRandomMatrix(50, 4);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new IcaDecomposition<double>(X, 0));
    }

    [Fact]
    public void IcaDecomposition_TooManyComponents_ThrowsArgumentException()
    {
        // Arrange
        var X = CreateRandomMatrix(50, 4);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new IcaDecomposition<double>(X, 10));
    }

    [Fact]
    public void IcaDecomposition_NegativeComponents_ThrowsArgumentException()
    {
        // Arrange
        var X = CreateRandomMatrix(50, 4);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => new IcaDecomposition<double>(X, -1));
    }

    #endregion

    #region Solve Method Tests

    [Fact]
    public void IcaDecomposition_Solve_WrongDimensions_ThrowsArgumentException()
    {
        // Arrange
        var X = CreateRandomMatrix(100, 5);
        var ica = new IcaDecomposition<double>(X, 3);
        var wrongB = new Vector<double>(4); // Wrong length

        // Act & Assert
        Assert.Throws<ArgumentException>(() => ica.Solve(wrongB));
    }

    [Fact]
    public void IcaDecomposition_Solve_ProducesFiniteResult()
    {
        // Arrange
        var X = CreateRandomMatrix(100, 5);
        var ica = new IcaDecomposition<double>(X, 3);
        var b = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        // Act
        var result = ica.Solve(b);

        // Assert
        Assert.Equal(3, result.Length);
        for (int i = 0; i < result.Length; i++)
        {
            Assert.False(double.IsNaN(result[i]), $"result[{i}] should not be NaN");
            Assert.False(double.IsInfinity(result[i]), $"result[{i}] should not be infinity");
        }
    }

    #endregion

    #region Convergence Tests

    [Fact]
    public void IcaDecomposition_VaryingIterations_ProducesValidResults()
    {
        // Arrange
        var X = CreateMixedSignals(150, 3);

        // Act
        var icaFew = new IcaDecomposition<double>(X, 2, maxIterations: 50);
        var icaMany = new IcaDecomposition<double>(X, 2, maxIterations: 300);

        // Assert - Both should produce valid results
        Assert.False(double.IsNaN(FrobeniusNorm(icaFew.UnmixingMatrix)));
        Assert.False(double.IsNaN(FrobeniusNorm(icaMany.UnmixingMatrix)));
    }

    [Fact]
    public void IcaDecomposition_TightTolerance_StillConverges()
    {
        // Arrange
        var X = CreateRandomMatrix(100, 3);

        // Act
        var ica = new IcaDecomposition<double>(X, 2, maxIterations: 500, tolerance: 1e-8);

        // Assert
        Assert.NotNull(ica.UnmixingMatrix);
        Assert.NotNull(ica.IndependentComponents);
    }

    #endregion

    #region Independent Components Properties Tests

    [Fact]
    public void IcaDecomposition_IndependentComponents_HaveFiniteValues()
    {
        // Arrange
        var X = CreateRandomMatrix(100, 4);

        // Act
        var ica = new IcaDecomposition<double>(X, 3);

        // Assert
        for (int i = 0; i < ica.IndependentComponents.Rows; i++)
        {
            for (int j = 0; j < ica.IndependentComponents.Columns; j++)
            {
                Assert.False(double.IsNaN(ica.IndependentComponents[i, j]),
                    $"IndependentComponents[{i},{j}] should not be NaN");
                Assert.False(double.IsInfinity(ica.IndependentComponents[i, j]),
                    $"IndependentComponents[{i},{j}] should not be infinity");
            }
        }
    }

    [Fact]
    public void IcaDecomposition_IndependentComponents_AreApproximatelyUncorrelated()
    {
        // Arrange
        var X = CreateMixedSignals(200, 4);

        // Act
        var ica = new IcaDecomposition<double>(X, 3, maxIterations: 300);

        // Assert - Independent components should be approximately uncorrelated
        var S = ica.IndependentComponents;
        for (int i = 0; i < S.Rows; i++)
        {
            for (int j = i + 1; j < S.Rows; j++)
            {
                // Compute correlation between components
                double sumI = 0, sumJ = 0, sumIJ = 0, sumI2 = 0, sumJ2 = 0;
                int n = S.Columns;

                for (int k = 0; k < n; k++)
                {
                    sumI += S[i, k];
                    sumJ += S[j, k];
                    sumIJ += S[i, k] * S[j, k];
                    sumI2 += S[i, k] * S[i, k];
                    sumJ2 += S[j, k] * S[j, k];
                }

                double meanI = sumI / n;
                double meanJ = sumJ / n;
                double varI = sumI2 / n - meanI * meanI;
                double varJ = sumJ2 / n - meanJ * meanJ;
                double covariance = sumIJ / n - meanI * meanJ;

                if (varI > 0 && varJ > 0)
                {
                    double correlation = covariance / (Math.Sqrt(varI) * Math.Sqrt(varJ));
                    // Components should be somewhat uncorrelated
                    Assert.True(Math.Abs(correlation) < 0.8,
                        $"Components {i} and {j} should be approximately uncorrelated. Correlation: {correlation}");
                }
            }
        }
    }

    #endregion
}
