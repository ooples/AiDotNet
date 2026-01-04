using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.NumericOperations;
using System.Diagnostics;
using Xunit;

namespace AiDotNet.Tests.Benchmarks;

/// <summary>
/// Performance regression tests for SIMD-optimized activation functions.
/// These tests verify that the vectorized implementations are correct and
/// provide basic performance sanity checks.
/// </summary>
public class SimdActivationFunctionBenchmarks
{
    private const int SmallSize = 128;
    private const int MediumSize = 4096;
    private const int LargeSize = 65536;
    private const int WarmupIterations = 3;
    private const int BenchmarkIterations = 10;

    private readonly FloatOperations _floatOps = new FloatOperations();
    private readonly DoubleOperations _doubleOps = new DoubleOperations();

    #region Correctness Tests

    [Fact]
    public void ReLU_Float_ProducesCorrectOutput()
    {
        // Arrange
        var input = new float[] { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 0.5f };
        var expected = new float[] { 0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 0.0f, 0.5f };
        var output = new float[input.Length];

        // Act
        _floatOps.ReLU(input, output);

        // Assert
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], output[i], precision: 5);
        }
    }

    [Fact]
    public void ReLU_Double_ProducesCorrectOutput()
    {
        // Arrange
        var input = new double[] { -2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 0.5 };
        var expected = new double[] { 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.5 };
        var output = new double[input.Length];

        // Act
        _doubleOps.ReLU(input, output);

        // Assert
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], output[i], precision: 10);
        }
    }

    [Fact]
    public void LeakyReLU_Float_ProducesCorrectOutput()
    {
        // Arrange
        float alpha = 0.01f;
        var input = new float[] { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f };
        var expected = new float[] { -0.02f, -0.01f, 0.0f, 1.0f, 2.0f };
        var output = new float[input.Length];

        // Act
        _floatOps.LeakyReLU(input, alpha, output);

        // Assert
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], output[i], precision: 5);
        }
    }

    [Fact]
    public void LeakyReLU_Double_ProducesCorrectOutput()
    {
        // Arrange
        double alpha = 0.01;
        var input = new double[] { -2.0, -1.0, 0.0, 1.0, 2.0 };
        var expected = new double[] { -0.02, -0.01, 0.0, 1.0, 2.0 };
        var output = new double[input.Length];

        // Act
        _doubleOps.LeakyReLU(input, alpha, output);

        // Assert
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], output[i], precision: 10);
        }
    }

    [Fact]
    public void ELU_Float_ProducesCorrectOutput()
    {
        // Arrange
        float alpha = 1.0f;
        var input = new float[] { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f };
        var output = new float[input.Length];

        // Act
        _floatOps.ELU(input, alpha, output);

        // Assert
        // ELU(x) = x if x > 0, else alpha * (exp(x) - 1)
        Assert.True(output[0] < 0); // ELU(-2) < 0
        Assert.True(output[1] < 0); // ELU(-1) < 0
        Assert.Equal(0.0f, output[2], precision: 5); // ELU(0) = 0
        Assert.Equal(1.0f, output[3], precision: 5); // ELU(1) = 1
        Assert.Equal(2.0f, output[4], precision: 5); // ELU(2) = 2
    }

    [Fact]
    public void GELU_Float_ProducesCorrectOutput()
    {
        // Arrange
        var input = new float[] { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f };
        var output = new float[input.Length];

        // Act
        _floatOps.GELU(input, output);

        // Assert
        // GELU(0) should be 0
        Assert.Equal(0.0f, output[2], precision: 5);
        // GELU(x) > 0 for x > 0
        Assert.True(output[3] > 0);
        Assert.True(output[4] > 0);
        // GELU is roughly x for large positive x
        Assert.True(output[4] > 1.5f);
    }

    [Fact]
    public void Swish_Float_ProducesCorrectOutput()
    {
        // Arrange
        var input = new float[] { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f };
        var output = new float[input.Length];

        // Act
        _floatOps.Swish(input, output);

        // Assert
        // Swish(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        Assert.Equal(0.0f, output[2], precision: 5);
        // Swish(x) > 0 for x > 0
        Assert.True(output[3] > 0);
        Assert.True(output[4] > 0);
    }

    [Fact]
    public void Mish_Float_ProducesCorrectOutput()
    {
        // Arrange
        var input = new float[] { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f };
        var output = new float[input.Length];

        // Act
        _floatOps.Mish(input, output);

        // Assert
        // Mish(0) = 0 * tanh(softplus(0)) = 0 * tanh(ln(2)) ~ 0
        Assert.Equal(0.0f, output[2], precision: 5);
        // Mish(x) > 0 for x > 0
        Assert.True(output[3] > 0);
        Assert.True(output[4] > 0);
    }

    #endregion

    #region Large Array Correctness Tests

    [Theory]
    [InlineData(SmallSize)]
    [InlineData(MediumSize)]
    public void ReLU_Float_LargeArray_NoNaNOrInfinity(int size)
    {
        // Arrange
        var random = RandomHelper.CreateSeededRandom(42);
        var input = CreateRandomFloatArray(size, random, -10f, 10f);
        var output = new float[size];

        // Act
        _floatOps.ReLU(input, output);

        // Assert
        for (int i = 0; i < size; i++)
        {
            Assert.False(float.IsNaN(output[i]), $"NaN at index {i}");
            Assert.False(float.IsInfinity(output[i]), $"Infinity at index {i}");
            Assert.True(output[i] >= 0, $"ReLU output should be >= 0 at index {i}");
        }
    }

    [Theory]
    [InlineData(SmallSize)]
    [InlineData(MediumSize)]
    public void GELU_Float_LargeArray_NoNaNOrInfinity(int size)
    {
        // Arrange
        var random = RandomHelper.CreateSeededRandom(42);
        var input = CreateRandomFloatArray(size, random, -5f, 5f);
        var output = new float[size];

        // Act
        _floatOps.GELU(input, output);

        // Assert
        for (int i = 0; i < size; i++)
        {
            Assert.False(float.IsNaN(output[i]), $"NaN at index {i}");
            Assert.False(float.IsInfinity(output[i]), $"Infinity at index {i}");
        }
    }

    [Theory]
    [InlineData(SmallSize)]
    [InlineData(MediumSize)]
    public void Mish_Float_LargeArray_NoNaNOrInfinity(int size)
    {
        // Arrange
        var random = RandomHelper.CreateSeededRandom(42);
        var input = CreateRandomFloatArray(size, random, -5f, 5f);
        var output = new float[size];

        // Act
        _floatOps.Mish(input, output);

        // Assert
        for (int i = 0; i < size; i++)
        {
            Assert.False(float.IsNaN(output[i]), $"NaN at index {i}");
            Assert.False(float.IsInfinity(output[i]), $"Infinity at index {i}");
        }
    }

    [Theory]
    [InlineData(SmallSize)]
    [InlineData(MediumSize)]
    public void Swish_Float_LargeArray_NoNaNOrInfinity(int size)
    {
        // Arrange
        var random = RandomHelper.CreateSeededRandom(42);
        var input = CreateRandomFloatArray(size, random, -5f, 5f);
        var output = new float[size];

        // Act
        _floatOps.Swish(input, output);

        // Assert
        for (int i = 0; i < size; i++)
        {
            Assert.False(float.IsNaN(output[i]), $"NaN at index {i}");
            Assert.False(float.IsInfinity(output[i]), $"Infinity at index {i}");
        }
    }

    [Theory]
    [InlineData(SmallSize)]
    [InlineData(MediumSize)]
    public void ELU_Float_LargeArray_NoNaNOrInfinity(int size)
    {
        // Arrange
        var random = RandomHelper.CreateSeededRandom(42);
        var input = CreateRandomFloatArray(size, random, -5f, 5f);
        var output = new float[size];

        // Act
        _floatOps.ELU(input, 1.0f, output);

        // Assert
        for (int i = 0; i < size; i++)
        {
            Assert.False(float.IsNaN(output[i]), $"NaN at index {i}");
            Assert.False(float.IsInfinity(output[i]), $"Infinity at index {i}");
        }
    }

    #endregion

    #region Performance Sanity Checks

    [Fact]
    public void ReLU_Float_PerformanceSanityCheck()
    {
        // This test ensures the SIMD implementation completes in reasonable time
        // Arrange
        var random = RandomHelper.CreateSeededRandom(42);
        var input = CreateRandomFloatArray(LargeSize, random, -10f, 10f);
        var output = new float[LargeSize];

        // Warmup
        for (int i = 0; i < WarmupIterations; i++)
        {
            _floatOps.ReLU(input, output);
        }

        // Act
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < BenchmarkIterations; i++)
        {
            _floatOps.ReLU(input, output);
        }
        sw.Stop();

        // Assert - should complete 10 iterations of 64K elements in under 100ms
        var avgMs = sw.ElapsedMilliseconds / (double)BenchmarkIterations;
        Assert.True(avgMs < 10, $"ReLU took {avgMs}ms per iteration, expected < 10ms for {LargeSize} elements");
    }

    [Fact]
    public void GELU_Float_PerformanceSanityCheck()
    {
        // Arrange
        var random = RandomHelper.CreateSeededRandom(42);
        var input = CreateRandomFloatArray(LargeSize, random, -5f, 5f);
        var output = new float[LargeSize];

        // Warmup
        for (int i = 0; i < WarmupIterations; i++)
        {
            _floatOps.GELU(input, output);
        }

        // Act
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < BenchmarkIterations; i++)
        {
            _floatOps.GELU(input, output);
        }
        sw.Stop();

        // Assert - GELU is more expensive, allow up to 50ms per iteration
        var avgMs = sw.ElapsedMilliseconds / (double)BenchmarkIterations;
        Assert.True(avgMs < 50, $"GELU took {avgMs}ms per iteration, expected < 50ms for {LargeSize} elements");
    }

    [Fact]
    public void AllActivations_Float_CompletesWithoutError()
    {
        // This test ensures all activation functions work on various sizes
        var random = RandomHelper.CreateSeededRandom(42);

        int[] sizes = { 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025 };

        foreach (var size in sizes)
        {
            var input = CreateRandomFloatArray(size, random, -5f, 5f);
            var output = new float[size];

            // Should not throw
            _floatOps.ReLU(input, output);
            _floatOps.LeakyReLU(input, 0.01f, output);
            _floatOps.GELU(input, output);
            _floatOps.Mish(input, output);
            _floatOps.Swish(input, output);
            _floatOps.ELU(input, 1.0f, output);
        }
    }

    #endregion

    #region Double Precision Tests

    [Fact]
    public void AllActivations_Double_CompletesWithoutError()
    {
        // This test ensures all activation functions work on various sizes for double
        var random = RandomHelper.CreateSeededRandom(42);

        int[] sizes = { 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025 };

        foreach (var size in sizes)
        {
            var input = CreateRandomDoubleArray(size, random, -5.0, 5.0);
            var output = new double[size];

            // Should not throw
            _doubleOps.ReLU(input, output);
            _doubleOps.LeakyReLU(input, 0.01, output);
            _doubleOps.GELU(input, output);
            _doubleOps.Mish(input, output);
            _doubleOps.Swish(input, output);
            _doubleOps.ELU(input, 1.0, output);
        }
    }

    [Fact]
    public void ReLU_Double_PerformanceSanityCheck()
    {
        // Arrange
        var random = RandomHelper.CreateSeededRandom(42);
        var input = CreateRandomDoubleArray(LargeSize, random, -10.0, 10.0);
        var output = new double[LargeSize];

        // Warmup
        for (int i = 0; i < WarmupIterations; i++)
        {
            _doubleOps.ReLU(input, output);
        }

        // Act
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < BenchmarkIterations; i++)
        {
            _doubleOps.ReLU(input, output);
        }
        sw.Stop();

        // Assert
        var avgMs = sw.ElapsedMilliseconds / (double)BenchmarkIterations;
        Assert.True(avgMs < 20, $"ReLU (double) took {avgMs}ms per iteration, expected < 20ms for {LargeSize} elements");
    }

    #endregion

    #region Helper Methods

    private static float[] CreateRandomFloatArray(int size, Random random, float min, float max)
    {
        var array = new float[size];
        float range = max - min;
        for (int i = 0; i < size; i++)
        {
            array[i] = (float)(random.NextDouble() * range + min);
        }
        return array;
    }

    private static double[] CreateRandomDoubleArray(int size, Random random, double min, double max)
    {
        var array = new double[size];
        double range = max - min;
        for (int i = 0; i < size; i++)
        {
            array[i] = random.NextDouble() * range + min;
        }
        return array;
    }

    #endregion
}
