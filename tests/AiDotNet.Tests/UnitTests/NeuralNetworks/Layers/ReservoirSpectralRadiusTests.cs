using System;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// The echo state network scales its reservoir by the spectral radius max_i |lambda_i| (Jaeger 2001), where the
/// eigenvalues of the non-symmetric reservoir matrix are complex in general. Reference values are
/// numpy.linalg.eigvals moduli.
/// </summary>
public class ReservoirSpectralRadiusTests
{
    private const int Precision = 10;

    [Fact]
    public void ComputeMaxEigenvalue_ScaledRotation_ReturnsComplexEigenvalueModulus()
    {
        var layer = new ReservoirLayer<double>(inputSize: 1, reservoirSize: 2);

        // [[0, -2], [2, 0]] has eigenvalues +/-2i, so its spectral radius is 2. A Rayleigh-quotient power
        // iteration returns 0 here because v . (A v) = 0 for every v.
        var matrix = new Tensor<double>(new[] { 0.0, -2.0, 2.0, 0.0 }, new[] { 2, 2 });

        Assert.Equal(2.0, layer.ComputeMaxEigenvalue(matrix), Precision);
    }

    [Fact]
    public void ComputeMaxEigenvalue_RealEigenvalues_ReturnsLargestMagnitude()
    {
        var layer = new ReservoirLayer<double>(inputSize: 1, reservoirSize: 2);

        // Upper-triangular, eigenvalues 3 and -5: spectral radius 5.
        var matrix = new Tensor<double>(new[] { 3.0, 1.0, 0.0, -5.0 }, new[] { 2, 2 });

        Assert.Equal(5.0, layer.ComputeMaxEigenvalue(matrix), Precision);
    }

    // numpy.linalg.eigvals moduli of RandomReservoir(size, density, seed), dumped from .NET and read back with
    // np.loadtxt. AiDotNet.Tensors 0.152.1 Linalg.Eig, which this layer called before, returns 0.9640, 0.9200,
    // 2.4715 and 0.6683 for these four: a wrong spectral radius, so the reservoir was scaled to the wrong radius.
    [Theory]
    [InlineData(100, 0.1, 1, 0.987483918802371)]
    [InlineData(37, 0.3, 4, 0.9486365196746817)]
    [InlineData(100, 1.0, 3, 2.9365774244347898)]
    [InlineData(200, 0.05, 5, 1.053266845019182)]
    public void ComputeMaxEigenvalue_RandomReservoir_MatchesNumpy(int size, double density, int seed, double expected)
    {
        var layer = new ReservoirLayer<double>(inputSize: 1, reservoirSize: 2);
        Assert.Equal(expected, layer.ComputeMaxEigenvalue(RandomReservoir(size, density, seed)), 10);
    }

    [SkippableFact]
    public void ComputeMaxEigenvalue_DefaultReservoir_AllocatesOnlyItsWorkingCopy()
    {
#if !NETCOREAPP3_0_OR_GREATER
        // net471 has no per-thread allocation counter, and a process-wide one counts parallel tests too.
        Skip.If(true, "GC.GetAllocatedBytesForCurrentThread needs .NET Core 3.0 or later.");
#else
        // The census shape: LiquidStateMachine's 100-neuron reservoir. The full Linalg.Eig allocated ~98 MB and took
        // ~170 ms here, 4.7x LiquidStateMachine's construction time; the in-place QR needs one n x n copy.
        var layer = new ReservoirLayer<double>(inputSize: 1, reservoirSize: 2);
        var matrix = RandomReservoir(100, 0.1, 7);
        layer.ComputeMaxEigenvalue(matrix);

        long before = GC.GetAllocatedBytesForCurrentThread();
        layer.ComputeMaxEigenvalue(matrix);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.True(allocated < 1_000_000, $"spectral radius of a 100 x 100 reservoir allocated {allocated:N0} bytes");
#endif
    }

    private static Tensor<double> RandomReservoir(int size, double density, int seed)
    {
        var random = new Random(seed);
        var values = new double[size * size];
        for (int i = 0; i < values.Length; i++)
            values[i] = random.NextDouble() < density ? random.NextDouble() - 0.5 : 0.0;
        return new Tensor<double>(values, new[] { size, size });
    }
}
