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
}
