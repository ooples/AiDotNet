using AiDotNet.PhysicsInformed.NeuralOperators;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PhysicsInformed.NeuralOperators;

/// <summary>
/// FourierLayer's spectral convolution runs its transform as dense DFT matmuls. The split-complex Engine.FFT family
/// records nothing on the gradient tape, so the spectral weights it fed never received a gradient and FNO never
/// trained them.
/// </summary>
public class FourierLayerSpectralTransformTests
{
    private static Tensor<double> Random(int[] shape, int seed)
    {
        var rng = new System.Random(seed);
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    [Theory]
    [InlineData(4)]
    [InlineData(8)]
    [InlineData(16)]
    public void Dft_EqualsEngineFft_ForwardAndInverse(int n)
    {
        var layer = new FourierLayer<double>(width: 2, modes: 2, spatialDimensions: new[] { n });
        var re = Random([1, 2, n], 1);
        var im = Random([1, 2, n], 2);

        var (dftRe, dftIm) = layer.ApplySeparableFft(re, im, inverse: false);
        AiDotNetEngine.Current.FFT(re, im, out var fftRe, out var fftIm);
        var (idftRe, idftIm) = layer.ApplySeparableFft(re, im, inverse: true);
        AiDotNetEngine.Current.IFFT(re, im, out var ifftRe, out var ifftIm);

        for (int i = 0; i < re.Length; i++)
        {
            Assert.Equal(fftRe[i], dftRe[i], 10);
            Assert.Equal(fftIm[i], dftIm[i], 10);
            Assert.Equal(ifftRe[i], idftRe[i], 10);
            Assert.Equal(ifftIm[i], idftIm[i], 10);
        }
    }

    [Theory]
    [InlineData(new[] { 8 })]
    [InlineData(new[] { 4, 4 })]
    public void SpectralWeights_ReceiveAGradient(int[] spatial)
    {
        var layer = new FourierLayer<double>(width: 2, modes: 2, spatialDimensions: spatial);
        var shape = new int[spatial.Length + 2];
        shape[0] = 1;
        shape[1] = 2;
        Array.Copy(spatial, 0, shape, 2, spatial.Length);
        var x = Random(shape, 3);
        var parameters = layer.GetTrainableParameters().ToArray();

        Dictionary<Tensor<double>, Tensor<double>> gradients;
        using (var tape = new GradientTape<double>())
        {
            var y = layer.Forward(x);
            var loss = AiDotNetEngine.Current.ReduceSum(AiDotNetEngine.Current.TensorMultiply(y, y), null);
            gradients = tape.ComputeGradients(loss, parameters);
        }

        // [spectral real, spectral imaginary, pointwise weights, pointwise bias]: all four must move.
        Assert.Equal(4, parameters.Length);
        foreach (var parameter in parameters)
        {
            Assert.True(gradients.TryGetValue(parameter, out var gradient), "a Fourier layer weight got no gradient.");
            Assert.Contains(gradient.ToArray(), v => v != 0.0);
        }
    }
}