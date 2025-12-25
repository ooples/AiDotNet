using AiDotNet.ActivationFunctions;
using AiDotNet.PhysicsInformed.NeuralOperators;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PhysicsInformed.NeuralOperators;

public class FourierLayerTests
{
    [Fact]
    public void FourierLayer_IdentityWeightsPreserveConstantInput()
    {
        var layer = new FourierLayer<double>(
            width: 1,
            modes: 1,
            spatialDimensions: new[] { 4 },
            activation: new IdentityActivation<double>());

        var spectral = new Tensor<Complex<double>>(new[] { 1, 1, 1 });
        spectral[0, 0, 0] = new Complex<double>(1.0, 0.0);

        var parameters = new Vector<double>(layer.ParameterCount);
        int index = 0;
        for (int i = 0; i < spectral.Length; i++)
        {
            parameters[index++] = spectral[i].Real;
        }
        for (int i = 0; i < spectral.Length; i++)
        {
            parameters[index++] = spectral[i].Imaginary;
        }
        parameters[index++] = 0.0;
        parameters[index++] = 0.0;
        layer.SetParameters(parameters);

        var input = new Tensor<double>(new[] { 1, 1, 4 });
        for (int s = 0; s < 4; s++)
        {
            input[0, 0, s] = 3.5;
        }

        var output = layer.Forward(input);
        for (int s = 0; s < 4; s++)
        {
            Assert.Equal(3.5, output[0, 0, s], 6);
        }
    }

    [Fact]
    public void FourierLayer_ChannelMixingSwapsConstantChannels()
    {
        var layer = new FourierLayer<double>(
            width: 2,
            modes: 1,
            spatialDimensions: new[] { 4 },
            activation: new IdentityActivation<double>());

        var spectral = new Tensor<Complex<double>>(new[] { 2, 2, 1 });
        spectral[0, 1, 0] = new Complex<double>(1.0, 0.0);
        spectral[1, 0, 0] = new Complex<double>(1.0, 0.0);

        var parameters = new Vector<double>(layer.ParameterCount);
        int index = 0;
        for (int i = 0; i < spectral.Length; i++)
        {
            parameters[index++] = spectral[i].Real;
        }
        for (int i = 0; i < spectral.Length; i++)
        {
            parameters[index++] = spectral[i].Imaginary;
        }
        for (int i = 0; i < 4; i++)
        {
            parameters[index++] = 0.0;
        }
        for (int i = 0; i < 2; i++)
        {
            parameters[index++] = 0.0;
        }
        layer.SetParameters(parameters);

        var input = new Tensor<double>(new[] { 1, 2, 4 });
        for (int s = 0; s < 4; s++)
        {
            input[0, 0, s] = 1.0;
            input[0, 1, s] = 2.0;
        }

        var output = layer.Forward(input);
        for (int s = 0; s < 4; s++)
        {
            Assert.Equal(2.0, output[0, 0, s], 6);
            Assert.Equal(1.0, output[0, 1, s], 6);
        }
    }
}
