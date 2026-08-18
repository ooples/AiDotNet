using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

public sealed class Conv3DPackageIntegrationTests : IDisposable
{
    private readonly IEngine _priorEngine = AiDotNetEngine.Current;

    public Conv3DPackageIntegrationTests() => AiDotNetEngine.Current = new CpuEngine();

    public void Dispose() => AiDotNetEngine.Current = _priorEngine;

    [Fact]
    public void TrainingForward_MatchesPackageConv3DAndConnectsAllGradients()
    {
        var engine = AiDotNetEngine.Current;
        var layer = Conv3DLayer<double>.WithInputChannels(
            inputChannels: 2,
            outputChannels: 3,
            kernelSize: 3,
            padding: 1,
            activationFunction: (IActivationFunction<double>)new IdentityActivation<double>());
        layer.SetTrainingMode(true);

        var parameters = layer.GetTrainableParameters();
        Assert.Equal(2, parameters.Count);
        var kernel = parameters[0];
        var bias = parameters[1];
        for (int i = 0; i < kernel.Length; i++)
            kernel[i] = -0.08 + (i * 0.001);
        for (int i = 0; i < bias.Length; i++)
            bias[i] = 0.02 * (i + 1);

        var input = new Tensor<double>([1, 2, 4, 4, 4]);
        for (int i = 0; i < input.Length; i++)
            input[i] = -0.2 + (i * 0.004);

        Tensor<double> expected;
        using (GradientTape<double>.NoGrad())
        {
            var convolution = engine.Conv3D(input, kernel, [1, 1, 1], [1, 1, 1], [1, 1, 1]);
            expected = engine.TensorBroadcastAdd(
                convolution,
                engine.Reshape(bias, [1, 3, 1, 1, 1]));
        }

        using var tape = new GradientTape<double>();
        var actual = layer.Forward(input);
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        for (int i = 0; i < actual.Length; i++)
            Assert.Equal(expected[i], actual[i], 10);

        var outputWeight = new Tensor<double>(actual.Shape.ToArray());
        for (int i = 0; i < outputWeight.Length; i++)
            outputWeight[i] = 0.1 + (i * 0.0005);
        var loss = engine.ReduceSum(engine.TensorMultiply(actual, outputWeight), null);
        var gradients = tape.ComputeGradients(loss, [input, kernel, bias]);

        foreach (var source in new[] { input, kernel, bias })
        {
            Assert.True(gradients.TryGetValue(source, out var gradient));
            Assert.NotNull(gradient);
            Assert.Contains(gradient!.AsSpan().ToArray(), value => Math.Abs(value) > 1e-12);
            Assert.All(gradient.AsSpan().ToArray(), value =>
                Assert.True(!double.IsNaN(value) && !double.IsInfinity(value)));
        }
    }
}
