using System.IO;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Cloning;

/// <summary>
/// DenseLayer with <see cref="BiasMode.Never"/> is nn.Linear(bias=False): it computes x W, and the bias
/// is absent from parameters, clones and checkpoints rather than frozen at zero.
/// </summary>
public class DenseLayerBiasModeTests
{
    private static Tensor<double> Input()
    {
        var x = new Tensor<double>([3, 5]);
        for (int i = 0; i < x.Length; i++) x[i] = i * 0.17 - 1.1;
        return x;
    }

    [Fact(Timeout = 120000)]
    public async Task BiasModeNever_ComputesExactlyXW()
    {
        await Task.Yield();
        var layer = new DenseLayer<double>(outputSize: 4, activationFunction: null, biasMode: BiasMode.Never);
        var x = Input();
        var y = layer.Forward(x);
        var w = layer.GetWeights();

        Assert.False(layer.UseBias);
        for (int r = 0; r < 3; r++)
        {
            for (int c = 0; c < 4; c++)
            {
                double expected = 0;
                for (int k = 0; k < 5; k++) expected += x[r * 5 + k] * w[k * 4 + c];
                Assert.Equal(expected, y[r * 4 + c], precision: 12);
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task BiasModeNever_LeavesTheBiasOutOfEveryParameterSurface()
    {
        await Task.Yield();
        var withoutBias = new DenseLayer<double>(outputSize: 4, activationFunction: null, biasMode: BiasMode.Never);
        var withBias = new DenseLayer<double>(outputSize: 4, activationFunction: null);
        _ = withoutBias.Forward(Input());
        _ = withBias.Forward(Input());

        Assert.True(withBias.UseBias);
        Assert.Equal(20, withoutBias.ParameterCount);
        Assert.Equal(24, withBias.ParameterCount);
        Assert.Equal(20, withoutBias.GetParameters().Length);
    }

    [Fact(Timeout = 120000)]
    public async Task BiasModeNever_SurvivesCloneAndCheckpoint()
    {
        await Task.Yield();
        var source = new DenseLayer<double>(outputSize: 4, activationFunction: null, biasMode: BiasMode.Never);
        var x = Input();
        var expected = source.Forward(x);

        var clone = (DenseLayer<double>)source.Clone();
        var cloned = clone.Forward(x);
        Assert.False(clone.UseBias);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], cloned[i], precision: 12);

        var restored = new DenseLayer<double>(outputSize: 4, activationFunction: null, biasMode: BiasMode.Never);
        _ = restored.Forward(x);
        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
            source.Serialize(writer);
        stream.Position = 0;
        using (var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true))
            restored.Deserialize(reader);
        var actual = restored.Forward(x);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], precision: 12);
    }
}
