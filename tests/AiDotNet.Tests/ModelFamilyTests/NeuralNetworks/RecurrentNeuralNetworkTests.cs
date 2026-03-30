using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class RecurrentNeuralNetworkTests : NeuralNetworkModelTestBase
{
    private readonly ITestOutputHelper _output;

    public RecurrentNeuralNetworkTests(ITestOutputHelper output)
    {
        _output = output;
    }

    // RNNs process sequences: [seqLen, features]. Default arch has inputSize=128.
    // Using 4 timesteps with 128 features tests actual recurrent behavior.
    protected override int[] InputShape => [4, 128];
    protected override int[] OutputShape => [1];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new RecurrentNeuralNetwork<double>();

}
