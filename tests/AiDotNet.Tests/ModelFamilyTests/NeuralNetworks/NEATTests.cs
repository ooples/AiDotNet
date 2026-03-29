using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class NEATTests : NeuralNetworkModelTestBase
{
    // Default NEAT: inputSize=10, outputSize=1
    // Must use 2D shapes [batch, features] because NEAT.Train and ExtractTrainingData
    // require input.Shape[0] (batch) and input.Shape[1] (features)
    protected override int[] InputShape => [1, 10];
    protected override int[] OutputShape => [1, 1];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new NEAT<double>();
}
