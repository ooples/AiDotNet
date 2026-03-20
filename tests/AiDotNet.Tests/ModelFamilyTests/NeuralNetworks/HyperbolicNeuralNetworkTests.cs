using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class HyperbolicNeuralNetworkTests : NeuralNetworkModelTestBase
{
    // Use small input dimension to keep vectors inside the Poincaré ball
    // (high-dim random vectors exceed ball radius and get clamped identically)
    protected override int[] InputShape => [8];
    protected override int[] OutputShape => [1];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new HyperbolicNeuralNetwork<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: AiDotNet.Enums.InputType.OneDimensional,
                taskType: AiDotNet.Enums.NeuralNetworkTaskType.Regression,
                inputSize: 8,
                outputSize: 1),
            curvature: -0.01); // Small curvature = large ball = less projection
}
