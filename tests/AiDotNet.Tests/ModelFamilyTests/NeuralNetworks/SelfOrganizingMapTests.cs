using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class SelfOrganizingMapTests : NeuralNetworkModelTestBase
{
    // SOM with outputSize=64 adjusts to 10x6=60 neurons for golden ratio aspect
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [60];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new SelfOrganizingMap<double>();

    /// <summary>
    /// SOM uses competitive learning with one-hot BMU output.
    /// MSE tolerance is higher because output depends on which neuron wins.
    /// </summary>
    protected override double MoreDataTolerance => 0.05;
}
