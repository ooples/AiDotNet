using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

// Its full finite-difference coverage takes ~2-3 s alone but timed out at 120 s under shard-wide
// CPU contention. Dedicated-core execution preserves the sample count, tolerance, and timeout.
[Xunit.Collection("FoundationScaleSerial")]
public class ResidualNeuralNetworkTests : NeuralNetworkModelTestBase<float>
{
    protected override int[] InputShape => [128];
    protected override int[] OutputShape => [1];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new ResidualNeuralNetwork<float>();
}
