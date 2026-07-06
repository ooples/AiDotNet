using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class SiameseNeuralNetworkTests : NeuralNetworkModelTestBase<float>
{
    // #1706: 768-dim twin-encoder forward/backward fits its timeout in isolation (~25s) but times
    // out under parallel-shard core contention with single-threaded determinism BLAS — serialize it.
    protected override bool RequiresHeavySerialization => true;

    // SiameseNN default: inputSize=768, outputSize=768
    protected override int[] InputShape => [768];
    protected override int[] OutputShape => [768];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new SiameseNeuralNetwork<float>();
}
