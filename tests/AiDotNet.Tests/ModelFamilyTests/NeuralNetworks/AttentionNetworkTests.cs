using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class AttentionNetworkTests : NeuralNetworkModelTestBase<float>
{
    // AttentionNetwork takes ALREADY-EMBEDDED sequences: (sequenceLength, embeddingSize), which for
    // the parameterless constructor is 32 elements of 64 features, projected to 128 outputs. The
    // previous [128] declared here was a flat vector of the model's OUTPUT width; it only survived
    // because the default layer stack contained a spurious embedding lookup that reinterpreted those
    // 128 numbers as token indices. With the lookup gone the model's own declaration is correct, so
    // this fixture takes it rather than restating it.
    protected override int[] InputShape => [32, 64];
    protected override int[] OutputShape => [128];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new AttentionNetwork<float>();
}
