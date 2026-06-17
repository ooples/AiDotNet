using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class SPLADETests : NeuralNetworkModelTestBase<float>
{
    // SPLADE produces a sparse vocab-sized [VocabSize=30522] activation
    // vector (BERT vocabulary), not the architecture.OutputSize=768 it
    // advertises. Test base shape must match the actual prediction shape.
    protected override int[] InputShape => [1];
    protected override int[] OutputShape => [30522];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new SPLADE<float>();
}
