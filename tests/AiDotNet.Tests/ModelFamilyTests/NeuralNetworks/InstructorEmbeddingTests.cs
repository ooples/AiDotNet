using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class InstructorEmbeddingTests : EmbeddingModelTestBase
{
    protected override INeuralNetworkModel<double> CreateNetwork()
        => new InstructorEmbedding<double>();
}
