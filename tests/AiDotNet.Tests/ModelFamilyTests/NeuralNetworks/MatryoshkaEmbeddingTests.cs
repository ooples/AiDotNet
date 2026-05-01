using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class MatryoshkaEmbeddingTests : EmbeddingModelTestBase
{
    // MatryoshkaEmbedding (Kusupati et al. 2022 §3) emits a maxEmbeddingDimension-
    // wide vector at full resolution; nested sub-vectors of decreasing length
    // share the same forward pass. The parameterless ctor uses paper defaults
    // (12 transformer layers × 1536 hidden × 3072 FFN, roughly BERT-Large),
    // so a fresh network produces [1, 1536] outputs and a single Predict pass
    // is non-trivial on CI. Iteration counts are reduced from the test base
    // defaults (10, 50, 200) to keep individual tests inside the 120s
    // [Fact(Timeout)] budget while still exercising the invariants.
    protected override int[] InputShape => [1, 4];
    protected override int[] OutputShape => [1, 1536];

    protected override int TrainingIterations => 2;
    protected override int MoreDataShortIterations => 4;
    protected override int MoreDataLongIterations => 8;

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new MatryoshkaEmbedding<double>();
}
