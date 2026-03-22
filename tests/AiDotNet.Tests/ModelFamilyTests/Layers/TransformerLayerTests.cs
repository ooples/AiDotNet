using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class TransformerEncoderLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new TransformerEncoderLayer<double>(embeddingSize: 8, numHeads: 2, feedForwardDim: 16);
    // Transformer expects [batch, seq, embed]
    protected override int[] InputShape => [1, 4, 8];
}

public class CrossAttentionLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new CrossAttentionLayer<double>(queryDim: 8, contextDim: 8, headCount: 2, sequenceLength: 4);
    protected override int[] InputShape => [1, 4, 8];
}
