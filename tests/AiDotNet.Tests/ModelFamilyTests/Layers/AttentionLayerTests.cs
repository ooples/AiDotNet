using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class AttentionLayerInvariantTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new AttentionLayer<double>(inputSize: 4, attentionSize: 8,
            activation: new TanhActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [1, 4];
}

public class SelfAttentionLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new SelfAttentionLayer<double>(sequenceLength: 2, embeddingDimension: 4, headCount: 2,
            activationFunction: new ReLUActivation<double>() as IActivationFunction<double>);
    // Self-attention expects [batch, seq, embed]
    protected override int[] InputShape => [1, 2, 4];
}

public class MultiHeadAttentionLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new MultiHeadAttentionLayer<double>(sequenceLength: 2, embeddingDimension: 4, headCount: 2,
            activationFunction: new ReLUActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [1, 2, 4];
}
