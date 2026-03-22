using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class PaddingLayerTests : LayerTestBase
{
    // PaddingLayer: Forward expects padding.Length == input.Shape.Length (BHWC)
    // padding = [batch_pad, height_pad, width_pad, channel_pad]
    protected override ILayer<double> CreateLayer()
        => new PaddingLayer<double>(inputShape: [1, 4, 4, 1], padding: [0, 1, 1, 0],
            activationFunction: new IdentityActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [1, 4, 4, 1]; // BHWC
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}

public class TimeEmbeddingLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new TimeEmbeddingLayer<double>(embeddingDim: 8, outputDim: 8, maxTimestep: 100);
    protected override int[] InputShape => [1, 1];
}

public class RotaryPositionalEncodingLayerTests : LayerTestBase
{
    // RoPE per Su et al. 2021: input [..., seqLen, headDim], headDim must be even
    protected override ILayer<double> CreateLayer()
        => new RotaryPositionalEncodingLayer<double>(maxSequenceLength: 8, headDimension: 4);
    // Last two dims: [seqLen <= maxSeqLen, headDim matching constructor]
    protected override int[] InputShape => [1, 4, 4]; // [batch, seqLen=4, headDim=4]
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}

public class DecoderLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new DecoderLayer<double>(inputSize: 8, attentionSize: 8, feedForwardSize: 16,
            activation: new ReLUActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [1, 4, 8];
}

public class TransformerDecoderLayerTests : LayerTestBase
{
    // Per Vaswani et al. 2017: decoder-only mode uses self-attention on input
    protected override ILayer<double> CreateLayer()
        => new TransformerDecoderLayer<double>(embeddingSize: 8, numHeads: 2, feedForwardDim: 16, sequenceLength: 4,
            ffnActivation: new ReLUActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [1, 4, 8]; // [batch, seq, embed]
}

public class DigitCapsuleLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new DigitCapsuleLayer<double>(
            inputCapsules: 4, inputCapsuleDimension: 2, numClasses: 3,
            outputCapsuleDimension: 4, routingIterations: 3);
    protected override int[] InputShape => [1, 4, 2];
}

public class SeparableConvolutionalLayerTests : LayerTestBase
{
    // SeparableConv uses NHWC format: inputShape = [batch, H, W, C]
    protected override ILayer<double> CreateLayer()
        => new SeparableConvolutionalLayer<double>(
            inputShape: [1, 8, 8, 2], outputDepth: 4, kernelSize: 3,
            scalarActivation: new ReLUActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [1, 8, 8, 2]; // NHWC
}

public class GraphAttentionLayerTests2 : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
    {
        var layer = new GraphAttentionLayer<double>(inputFeatures: 4, outputFeatures: 8, numHeads: 2);
        var adj = new Tensor<double>([1, 2, 2]);
        adj[0, 0, 0] = 1; adj[0, 0, 1] = 1; adj[0, 1, 0] = 1; adj[0, 1, 1] = 1;
        layer.SetAdjacencyMatrix(adj);
        return layer;
    }
    protected override int[] InputShape => [1, 2, 4];
}

public class GraphIsomorphismLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
    {
        var layer = new GraphIsomorphismLayer<double>(inputFeatures: 4, outputFeatures: 8,
            activationFunction: new ReLUActivation<double>() as IActivationFunction<double>);
        var adj = new Tensor<double>([1, 2, 2]);
        adj[0, 0, 0] = 1; adj[0, 0, 1] = 1; adj[0, 1, 0] = 1; adj[0, 1, 1] = 1;
        layer.SetAdjacencyMatrix(adj);
        return layer;
    }
    protected override int[] InputShape => [1, 2, 4];
}

public class GraphSAGELayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
    {
        var layer = new GraphSAGELayer<double>(inputFeatures: 4, outputFeatures: 8,
            activationFunction: new ReLUActivation<double>() as IActivationFunction<double>);
        var adj = new Tensor<double>([1, 2, 2]);
        adj[0, 0, 0] = 1; adj[0, 0, 1] = 1; adj[0, 1, 0] = 1; adj[0, 1, 1] = 1;
        layer.SetAdjacencyMatrix(adj);
        return layer;
    }
    protected override int[] InputShape => [1, 2, 4];
}
