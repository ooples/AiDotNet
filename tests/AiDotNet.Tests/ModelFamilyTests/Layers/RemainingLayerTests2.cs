using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class ALiBiPositionalBiasLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new ALiBiPositionalBiasLayer<double>(numHeads: 2, maxSequenceLength: 8);
    protected override int[] InputShape => [2, 4, 4]; // [numHeads, seqLen, seqLen]
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}

public class SubpixelConvolutionalLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new SubpixelConvolutionalLayer<double>(inputDepth: 4, outputDepth: 1, upscaleFactor: 2,
            kernelSize: 3, inputHeight: 4, inputWidth: 4,
            activation: new ReLUActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [1, 4, 4, 4]; // [batch, C, H, W]
}

public class ReadoutLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new ReadoutLayer<double>(inputSize: 4, outputSize: 8,
            scalarActivation: new ReLUActivation<double>());
    protected override int[] InputShape => [1, 4];
}

public class RBFLayerInvariantTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new RBFLayer<double>(inputSize: 4, outputSize: 8, rbf: new AiDotNet.RadialBasisFunctions.GaussianRBF<double>());
    protected override int[] InputShape => [4];
}

public class GroupedQueryAttentionLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new GroupedQueryAttentionLayer<double>(sequenceLength: 4, embeddingDimension: 8,
            numHeads: 4, numKVHeads: 2);
    protected override int[] InputShape => [1, 4, 8]; // [batch, seq, embed]
}

public class SwinPatchMergingLayerTests : LayerTestBase
{
    // SwinPatchMerging expects [batch, seqLen, dim] where seqLen = H*W, H and W even
    protected override ILayer<double> CreateLayer()
        => new SwinPatchMergingLayer<double>(inputDim: 4);
    protected override int[] InputShape => [1, 4, 4]; // [batch, seqLen=4 (2x2), dim=4]
}

public class ResidualDenseBlockTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new ResidualDenseBlock<double>(numFeatures: 4, growthChannels: 4,
            inputHeight: 4, inputWidth: 4);
    protected override int[] InputShape => [1, 4, 4, 4];
}

public class RRDBLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new RRDBLayer<double>(numFeatures: 4, growthChannels: 4,
            inputHeight: 4, inputWidth: 4);
    protected override int[] InputShape => [1, 4, 4, 4];
}
