using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class CroppingLayerTests : LayerTestBase
{
    // CroppingLayer: inputShape=[H,W,C], crop arrays per dim
    // output[i] = input[i] - top[i] - bottom[i] - left[i] - right[i]
    protected override ILayer<double> CreateLayer()
        => new CroppingLayer<double>(inputShape: [8, 8, 1],
            cropTop: [1, 1, 0], cropBottom: [1, 1, 0], cropLeft: [0, 0, 0], cropRight: [0, 0, 0],
            scalarActivation: new IdentityActivation<double>() as IActivationFunction<double>);
    // Output shape: [8-1-1, 8-1-1, 1] = [6, 6, 1]. Forward adds batch → [1, 6, 6, 1]
    protected override int[] InputShape => [8, 8, 1]; // 3D HWC (Forward adds batch)
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}

public class ConditionalRandomFieldLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new ConditionalRandomFieldLayer<double>(numClasses: 3, sequenceLength: 4,
            scalarActivation: new IdentityActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [4, 3]; // [seqLen, numClasses]
    // CRF uses Viterbi decoding producing discrete class labels — constant inputs give same labels
    protected override bool ExpectsDifferentOutputForConstantInputs => false;
}

public class SwinPatchEmbeddingLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new SwinPatchEmbeddingLayer<double>(inputHeight: 8, inputWidth: 8, inputChannels: 1,
            patchSize: 4, embedDim: 16);
    protected override int[] InputShape => [1, 1, 8, 8];
}

public class SwinTransformerBlockLayerTests : LayerTestBase
{
    // SwinBlock expects [batch, seqLen, dim] where seqLen = H*W, H divisible by windowSize
    protected override ILayer<double> CreateLayer()
        => new SwinTransformerBlockLayer<double>(dim: 8, numHeads: 2, windowSize: 2);
    protected override int[] InputShape => [1, 16, 8]; // [batch, seqLen=16 (4x4), dim=8]
}

public class SpatialTransformerLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new SpatialTransformerLayer<double>(inputHeight: 4, inputWidth: 4, outputHeight: 4, outputWidth: 4,
            activationFunction: new TanhActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [1, 1, 4, 4];
}
