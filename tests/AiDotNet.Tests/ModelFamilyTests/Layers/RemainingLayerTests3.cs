using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class CroppingLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new CroppingLayer<double>(inputShape: [1, 6, 6],
            cropTop: [0, 1, 1], cropBottom: [0, 1, 1], cropLeft: [0, 1, 1], cropRight: [0, 1, 1],
            scalarActivation: new IdentityActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [1, 1, 6, 6];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}

public class ConditionalRandomFieldLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new ConditionalRandomFieldLayer<double>(numClasses: 3, sequenceLength: 4,
            scalarActivation: new IdentityActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [4, 3]; // [seqLen, numClasses]
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
    protected override ILayer<double> CreateLayer()
        => new SwinTransformerBlockLayer<double>(dim: 8, numHeads: 2, windowSize: 2);
    protected override int[] InputShape => [1, 4, 4, 8]; // [batch, H, W, C]
}

public class SpatialTransformerLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new SpatialTransformerLayer<double>(inputHeight: 4, inputWidth: 4, outputHeight: 4, outputWidth: 4,
            activationFunction: new TanhActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [1, 1, 4, 4];
}
