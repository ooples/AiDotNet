using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class AveragePoolingLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new AveragePoolingLayer<double>(inputShape: [1, 4, 4], poolSize: 2, strides: 2);
    protected override int[] InputShape => [1, 1, 4, 4];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}

public class AdaptiveAveragePoolingLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new AdaptiveAveragePoolingLayer<double>(inputChannels: 1, inputHeight: 4, inputWidth: 4);
    protected override int[] InputShape => [1, 1, 4, 4];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}

public class GlobalPoolingLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new GlobalPoolingLayer<double>(inputShape: [1, 4, 4], poolingType: PoolingType.Average);
    protected override int[] InputShape => [1, 1, 4, 4];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}

public class UpsamplingLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new UpsamplingLayer<double>(inputShape: [1, 2, 2], scaleFactor: 2);
    protected override int[] InputShape => [1, 1, 2, 2];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}

public class PixelShuffleLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new PixelShuffleLayer<double>(inputShape: [4, 2, 2], upscaleFactor: 2);
    // PixelShuffle needs channels = upscale^2 * output_channels
    protected override int[] InputShape => [1, 4, 2, 2];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}
