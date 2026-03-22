using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class Conv3DLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new Conv3DLayer<double>(inputChannels: 1, outputChannels: 2, kernelSize: 3,
            inputDepth: 4, inputHeight: 4, inputWidth: 4, stride: 1, padding: 0,
            activationFunction: new AiDotNet.ActivationFunctions.ReLUActivation<double>() as IActivationFunction<double>);
    // 3D conv expects [batch, channels, depth, height, width]
    protected override int[] InputShape => [1, 1, 4, 4, 4];
}

public class MaxPool3DLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new MaxPool3DLayer<double>(inputShape: [1, 4, 4, 4], poolSize: 2, stride: 2);
    protected override int[] InputShape => [1, 1, 4, 4, 4];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}

public class Upsample3DLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new Upsample3DLayer<double>(inputShape: [1, 2, 2, 2], scaleFactor: 2);
    protected override int[] InputShape => [1, 1, 2, 2, 2];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}
