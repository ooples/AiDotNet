using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

// DeconvolutionalLayer: inputShape format unclear, crashes in CalculateOutputShape
// LocallyConnectedLayer: input channel mismatch in engine Conv2D call
// TODO: Investigate correct input shapes for these layers

public class PatchEmbeddingLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new PatchEmbeddingLayer<double>(imageHeight: 8, imageWidth: 8, channels: 1, patchSize: 4,
            embeddingDim: 16);
    protected override int[] InputShape => [1, 1, 8, 8];
}

public class SqueezeAndExcitationLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new SqueezeAndExcitationLayer<double>(channels: 4, reductionRatio: 2,
            firstActivation: new ReLUActivation<double>() as IActivationFunction<double>,
            secondActivation: new SigmoidActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [1, 4, 4, 4];
}

public class SpectralNormalizationLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new SpectralNormalizationLayer<double>(new DenseLayer<double>(4, 8));
    protected override int[] InputShape => [1, 4];
}

public class CapsuleLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new CapsuleLayer<double>(
            inputCapsules: 4, inputDimension: 2, numCapsules: 2, capsuleDimension: 4,
            numRoutingIterations: 3);
    protected override int[] InputShape => [1, 4, 2];
}

public class PrimaryCapsuleLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new PrimaryCapsuleLayer<double>(
            inputChannels: 1, capsuleChannels: 2, capsuleDimension: 4,
            kernelSize: 3, stride: 1,
            scalarActivation: new ReLUActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [1, 1, 8, 8];
}
