using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class MemoryReadLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new MemoryReadLayer<double>(inputDimension: 4, memoryDimension: 8, outputDimension: 4,
            activationFunction: new AiDotNet.ActivationFunctions.TanhActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [4];
}

public class MemoryWriteLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new MemoryWriteLayer<double>(inputDimension: 4, memoryDimension: 8,
            activationFunction: new AiDotNet.ActivationFunctions.TanhActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [4];
}
