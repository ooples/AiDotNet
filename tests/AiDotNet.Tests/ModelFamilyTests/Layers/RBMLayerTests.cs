using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class RBMLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new RBMLayer<double>(visibleUnits: 4, hiddenUnits: 8,
            new SigmoidActivation<double>() as IActivationFunction<double>);

    protected override int[] InputShape => [1, 4];
}
