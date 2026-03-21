using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class ReLUActivationLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new ActivationLayer<double>([4], new ReLUActivation<double>() as IActivationFunction<double>);

    protected override int[] InputShape => [1, 4];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}

public class SigmoidActivationLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new ActivationLayer<double>([4], new SigmoidActivation<double>() as IActivationFunction<double>);

    protected override int[] InputShape => [1, 4];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}

public class TanhActivationLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new ActivationLayer<double>([4], new TanhActivation<double>() as IActivationFunction<double>);

    protected override int[] InputShape => [1, 4];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}
