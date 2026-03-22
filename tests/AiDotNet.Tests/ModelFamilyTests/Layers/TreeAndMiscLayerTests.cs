using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class SoftTreeLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new SoftTreeLayer<double>(inputDim: 4, depth: 3, outputDim: 2);
    protected override int[] InputShape => [1, 4];
}

public class TimeDistributedLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new TimeDistributedLayer<double>(
            new DenseLayer<double>(4, 8),
            activationFunction: new ReLUActivation<double>() as IActivationFunction<double>,
            inputShape: [2, 4]);
    // TimeDistributed expects [batch, timesteps, features]
    protected override int[] InputShape => [1, 2, 4];
}

public class SynapticPlasticityLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new SynapticPlasticityLayer<double>(size: 4);
    protected override int[] InputShape => [4];
}

public class RepParameterizationLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new RepParameterizationLayer<double>(inputShape: [1, 4, 4]);
    protected override int[] InputShape => [1, 1, 4, 4];
}

public class AnomalyDetectorLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new AnomalyDetectorLayer<double>(inputSize: 4, anomalyThreshold: 2.0);
    protected override int[] InputShape => [4];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
    // AnomalyDetector uses statistical thresholding — constant inputs both fall within/outside threshold
    protected override bool ExpectsDifferentOutputForConstantInputs => false;
}

public class MeanLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new MeanLayer<double>(inputShape: [2, 4], axis: 0);
    protected override int[] InputShape => [2, 4];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
    // Mean along axis produces same output for constant inputs (mean of identical values = that value)
    // But different constant values DO produce different means
}

public class LogVarianceLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new LogVarianceLayer<double>(inputShape: [2, 4], axis: 0);
    protected override int[] InputShape => [2, 4];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
    // LogVariance of constant = -Inf (variance = 0), but different constants give same -Inf
    protected override bool ExpectsDifferentOutputForConstantInputs => false;
}

public class SplitLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new SplitLayer<double>(inputShape: [4], numSplits: 2);
    protected override int[] InputShape => [1, 4];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}

public class ReconstructionLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new ReconstructionLayer<double>(
            inputDimension: 4, hidden1Dimension: 8, hidden2Dimension: 8, outputDimension: 4,
            hiddenActivation: new AiDotNet.ActivationFunctions.ReLUActivation<double>() as IActivationFunction<double>);
    protected override int[] InputShape => [1, 4];
}
