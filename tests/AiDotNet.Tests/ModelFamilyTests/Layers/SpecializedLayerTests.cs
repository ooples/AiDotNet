using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class QuantumLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new QuantumLayer<double>(inputSize: 4, outputSize: 4, numQubits: 4);
    protected override int[] InputShape => [4];
}

public class MeasurementLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new MeasurementLayer<double>(size: 4);
    protected override int[] InputShape => [4];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}

// RBFLayer already tested in RBMLayerTests.cs — skip duplicate

public class GaussianNoiseLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new GaussianNoiseLayer<double>(inputShape: [4], standardDeviation: 0.1);
    protected override int[] InputShape => [1, 4];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}

public class SpatialPoolerLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new SpatialPoolerLayer<double>(inputSize: 4, columnCount: 8, sparsityThreshold: 0.02);
    protected override int[] InputShape => [4];
    protected override bool ExpectsNonZeroGradients => false;
}

public class TemporalMemoryLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new TemporalMemoryLayer<double>(columnCount: 4, cellsPerColumn: 2);
    protected override int[] InputShape => [4];
    protected override bool ExpectsNonZeroGradients => false;
}
