using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

public class SequenceLastLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new SequenceLastLayer<double>(featureSize: 4);
    // SequenceLastLayer expects [batch, seq, features] or similar
    protected override int[] InputShape => [2, 4];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}

public class PositionalEncodingLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new PositionalEncodingLayer<double>(maxSequenceLength: 8, embeddingSize: 4);
    protected override int[] InputShape => [1, 8, 4]; // match maxSeqLen for OutputShape consistency
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}

public class MaskingLayerTests : LayerTestBase
{
    protected override ILayer<double> CreateLayer()
        => new MaskingLayer<double>(inputShape: [1, 4], maskValue: 0.0);
    protected override int[] InputShape => [1, 4];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}

public class ReshapeLayerTests : LayerTestBase
{
    // ReshapeLayer assumes batch dim at [0], so input [1, 2, 3] → output [1, 6]
    protected override ILayer<double> CreateLayer()
        => new ReshapeLayer<double>(inputShape: [2, 3], outputShape: [6]);
    protected override int[] InputShape => [1, 2, 3];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
}
