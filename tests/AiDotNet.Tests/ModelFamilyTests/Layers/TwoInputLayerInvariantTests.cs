using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

// Two-input layers added with the TTS paper models, run through the dual-input invariant harness (finite and deterministic
// output, sensitivity to the primary input, the parameter round trip and state reset).

/// <summary>Conditional LayerNorm: features [batch, time, hidden] and a condition [batch, conditionSize].</summary>
public sealed class ConditionalLayerNormalizationLayerInvariantTests : DualInputLayerTestBase<double>
{
    protected override int[] PrimaryInputShape => [1, 3, 8];
    protected override int[] SecondaryInputShape => [1, 4];
    protected override ILayer<double> CreateLayer() => new ConditionalLayerNormalizationLayer<double>(8, 4);
}

/// <summary>GRU cell: the step input [batch, input] and the state [batch, hidden].</summary>
public sealed class GRUCellLayerInvariantTests : DualInputLayerTestBase<double>
{
    protected override int[] PrimaryInputShape => [1, 4];
    protected override int[] SecondaryInputShape => [1, 6];
    protected override ILayer<double> CreateLayer() => new GRUCellLayer<double>(4, 6);
}

/// <summary>LSTM cell: the step input [batch, input] and the state [h; c], [batch, 2 hidden].</summary>
public sealed class LSTMCellLayerInvariantTests : DualInputLayerTestBase<double>
{
    protected override int[] PrimaryInputShape => [1, 4];
    protected override int[] SecondaryInputShape => [1, 12];
    protected override ILayer<double> CreateLayer() => new LSTMCellLayer<double>(4, 6);
}

/// <summary>Post-norm Transformer decoder block: the target sequence [batch, length, hidden] and the encoder memory.</summary>
public sealed class PostNormTransformerDecoderBlockInvariantTests : DualInputLayerTestBase<double>
{
    protected override int[] PrimaryInputShape => [1, 3, 8];
    protected override int[] SecondaryInputShape => [1, 5, 8];
    protected override ILayer<double> CreateLayer() => new PostNormTransformerDecoderBlock<double>(8, 2, 16, dropoutRate: 0.0);
}
