using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

/// <summary>
/// InteractingLayer (AutoInt's multi-head self-attention over feature fields) had no invariant subclass. Wiring it in
/// brings the uncovered count back under LayerInvariantCoverageTests' ceiling, which master had crossed (122 of 121).
/// </summary>
public sealed class InteractingLayerInvariantTests : LayerTestBase<double>
{
    // [batch, fields, embeddingDim].
    protected override int[] InputShape => [1, 3, 4];
    protected override ILayer<double> CreateLayer() => new InteractingLayer<double>(embeddingDim: 4, numHeads: 2);
}
