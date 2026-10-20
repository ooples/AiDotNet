using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Layers;

/// <summary>
/// Subscribes the core layers to the bottom-up invariant harness in <see cref="LayerTestBase{T}"/>.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="LayerTestBase{T}"/> asserts eleven properties every layer must satisfy — finite forward
/// output, deterministic replay, input sensitivity, declared-vs-actual output shape, parameter
/// count/roundtrip consistency, serialization fidelity, state reset, and agreement between the taped
/// gradient and a numerical one. Its own documentation says subclasses "override CreateLayer() and all
/// invariant tests are inherited automatically".
/// </para>
/// <para>
/// Nothing subscribed to it. Before this file the only type deriving from the harness was the harness's
/// own regression fixture, using deliberately broken toy layers to prove the checks fire — so the eleven
/// invariants ran against zero real layers, while the README advertised the subsystem as stable on the
/// strength of "bottom-up invariant tests at 94% pass rate". A harness with no subscribers cannot have a
/// pass rate.
/// </para>
/// <para>
/// Each class below is the whole cost of covering one layer: name the layer, give it an input shape, and
/// declare whether it owns trainable parameters. Adding a layer to this file is the cheapest coverage in
/// the repository, and this set is a starting point rather than the finished job — the layer surface is
/// far larger than what is wired here.
/// </para>
/// </remarks>
public sealed class DenseLayerInvariantTests : LayerTestBase<double>
{
    protected override int[] InputShape => [1, 4];
    protected override ILayer<double> CreateLayer() => new DenseLayer<double>(4);
}

public sealed class DenseLayerWithReLUInvariantTests : LayerTestBase<double>
{
    protected override int[] InputShape => [1, 4];

    // ReLU is flat for negative inputs, so a numerical gradient probe that straddles zero disagrees with
    // the analytic one for reasons that are not a bug. Huber keeps the probe's loss smooth so the check
    // measures the layer instead of the kink.
    protected override GradientCheckLossStrategy DefaultLossStrategy => GradientCheckLossStrategy.Huber;

    // The activation classes implement IActivationFunction<T> and IVectorActivationFunction<T> alike, and
    // the layer offers a ctor for each, so an unqualified `new ReLUActivation<double>()` is ambiguous.
    // Naming the interface picks the scalar overload.
    protected override ILayer<double> CreateLayer() =>
        new DenseLayer<double>(4, (IActivationFunction<double>)new ReLUActivation<double>());
}

public sealed class FullyConnectedLayerInvariantTests : LayerTestBase<double>
{
    protected override int[] InputShape => [1, 4];
    protected override ILayer<double> CreateLayer() => new FullyConnectedLayer<double>(4, (IActivationFunction<double>)new ReLUActivation<double>());
}

public sealed class ActivationLayerInvariantTests : LayerTestBase<double>
{
    protected override int[] InputShape => [1, 4];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
    protected override ILayer<double> CreateLayer() =>
        new ActivationLayer<double>((IActivationFunction<double>)new TanhActivation<double>());
}

// Attentional feature fusion (CLAP's HTS-AT fusion, #2313). Its dual-tensor API keeps the generated harness away, and
// the single-input forward fuses the input with itself, so the invariants still apply here.
public sealed class AttentionalFeatureFusionInvariantTests : LayerTestBase<double>
{
    protected override int[] InputShape => [2, 8, 4, 4];
    protected override ILayer<double> CreateLayer() => new AttentionalFeatureFusionLayer<double>(8, 4);

    // With one input the layer fuses the map with itself: x - residual is exactly zero, so the attention weight
    // multiplies zero and no fusion parameter can receive a gradient. That is the math, not a tape break;
    // ClapArchitectureTests.AttentionalFeatureFusion_DistinctMaps_GradientsReachEveryParameter checks the gradients
    // with two different maps.
    protected override bool ExpectsNonZeroGradients => false;
}

public sealed class LayerNormalizationInvariantTests : LayerTestBase<double>
{
    protected override int[] InputShape => [1, 4];
    protected override ILayer<double> CreateLayer() => new LayerNormalizationLayer<double>();
}

public sealed class RMSNormalizationInvariantTests : LayerTestBase<double>
{
    protected override int[] InputShape => [1, 4];
    protected override ILayer<double> CreateLayer() => new RMSNormalizationLayer<double>();
}

public sealed class BatchNormalizationInvariantTests : LayerTestBase<double>
{
    // Batch norm normalises across the batch, so a single row leaves zero variance and the layer would be
    // measured on a degenerate input. Four rows give it something real to normalise.
    protected override int[] InputShape => [4, 4];
    protected override ILayer<double> CreateLayer() => new BatchNormalizationLayer<double>();
}

public sealed class TiedEmbeddingHeadLayerInvariantTests : LayerTestBase<double>
{
    // The head projects a hidden vector onto the vocabulary through the embedding's own table, so it owns no
    // parameters and computes no weight gradients of its own: both belong to the embedding it is tied to.
    //
    // That table is the embedding's state, saved with the embedding by the owning network, not with the head.
    // Every head this fixture creates is therefore tied to the same embedding, as two heads restored into one
    // network would be; a fresh random embedding per head would compare two different tables, not a round trip.
    private readonly EmbeddingLayer<double> _embedding = new(vocabularySize: 6, embeddingDimension: 4);

    protected override int[] InputShape => [1, 4];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
    protected override ILayer<double> CreateLayer() => new TiedEmbeddingHeadLayer<double>(_embedding);
}

public sealed class FlattenLayerInvariantTests : LayerTestBase<double>
{
    protected override int[] InputShape => [2, 3, 4];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
    protected override ILayer<double> CreateLayer() => new FlattenLayer<double>();
}

// Layers whose forward takes a second tensor: the harness feeds a fixed companion of the given shape and perturbs only
// the primary input.
public sealed class ConditionalLayerNormalizationInvariantTests : LayerTestBase<double>
{
    // Two sequences of three steps; the conditioning vector is one per sequence.
    protected override int[] InputShape => [2, 3, 4];
    protected override int[][] CompanionInputShapes(int[] primaryShape) => [[primaryShape[0], 3]];
    protected override ILayer<double> CreateLayer() => new ConditionalLayerNormalizationLayer<double>(4, 3);
}

public sealed class GRUCellInvariantTests : LayerTestBase<double>
{
    protected override int[] InputShape => [2, 4];
    protected override int[][] CompanionInputShapes(int[] primaryShape) => [[primaryShape[0], 3]];
    protected override ILayer<double> CreateLayer() => new GRUCellLayer<double>(4, 3);
}

public sealed class LSTMCellInvariantTests : LayerTestBase<double>
{
    protected override int[] InputShape => [2, 4];
    // The state is the hidden and cell state side by side, [B, 2·hidden].
    protected override int[][] CompanionInputShapes(int[] primaryShape) => [[primaryShape[0], 6]];
    protected override ILayer<double> CreateLayer() => new LSTMCellLayer<double>(4, 3);
}

public sealed class PostNormTransformerDecoderBlockInvariantTests : LayerTestBase<double>
{
    protected override int[] InputShape => [1, 3, 8];
    // The encoder memory: five steps of the same width.
    protected override int[][] CompanionInputShapes(int[] primaryShape) => [[primaryShape[0], 5, 8]];
    protected override ILayer<double> CreateLayer() => new PostNormTransformerDecoderBlock<double>(8, 2, 16, 0.0);
}
