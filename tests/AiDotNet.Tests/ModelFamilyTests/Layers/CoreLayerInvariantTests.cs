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
    protected override ILayer<double> CreateLayer() => new FullyConnectedLayer<double>(4);
}

public sealed class ActivationLayerInvariantTests : LayerTestBase<double>
{
    protected override int[] InputShape => [1, 4];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
    protected override ILayer<double> CreateLayer() =>
        new ActivationLayer<double>((IActivationFunction<double>)new TanhActivation<double>());
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

public sealed class FlattenLayerInvariantTests : LayerTestBase<double>
{
    protected override int[] InputShape => [2, 3, 4];
    protected override bool ExpectsTrainableParameters => false;
    protected override bool ExpectsNonZeroGradients => false;
    protected override ILayer<double> CreateLayer() => new FlattenLayer<double>();
}
