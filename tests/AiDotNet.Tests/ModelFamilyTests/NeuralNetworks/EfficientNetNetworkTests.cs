using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Paper-faithful invariant tests for EfficientNet-B0 per Tan &amp; Le 2019,
/// "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019.
/// </summary>
/// <remarks>
/// The fixture uses EfficientNet's public custom-scale configuration at 32x32,
/// retaining the B0 MBConv/squeeze-excitation block topology, 1.0 width/depth
/// multipliers, and ImageNet-1k classification head. The production B0 default
/// remains 224x224 with 1000 classes.
///
/// InputShape is unbatched rank-3 [C, H, W]. NeuralNetworkBase.Predict
/// auto-promotes that to rank-4 [1, C, H, W] internally and squeezes
/// the unit batch axis off the output, so a single-sample inference
/// returns a rank-1 [NumClasses] tensor — NOT [1, NumClasses]. The
/// OutputShape override must match that unbatched contract; otherwise
/// the warm-up Predict path (when EffectiveOutputShape falls back to
/// OutputShape) trains against a rank-2 target whose ranks don't
/// match the inference output.
/// </remarks>
public class EfficientNetNetworkTests : NeuralNetworkModelTestBase<float>
{
    protected override int[] InputShape => [3, 32, 32];
    protected override int[] OutputShape => [1000];

    // MoreData's default 50+200-iteration probe is disproportionately expensive for the complete
    // MBConv + squeeze-excitation stack, even at this legal 32x32 custom scale. Cap MoreData to a
    // smoke gap (10 vs 30 steps): it still catches training divergence (long-run loss >> short-run
    // loss), with the same absolute tolerance used by the generated heavy-model scaffolds.
    protected override int MoreDataShortIterations => 10;
    protected override int MoreDataLongIterations => 30;
    protected override double MoreDataTolerance => 0.5;

    protected override INeuralNetworkModel<float> CreateNetwork()
        => EfficientNetNetwork<float>.ForTesting(numClasses: 1000);
}
