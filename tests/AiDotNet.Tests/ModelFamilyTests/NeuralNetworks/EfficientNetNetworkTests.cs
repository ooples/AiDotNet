using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Paper-faithful invariant tests for EfficientNet-B0 per Tan &amp; Le 2019,
/// "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019.
/// </summary>
/// <remarks>
/// Default ctor instantiates EfficientNet-B0 with the ImageNet-1k classification
/// head (NumClasses = 1000), per paper Table 1. OutputShape mirrors that
/// contract — overriding to a smaller class count would not match the
/// paper-faithful default model.
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
public class EfficientNetNetworkTests : NeuralNetworkModelTestBase
{
    protected override int[] InputShape => [3, 64, 64];
    protected override int[] OutputShape => [1000];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new EfficientNetNetwork<double>();
}
