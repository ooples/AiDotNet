using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.VisionLanguage.ThreeD;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for GPT4Point (Qi et al. 2024, "GPT4Point: A Unified
/// Framework for Point-Language Understanding and Generation",
/// arXiv:2312.02980). The auto-generator is told to skip GPT4Point
/// (<c>ExcludedClassNames</c>) so this hand-written scaffold is authoritative.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why a reduced-scale config (Janus precedent).</b> GPT4Point's paper
/// defaults make it a ~6.7B-parameter point-language VLM (point-cloud encoder +
/// Q-Former bridge + LLM decoder DecoderDim=4096 × 32 layers). As with Helix, a
/// single full-model Adam step at that scale cannot complete in the 120 s CI
/// budget on CPU at any precision; the memory-bounded streaming training path
/// makes such a step possible where it would OOM, not unit-test-fast.
/// </para>
/// <para>
/// These invariants validate the <i>architecture's code paths</i> (point encoder,
/// Q-Former alignment, projection to LLM space, decoder, backprop, optimizer
/// step, clone) at a reduced float scale — same wiring, ~8× smaller dims — so
/// they fit the CI budget in seconds. The streaming training subsystem is
/// exercised by the dedicated streaming integration tests.
/// </para>
/// </remarks>
public class GPT4PointTests : VisionLanguageTestBase<float>
{
    // GPT4Point's native chain begins with the point-cloud encoder MHA, whose
    // dim is the helper's pointEncoderDim = 512 (hard-wired in InitializeLayers),
    // so it consumes [batch, num_tokens, 512] token features.
    protected override int[] InputShape => [1, 4, 512];

    // Flat forward ends at the LLM decoder width (reduced DecoderDim = 512).
    protected override int[] OutputShape => [1, 4, 512];

    protected override INeuralNetworkModel<float> CreateNetwork()
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 224,
            inputWidth: 224,
            inputDepth: 3,
            outputSize: 4);

        // Reduced-scale config (see <remarks>): same architecture shape as the
        // ~6.7B paper model, ~8× smaller decoder so all invariants fit the CI
        // budget. The point-encoder dim is fixed at 512 by the layer helper.
        var options = new GPT4PointOptions
        {
            DecoderDim = 512,
            NumVisionLayers = 4,
            NumDecoderLayers = 4,
            NumHeads = 8,
            DropoutRate = 0.0,
        };

        return new GPT4Point<float>(architecture, options);
    }
}
