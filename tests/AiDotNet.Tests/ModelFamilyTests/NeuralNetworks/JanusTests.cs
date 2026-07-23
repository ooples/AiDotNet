using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.VisionLanguage.Unified;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for Janus (Wu et al., DeepSeek 2024, "Janus:
/// Decoupling Visual Encoding for Unified Multimodal Understanding and
/// Generation", arXiv:2410.13848). The auto-generator is told to skip Janus
/// (<c>ExcludedClassNames</c>) so this hand-written scaffold is authoritative.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why a reduced-scale config:</b> Janus's paper defaults (VisionDim=1024,
/// DecoderDim=2048, 24+24 layers, 16 heads, ImageSize=384, VocabSize=32000,
/// NumVisualTokens=8192) make it a ~1.5B-parameter model whose weight
/// allocation + backward+optimizer step are dominated by the parameter COUNT
/// and cannot fit the 120s CI budget on CPU without a GPU.
/// </para>
/// <para>
/// These model-family invariants validate the <i>architecture's code paths</i>
/// (decoupled vision encoding, attention/FFN wiring, backprop, optimizer step,
/// clone) — not paper-scale numerical behaviour. A smaller config exercises
/// every one of those paths in seconds while keeping the architecture's SHAPE
/// faithful; the dims below are scaled down ~4-8x, the wiring is unchanged.
/// </para>
/// </remarks>
public class JanusTests : VisionLanguageTestBase<float>
{
    // Reduced input: (64/16)^2 = 16 visual tokens through the patch embedder.
    // VisionLanguageModelBase's contract is [batch, channels=3, height, width].
    protected override int[] InputShape => [1, 3, 64, 64];

    protected override INeuralNetworkModel<float> CreateNetwork()
    {
        // Architecture image dims must match the options' ImageSize so the
        // vision encoder's patch embedder sees the expected spatial extent.
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.ImageClassification,
            inputHeight: 64,
            inputWidth: 64,
            inputDepth: 3,
            outputSize: 512);

        // Reduced-scale config (see <remarks>): same architecture shape as the
        // ~1.5B paper model, ~4-8x smaller dims so all invariants fit the CI budget.
        var options = new JanusOptions
        {
            ImageSize = 64,
            VisionDim = 256,
            DecoderDim = 512,
            NumVisionLayers = 4,
            NumDecoderLayers = 4,
            NumHeads = 8,
            VocabSize = 1000,
            NumVisualTokens = 256,
            // Dropout is a regularizer that intentionally perturbs each training
            // forward; the memorization-based invariants (MoreData / strictly-
            // decreasing loss) need clean, monotonic convergence, so disable it
            // here — these tests validate the optimization path, not regularization.
            DropoutRate = 0.0,
        };
        return new Janus<float>(architecture, options);
    }
}
