using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.VisionLanguage.Unified;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for Janus-Pro (Chen et al., DeepSeek 2025,
/// "Janus-Pro: Unified Multimodal Understanding and Generation with Data and
/// Model Scaling", arXiv:2501.17811). The auto-generator is told to skip
/// Janus-Pro (<c>ExcludedClassNames</c>) so this hand-written scaffold is
/// authoritative.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why a reduced-scale config:</b> Janus-Pro's paper defaults
/// (VisionDim=1024, DecoderDim=4096, 24+32 layers, 32 heads, ImageSize=384,
/// VocabSize=32000, NumVisualTokens=16384) make it a ~7B-parameter model.
/// Measured on a 32-core CPU, that's ~35s just to allocate+initialize the
/// weights and &gt;76s for a single backward+Adam step — the cost is dominated
/// by the parameter COUNT (image size / iteration count don't change it), so
/// the paper-scale model fundamentally cannot train inside the 120s CI budget
/// without a GPU.
/// </para>
/// <para>
/// These model-family invariants validate the <i>architecture's code paths</i>
/// (decoupled vision encoders + VQ head, attention/FFN wiring, backprop,
/// optimizer step, clone) — not paper-scale numerical behaviour. A smaller
/// config exercises every one of those paths in seconds while keeping the
/// architecture's SHAPE faithful (decoupled understanding/generation decoders,
/// patch embedding, vocab+codebook head). The dims below are scaled down ~8x
/// (DecoderDim 4096→512, layers 32→4, etc.); the model wiring is unchanged.
/// </para>
/// </remarks>
public class JanusProTests : VisionLanguageTestBase<float>
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
        // ~7B paper model, ~8x smaller dims so all invariants fit the CI budget.
        var options = new JanusProOptions
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
        return new JanusPro<float>(architecture, options);
    }
}
