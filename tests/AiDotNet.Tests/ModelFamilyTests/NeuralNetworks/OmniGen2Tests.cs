using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.VisionLanguage.Unified;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for OmniGen2 (VisionLanguage.Unified) — the unified
/// understanding + generation VLM with a Phi-3 decoder backbone. The auto-generator
/// is told to skip OmniGen2 (<c>ExcludedClassNames</c>) so this hand-written scaffold
/// is authoritative.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why a reduced-scale config:</b> OmniGen2's paper defaults (VisionDim=1024,
/// DecoderDim=4096, 24 vision + 32 decoder layers, 32 heads, ImageSize=512,
/// VocabSize=32000) make it a multi-billion-parameter model. At that scale it trips
/// disk-backed weight streaming (so a forward no longer OOMs), but a single
/// backward+AdamW step still cannot complete within the 120s CI budget on CPU — the
/// cost is dominated by the parameter COUNT (image size / iteration count don't change
/// it), exactly the Janus-Pro / Helix situation.
/// </para>
/// <para>
/// These model-family invariants validate the <i>architecture's code paths</i>
/// (dual-path vision encoder + decoder wiring, patch embedding, attention/FFN, backprop,
/// optimizer step) — not paper-scale numerical behaviour. A smaller config exercises
/// every one of those paths in seconds while keeping the architecture's SHAPE faithful.
/// The dims below are scaled down ~8x (DecoderDim 4096→512, layers 32→4, etc.); the
/// model wiring is unchanged. Dropout is disabled so the memorization-based invariants
/// (strictly-decreasing loss, params-change) see clean monotonic convergence — those
/// tests validate the optimization path, not regularization.
/// </para>
/// </remarks>
public class OmniGen2Tests : VisionLanguageTestBase<float>
{
    // Reduced input: (64/16)^2 visual tokens through the patch embedder.
    // VisionLanguageModelBase's contract is [batch, channels=3, height, width].
    protected override int[] InputShape => [1, 3, 64, 64];

    protected override INeuralNetworkModel<float> CreateNetwork()
    {
        // Architecture image dims must match the options' ImageSize so the vision
        // encoder's patch embedder sees the expected spatial extent.
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.ImageClassification,
            inputHeight: 64,
            inputWidth: 64,
            inputDepth: 3,
            outputSize: 512);

        // Reduced-scale config (see <remarks>): same dual-path architecture shape as the
        // multi-billion-parameter paper model, ~8x smaller dims so all invariants fit the
        // CI budget. NumHeads (8) divides both VisionDim (256) and DecoderDim (512).
        var options = new OmniGen2Options
        {
            ImageSize = 64,
            VisionDim = 256,
            DecoderDim = 512,
            NumVisionLayers = 4,
            NumDecoderLayers = 4,
            NumHeads = 8,
            VocabSize = 1000,
            NumVisualTokens = 256,
            DropoutRate = 0.0,
        };
        return new OmniGen2<float>(architecture, options);
    }
}
