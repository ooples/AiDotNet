using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.VisionLanguage.Generative;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for Emu (Sun et al. 2023, "Generative Pretraining in
/// Multimodality", arXiv:2307.05222). The auto-generator is told to skip Emu
/// (<c>ExcludedClassNames</c>) so this hand-written scaffold is authoritative.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why a reduced-scale config:</b> Emu's production defaults (EVA-CLIP vision
/// tower VisionDim=1408 × 39 layers, decoder DecoderDim=4096 × 32 layers) make it
/// a multi-billion-parameter generative VLM whose forward+backward+optimizer step
/// cannot fit the 120s CI budget on CPU (the auto-generated invariants — including
/// ScaledInput — time out at 120000ms).
/// </para>
/// <para>
/// These model-family invariants validate the <i>architecture's code paths</i>
/// (EVA-CLIP vision encoder with its affine input projection, multimodal decoder,
/// regression head, backprop, optimizer step, clone) — not paper-scale numerical
/// behaviour. A smaller config exercises every one of those paths in seconds while
/// keeping the architecture's SHAPE faithful; only the dims shrink.
/// </para>
/// </remarks>
public class EmuTests : VisionLanguageTestBase<float>
{
    protected override int[] InputShape => [1, 3, 64, 64];

    protected override INeuralNetworkModel<float> CreateNetwork()
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.ImageClassification,
            inputHeight: 64,
            inputWidth: 64,
            inputDepth: 3,
            outputSize: 128);

        // Reduced-scale config (see <remarks>): same vision-tower + decoder +
        // regression-head architecture shape as the paper model, far smaller dims.
        // VisionDim == DecoderDim so the vision→decoder projection layer is elided
        // exactly as in the production factory when the two dims match.
        var options = new EmuOptions
        {
            ImageSize = 64,
            VisionDim = 128,
            DecoderDim = 128,
            RegressionDim = 128,
            NumVisionLayers = 4,
            NumDecoderLayers = 4,
            NumRegressionLayers = 2,
            NumHeads = 8,
            VocabSize = 1000,
            DropoutRate = 0.0,
        };
        return new Emu<float>(architecture, options);
    }
}
