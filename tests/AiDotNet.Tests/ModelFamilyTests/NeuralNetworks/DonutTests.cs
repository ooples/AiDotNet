using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.VisionLanguage.Document;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for Donut (Kim et al. 2022, NAVER, "OCR-free Document
/// Understanding Transformer", arXiv:2111.15664). The auto-generator is told to skip
/// Donut (<c>ExcludedClassNames</c>) so this hand-written scaffold is authoritative.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why a reduced-scale config:</b> Donut's paper defaults (VisionDim=1024,
/// DecoderDim=1024, 12 vision + 4 decoder layers, NumHeads=16, ImageSize=2560) make a
/// single AdamW train step ~9s on CPU, so the training-invariant counts (Training 30,
/// Memorization 100, MoreData 50/200) overflow the 120s CI budget.
/// </para>
/// <para>
/// These model-family invariants validate the <i>architecture's code paths</i> (Swin
/// patch embedding, attention/FFN wiring, vision->decoder projection, backprop,
/// optimizer step, clone) — not paper-scale numerical behaviour. A smaller config
/// exercises every one of those paths in seconds while keeping the architecture's SHAPE
/// faithful; the dims below are scaled down ~4x and the wiring is unchanged. Dropout is
/// disabled so the memorization-based invariants see clean, monotonic convergence.
/// </para>
/// </remarks>
public class DonutTests : VisionLanguageTestBase<float>
{
    // Reduced input: (64/16)^2 = 16 patch tokens through the Swin patch embedder.
    // VisionLanguageModelBase's contract is [batch, channels=3, height, width].
    protected override int[] InputShape => [1, 3, 64, 64];

    protected override INeuralNetworkModel<float> CreateNetwork()
    {
        // Architecture image dims must match the options' ImageSize so the Swin patch
        // embedder sees the expected spatial extent.
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 64,
            inputWidth: 64,
            inputDepth: 3,
            outputSize: 256);

        // Reduced-scale config (see <remarks>): same architecture shape as the paper
        // model, ~4x smaller dims so all invariants fit the CI budget. VisionDim ==
        // DecoderDim keeps the encoder/decoder dims aligned (no extra projection layer).
        var options = new DonutOptions
        {
            ImageSize = 64,
            VisionDim = 256,
            DecoderDim = 256,
            NumVisionLayers = 2,
            NumDecoderLayers = 2,
            NumHeads = 8,
            VocabSize = 1000,
            // Memorization-based invariants (MoreData / strictly-decreasing loss) need
            // clean, monotonic convergence; dropout intentionally perturbs each training
            // forward, so disable it here — these tests validate the optimization path.
            DropoutRate = 0.0,
        };
        return new Donut<float>(architecture, options);
    }
}
