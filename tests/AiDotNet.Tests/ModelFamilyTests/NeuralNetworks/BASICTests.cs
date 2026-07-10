using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.VisionLanguage.Encoders;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for BASIC (Pham et al. 2022, "Combined Scaling for
/// Zero-shot Transfer Learning", arXiv:2111.10050). The auto-generator is told
/// to skip BASIC (<c>ExcludedClassNames</c>) so this hand-written scaffold is
/// authoritative.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why a reduced-scale config:</b> BASIC scales up the CLIP/ALIGN contrastive
/// framework — its production defaults (VisionEmbeddingDim=1536, 24 vision layers,
/// TextEmbeddingDim=768, 12 text layers) make it a multi-billion-parameter dual
/// encoder whose forward+backward+optimizer step cannot fit the 120s CI budget on
/// CPU (the auto-generated training invariants time out at 120000ms).
/// </para>
/// <para>
/// These model-family invariants validate the <i>architecture's code paths</i>
/// (CoAtNet-style CNN→transformer vision tower, transformer text tower, the
/// affine input projection that keeps the encoder scale-sensitive, contrastive
/// projection, backprop, optimizer step, clone) — not paper-scale numerical
/// behaviour. A ~12x smaller config exercises every one of those paths in seconds
/// while keeping the architecture's SHAPE faithful; only the dims shrink.
/// </para>
/// </remarks>
public class BASICTests : VisionLanguageTestBase<float>
{
    // VisionLanguageModelBase's contract is [batch, channels=3, height, width].
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

        // Reduced-scale config (see <remarks>): same dual-encoder architecture shape
        // as the paper model, ~12x smaller dims so all invariants fit the CI budget.
        var options = new BASICOptions
        {
            ImageSize = 64,
            VisionEmbeddingDim = 128,
            NumVisionLayers = 4,
            NumVisionHeads = 8,
            TextEmbeddingDim = 128,
            NumTextLayers = 2,
            NumTextHeads = 8,
            ProjectionDim = 128,
            VocabSize = 1000,
            // Dropout is a regularizer that intentionally perturbs each training
            // forward; the memorization-based invariants need clean monotonic
            // convergence, so disable it — these tests validate the optimization
            // path, not regularization.
            DropoutRate = 0.0,
        };
        return new BASIC<float>(architecture, options);
    }
}
