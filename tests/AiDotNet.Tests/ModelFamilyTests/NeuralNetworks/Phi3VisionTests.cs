using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.VisionLanguage.InstructionTuned;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for Phi-3-Vision (Abdin et al. 2024, "Phi-3
/// Technical Report"). The auto-generator can't emit this scaffold
/// because Phi3Vision's constructors require either an ONNX model file
/// or an explicit <see cref="NeuralNetworkArchitecture{T}"/> +
/// <see cref="Phi3VisionOptions"/>; neither satisfies the
/// parameterless-ctor or all-defaulted-ctor rule.
/// </summary>
/// <remarks>
/// Paper-faithful configuration: <see cref="Phi3VisionOptions"/>'s own
/// defaults reflect the small Phi-3-Vision variant (VisionDim=1024,
/// DecoderDim=3072, ProjectionDim=3072, NumVisionLayers=24,
/// NumDecoderLayers=32, NumHeads=32, ImageSize=336, MaxVisualTokens=576,
/// MLPProjection architecture). Do not override them — per the
/// project's never-shrink-tests rule, slow or saturating tests at
/// paper scale are model-side performance bugs to fix in the model
/// code, not papered over here.
/// </remarks>
// Float precision: Phi-3-Vision at paper scale is ~3.9B parameters (VisionDim=1024,
// DecoderDim=3072, FFN=12288, 24 vision + 32 decoder layers). A single forward is
// ~4.7 TFLOP; in double on CPU that exceeds the 120s test budget even though the
// underlying layers already use optimal fused GEMM/SDPA kernels. The paper itself
// runs in fp16/bf16, never double, so float is both faster and more paper-faithful.
// #1706: Phi-3-Vision is a ~3.9B foundation VLM. Even in float with whole-machine cores
// (FoundationScaleSerial), the Train-based and generation tests (Metadata_ShouldExist,
// ImageOnly_ShouldProduceOutput) run a full forward+backward / multi-token generation that is
// inherently >120s under the suite's single-threaded determinism BLAS — not a regression and not
// shrinkable (never-shrink rule, see <remarks>). Tag HeavyTimeout so the class is excluded from the
// default gate and runs full-fidelity in the nightly heavy lane (deferred, not skipped — it
// graduates back once the 3.9B forward is fast enough). The separate WeightRegistry-leak failures
// (ForwardPass/ScaledInput/Clone) are fixed by ResetsWeightStreamingBetweenTests below, so the
// nightly lane no longer sees the spurious "existing streaming pool has N registered entries".
[Trait("Category", "HeavyTimeout")]
[Collection("FoundationScaleSerial")] // dedicated cores (#1622 L4): serialized so its forward gets the whole machine
public class Phi3VisionTests : VisionLanguageTestBase<float>
{
    // Phi3Vision auto-enables weight streaming (foundation-scale), registering its weights with the
    // process-global WeightRegistry. Reset that registry between tests so the next Phi3Vision ctor
    // doesn't fail on leftover entries (#1706). Safe: the FoundationScaleSerial collection disables
    // parallelization, so nothing else runs concurrently with the reset.
    protected override bool ResetsWeightStreamingBetweenTests => true;

    // Paper-faithful image size (336×336 RGB per Phi-3-Vision §3 and
    // Phi3VisionOptions.ImageSize). VisionLanguageModelBase's contract
    // is [batch, channels=3, height, width].
    protected override int[] InputShape => [1, 3, 336, 336];

    protected override INeuralNetworkModel<float> CreateNetwork()
    {
        // Architecture's image dims must match Phi3VisionOptions's
        // ImageSize default (336) so the vision encoder's patch
        // embedder sees the expected spatial extent. OutputSize is
        // the next-token classification head's vocabulary slice.
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.ImageClassification,
            inputHeight: 336,
            inputWidth: 336,
            inputDepth: 3,
            outputSize: 512);

        // Defaults intentional — see <remarks> above.
        return new Phi3Vision<float>(architecture, new Phi3VisionOptions());
    }
}
