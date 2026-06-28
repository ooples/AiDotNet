using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.VisionLanguage.InstructionTuned;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for SmolVLM (HuggingFace, 2024 — tiny efficient
/// VLMs spanning 256M / 500M / 2.2B parameters). The auto-generator
/// can't emit this scaffold because SmolVLM's constructors require
/// either an ONNX model file or an explicit
/// <see cref="NeuralNetworkArchitecture{T}"/> + <see cref="SmolVLMOptions"/>;
/// neither path satisfies the parameterless-ctor rule.
/// </summary>
/// <remarks>
/// Paper-faithful configuration: <see cref="SmolVLMOptions"/>'s own
/// defaults reflect the 2.2B variant (VisionDim=384, DecoderDim=576,
/// ProjectionDim=576, NumVisionLayers=12, NumDecoderLayers=16,
/// NumHeads=9, ImageSize=384, MaxVisualTokens=256, MLPProjection
/// architecture, "SmolLM" language backbone). Do not override them —
/// per the project's never-shrink-tests rule, slow or saturating
/// tests at paper scale are model-side performance bugs to fix in
/// the model code, not papered over here.
/// </remarks>
// #1706: SmolVLM runs its full paper-scale 2.2B config (SmolVLMOptions defaults: VisionDim 384 /
// DecoderDim 576 / 12 vision + 16 decoder layers / 384×384 images). A single forward+backward under
// the tests' single-threaded determinism BLAS is ~104s — inherently heavy at paper scale, not a
// regression, and (per the never-shrink rule above) NOT something to smoke-scale here. Tag it
// HeavyTimeout so it is excluded from the default gate and runs full-fidelity in the nightly heavy
// lane (deferred, not skipped — it graduates back once the 2.2B forward is fast enough). #1305/#1706.
[Trait("Category", "HeavyTimeout")]
public class SmolVLMTests : VisionLanguageTestBase<float>
{
    // Serialize the 2.2B forward in the nightly heavy lane so it doesn't self-contend (#1706).
    protected override bool RequiresHeavySerialization => true;

    // SmolVLM is foundation-scale and auto-enables weight streaming, registering its weights with
    // the process-global WeightRegistry. Reset that registry around each test so a later streaming
    // model isn't blocked by SmolVLM's leftover entries (#1706). Safe: RequiresHeavySerialization
    // serializes it, so the reset never races another model's streaming forward.
    protected override bool ResetsWeightStreamingBetweenTests => true;

    // Paper-faithful image size (384×384 RGB per SmolVLM cards and
    // SmolVLMOptions.ImageSize). VisionLanguageModelBase's contract
    // is [batch, channels=3, height, width].
    protected override int[] InputShape => [1, 3, 384, 384];

    protected override INeuralNetworkModel<float> CreateNetwork()
    {
        // Architecture's image dims must match SmolVLMOptions.ImageSize
        // (384) so the vision encoder's patch embedder sees the
        // expected spatial extent. OutputSize is the next-token
        // classification head's vocabulary slice.
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.ImageClassification,
            inputHeight: 384,
            inputWidth: 384,
            inputDepth: 3,
            outputSize: 512);

        // Defaults intentional — see <remarks> above.
        return new SmolVLM<float>(architecture, new SmolVLMOptions());
    }
}
