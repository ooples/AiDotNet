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
/// <see cref="SmolVLMOptions"/>'s own defaults reflect the 2.2B variant (VisionDim=384,
/// DecoderDim=576, ProjectionDim=576, NumVisionLayers=12, NumDecoderLayers=16, NumHeads=9,
/// ImageSize=384, MaxVisualTokens=256, MLPProjection architecture, "SmolLM" language backbone).
/// The test shrinks the towers and the image (see below) and keeps the architecture, the patch
/// size and the vocabulary.
/// </remarks>
// #2087 / #1706: out of the HeavyTimeout lane. At its paper-scale 2.2B config (384x384 images, a
// 384-wide 12-layer vision encoder and a 576-wide 16-layer decoder) one forward and backward took
// ~104 s. Per the #1706 out-of-memory ladder the test caps the image at 96x96 with 16 visual tokens,
// which keeps the paper's 24-pixel patch, and shrinks both towers to 64 wide and 2 layers. The
// vocabulary and the pixel-shuffle projector keep their defaults.
public class SmolVLMTests : VisionLanguageTestBase<float>
{
    // Serialize the 2.2B forward in the nightly heavy lane so it doesn't self-contend (#1706).
    protected override bool RequiresHeavySerialization => true;

    // SmolVLM is foundation-scale and auto-enables weight streaming, registering its weights with
    // the process-global WeightRegistry. Reset that registry around each test so a later streaming
    // model isn't blocked by SmolVLM's leftover entries (#1706). Safe: RequiresHeavySerialization
    // serializes it, so the reset never races another model's streaming forward.
    protected override bool ResetsWeightStreamingBetweenTests => true;

    // VisionLanguageModelBase's contract is [batch, channels=3, height, width].
    private const int ImageSize = 96;

    protected override int[] InputShape => [1, 3, ImageSize, ImageSize];

    protected override INeuralNetworkModel<float> CreateNetwork()
    {
        // Architecture's image dims must match SmolVLMOptions.ImageSize
        // so the vision encoder's patch embedder sees the
        // expected spatial extent. OutputSize is the next-token
        // classification head's vocabulary slice.
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.ImageClassification,
            inputHeight: ImageSize,
            inputWidth: ImageSize,
            inputDepth: 3,
            outputSize: 512);

        return new SmolVLM<float>(architecture, new SmolVLMOptions
        {
            ImageSize = ImageSize,
            MaxVisualTokens = 16,
            VisionDim = 64,
            DecoderDim = 64,
            ProjectionDim = 64,
            NumVisionLayers = 2,
            NumDecoderLayers = 2,
            NumHeads = 2,
        });
    }
}
