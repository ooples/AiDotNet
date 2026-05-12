using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.VisionLanguage.InstructionTuned;

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
public class SmolVLMTests : VisionLanguageTestBase
{
    // Paper-faithful image size (384×384 RGB per SmolVLM cards and
    // SmolVLMOptions.ImageSize). VisionLanguageModelBase's contract
    // is [batch, channels=3, height, width].
    protected override int[] InputShape => [1, 3, 384, 384];

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        // Architecture's image dims must match SmolVLMOptions.ImageSize
        // (384) so the vision encoder's patch embedder sees the
        // expected spatial extent. OutputSize is the next-token
        // classification head's vocabulary slice.
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.ImageClassification,
            inputHeight: 384,
            inputWidth: 384,
            inputDepth: 3,
            outputSize: 512);

        // Defaults intentional — see <remarks> above.
        return new SmolVLM<double>(architecture, new SmolVLMOptions());
    }
}
