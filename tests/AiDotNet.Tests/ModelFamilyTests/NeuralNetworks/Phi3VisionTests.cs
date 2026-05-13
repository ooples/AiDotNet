using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.VisionLanguage.InstructionTuned;

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
public class Phi3VisionTests : VisionLanguageTestBase
{
    // Paper-faithful image size (336×336 RGB per Phi-3-Vision §3 and
    // Phi3VisionOptions.ImageSize). VisionLanguageModelBase's contract
    // is [batch, channels=3, height, width].
    protected override int[] InputShape => [1, 3, 336, 336];

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        // Architecture's image dims must match Phi3VisionOptions's
        // ImageSize default (336) so the vision encoder's patch
        // embedder sees the expected spatial extent. OutputSize is
        // the next-token classification head's vocabulary slice.
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.ImageClassification,
            inputHeight: 336,
            inputWidth: 336,
            inputDepth: 3,
            outputSize: 512);

        // Defaults intentional — see <remarks> above.
        return new Phi3Vision<double>(architecture, new Phi3VisionOptions());
    }
}
